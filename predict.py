import sys
import tempfile
from pathlib import Path
import random
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as transforms
import clip
import wav2clip
import librosa
import pydiffvg
import cog
import youtube_dl

class Predictor(cog.Predictor):
    def setup(self):
        # Load the model
        torch.cuda.empty_cache()
        self.device = torch.device('cuda')

        # Load WaveCLIP model
        self.wav2clip_model = wav2clip.get_model().to(device)
        self.model, _ = clip.load('ViT-B/32', self.device, jit=False)

        pydiffvg.set_print_timing(False)
        # Use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        pydiffvg.set_device(self.device)
        # Image Augmentation Transformation, use_normalized_clip = True
        self.augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.gamma = 1.0

    @cog.input("prompt", type=str, default="https://www.youtube.com/watch?v=XdlIbNrki5o",
               help="youtube link to get the sound/music to draw the image")
    @cog.input("num_paths", type=int, default=256, help="number of paths/curves")
    @cog.input("num_iterations", type=int, default=1000, help="number of iterations")
    @cog.input("display_frequency", type=int, default=10, help="display frequency of intermediate images")
    def predict(self, prompt="",
                num_paths=256,
                num_iterations=1000,
                display_frequency=10):

        assert isinstance(num_paths, int) and num_paths > 0, 'num_paths should be an positive integer'
        assert isinstance(num_iterations, int) and num_iterations > 0, 'num_iterations should be an positive integer'

        out_path = Path(tempfile.mkdtemp()) / "out.png"

        # download music
        outname = './music.mp3'
        ydl_opts = {
            'cachedir': False,
            'outtmpl': outname,
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([prompt])

        audio_sampling_freq = 16000
        # Calculate sound features
        audio, sr = librosa.load(outname, sr=audio_sampling_freq)
        music_features = wav2clip.embed_audio(torch.from_numpy(audio), wav2clip_model)
        music_features = music_features.to(self.device)

        # ARGUMENTS. Feel free to play around with these, especially num_paths.
        args = lambda: None
        args.num_paths = num_paths
        args.num_iter = num_iterations
        args.max_width = 50

        canvas_width, canvas_height = 224, 224
        num_paths = args.num_paths
        max_width = args.max_width

        # Initialize Random Curves
        shapes = []
        shape_groups = []
        for i in range(num_paths):
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.1
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points=num_control_points, points=points,
                                 stroke_width=torch.tensor(1.0),
                                 is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                             stroke_color=torch.tensor(
                                                 [random.random(), random.random(), random.random(),
                                                  random.random()]))
            shape_groups.append(path_group)

        # Just some diffvg setup
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

        points_vars = []
        stroke_width_vars = []
        color_vars = []
        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)

        # Optimizers
        points_optim = torch.optim.Adam(points_vars, lr=1.0)
        width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
        color_optim = torch.optim.Adam(color_vars, lr=0.01)

        # Run the main optimization loop
        for t in range(args.num_iter):

            # Anneal learning rate (makes videos look cleaner)
            if t == int(args.num_iter * 0.5):
                for g in points_optim.param_groups:
                    g['lr'] = 0.4
            if t == int(args.num_iter * 0.75):
                for g in points_optim.param_groups:
                    g['lr'] = 0.1

            points_optim.zero_grad()
            width_optim.zero_grad()
            color_optim.zero_grad()
            scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
            img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                              device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
            img = img[:, :, :3]
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

            loss = 0
            NUM_AUGS = 4
            img_augs = []
            for n in range(NUM_AUGS):
                img_augs.append(self.augment_trans(img))
            im_batch = torch.cat(img_augs)
            image_features = self.model.encode_image(im_batch)
            for n in range(NUM_AUGS):
                loss -= torch.cosine_similarity(music_features, image_features[n:n + 1], dim=1)

            # Backpropagate the gradients.
            loss.backward()

            # Take a gradient descent step.
            points_optim.step()
            width_optim.step()
            color_optim.step()
            for path in shapes:
                path.stroke_width.data.clamp_(1.0, max_width)
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)
            if t % display_frequency == 0 or t == num_iterations - 1:
                yield checkin(img.detach().cpu().numpy()[0], t, loss, out_path)
        return out_path


@torch.no_grad()
def checkin(img, t, loss, out_path=None):
    sys.stderr.write(f"iteration: {t}, render:loss: {loss.item()}\n")
    save_img(img, str(out_path))
    return out_path


def save_img(img, file_name):
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 254)
    pimg = PIL.Image.fromarray(img, mode="RGB")
    pimg.save(file_name)
