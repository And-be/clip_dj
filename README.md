# Clip draw wave 

A repo for making a AI-generated draw based on a sound/music file with Wav2CLIP and CLIP.

Repo code based on [Clipdraw](https://github.com/kvfrans/clipdraw) 

The CLIP embedding for audio was from: [Wav2CLIP](https://github.com/descriptinc/lyrebird-wav2clip)

Differential rendering from clipdraw: [diffvg](https://github.com/BachiLi/diffvg/blob/master/apps/painterly_rendering.py)

CLIP from: [CLIP](https://github.com/openai/CLIP.git)

Colab playground [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ykExkuj_8WiZuap-gqB1JV3lErlVvUOl/view?usp=sharing)


## Sample

A sample of a drawing created with this sound input [birds singing](https://www.youtube.com/watch?v=XdlIbNrki5o)

![sample](https://user-images.githubusercontent.com/95462960/148018641-c6b8a3d1-78ad-46a9-8f9b-1fda4b4c1613.png)

You can make one with your own sound/music too

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{frans2021clipdraw,
      title={CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders}, 
      author={Kevin Frans and L. B. Soros and Olaf Witkowski},
      year={2021},
      eprint={2106.14843},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@article{wu2021wav2clip,
  title={Wav2CLIP: Learning Robust Audio Representations From CLIP},
  author={Wu, Ho-Hsiang and Seetharaman, Prem and Kumar, Kundan and Bello, Juan Pablo},
  journal={arXiv preprint arXiv:2110.11499},
  year={2021}
}
```
