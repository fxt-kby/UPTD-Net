<h1 align="center">UPTD-Net</h1>

<p align="center">
  Unsupervised Polarization-Guided Two-Stage Decoupling<br>
  for Underwater Low-Light Image Restoration
</p>

<p align="center">
  <a href="mailto:feixiaotong@tju.edu.cn">Xiaotong Fei</a> ·
  Linghao Shen ·
  Yizhao Huang ·
  Haofeng Hu ·
  Dagong Jia ·
  Binglu Wang
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.optlastec.2026.116479"><em>Optics &amp; Laser Technology</em>, 204 (2026) 116479</a>
  ·
  <a href="paper/UPTD-Net.pdf">PDF</a>
</p>

Underwater images taken in low light are degraded by scattering and insufficient illumination at the same time. UPTD-Net separates these two effects in sequence, without paired clean images or a pretrained model.

BD-DeNet first separates direct transmission from backscattered light. A polarization-guided step then refines the descattered image, and RI-DeNet separates degraded illumination from reflectance.

<p align="center">
  <img src="https://raw.githubusercontent.com/fxt-kby/UPTD-Net/main/Figs/flowchart.png" width="100%">
</p>

<p align="center">
  <em>BD-DeNet removes scattering. RI-DeNet recovers illumination. Polarization guides both stages.</em>
</p>

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/fxt-kby/UPTD-Net/main/Figs/result.png" width="100%">
</p>

## Video Restoration Demos

<p align="center">
  <img src="https://raw.githubusercontent.com/fxt-kby/UPTD-Net/main/Figs/video3_gif.gif" width="48%">
  <img src="https://raw.githubusercontent.com/fxt-kby/UPTD-Net/main/Figs/video1_gif.gif" width="48%">
</p>

<p align="center">
  <em>Demos of underwater video data restored using our proposed method.</em>
</p>

## Setup

Python 3.9 or newer, and an NVIDIA GPU. The networks are placed on CUDA.

```bash
pip install -r requirements.txt
```

Tested with PyTorch 2.5.1.

## Data

Images are grayscale BMP files. Each scene uses three folders, and every image is square. The side length must match `--down_size` (default 450).

```text
input/illu3/
  s0/s0_1.bmp       S0 intensity, named with _<index>
  dop/1.bmp         DoP map, named <index>.bmp
  s_perp/s0_1.bmp   perpendicular component, same name as the S0 file
```

`S0` and `DoP` are paired by the index. `S⊥` follows the `S0` file name.

`illu1` to `illu4` are four illumination levels of the laboratory data. Each level contains scenes `s0_1` to `s0_15`. The default command uses `illu3`. Pool images and the original 1224×1024 captures are not included.

## Run

```bash
python run.py
```

```bash
python run.py --datasets input/illu1/s0 --datasets2 input/illu1/dop --datasets3 input/illu1/s_perp --output_folder output/illu1
```

Stage 1 runs 301 iterations. Stage 2 runs `--epochs` iterations, 100 by default. One 450×450 image takes a few minutes on a desktop GPU.

| Output | Content |
| --- | --- |
| `output/` | restored image |
| `output/d/` | direct transmission |
| `output/d_enhanced/` | result after polarization-guided refinement |
| `output/b/` | backscattered light |
| `output/illu/` | estimated illumination |
| `output/lowpass/` | low-pass component from stage 1 |
| `output/eme_log.txt` | EME of each image |

## Settings

| | |
| --- | --- |
| Stage 1 learning rates | 0.001 for D-Net, 0.0001 for B-Net |
| Stage 1 loss weights | 1, 1, 1 for transmission; 5, 1, 1 for backscatter |
| Stage 2 | 100 epochs, learning rate `1e-5` |
| Stage 2 weights | `--alpha 1` `--beta 20` `--gamma 10` `--delta 5` |
| Context window | `--window 5` |
| Illumination reference | 0.5, written as `It = It * 0.5` |
| Refinement strengths | detail 1, brightness 0.8 |
| Scale `c` | starts at 1, kept inside [1, 2] |

For the pool scenes, the backscatter low-frequency weight in `total_lossB` was set to 10 instead of 5. Some scenes also used 0.6 or 0.7 in place of 0.5.

## Citation

```bibtex
@article{fei2026uptdnet,
  title   = {UPTD-Net: Unsupervised Polarization-Guided Two-Stage Decoupling for Underwater Low-Light Image Restoration},
  author  = {Fei, Xiaotong and Shen, Linghao and Huang, Yizhao and Hu, Haofeng and Jia, Dagong and Wang, Binglu},
  journal = {Optics \& Laser Technology},
  volume  = {204},
  pages   = {116479},
  year    = {2026},
  doi     = {10.1016/j.optlastec.2026.116479}
}
```

## Acknowledgements

We thank the authors of [CoLIE](https://github.com/ctom2/colie) (Chobola et al., ECCV 2024).

```bibtex
@inproceedings{chobola2024fast,
  title     = {Fast Context-Based Low-Light Image Enhancement via Neural Implicit Representations},
  author    = {Chobola, Tom{\'a}{\v{s}} and Liu, Yu and Zhang, Hanyi and Schnabel, Julia A. and Peng, Tingying},
  booktitle = {European Conference on Computer Vision},
  year      = {2024}
}
```

## Contact

For questions, please contact [feixiaotong@tju.edu.cn](mailto:feixiaotong@tju.edu.cn).
