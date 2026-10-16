<h1 align="center">UPTD-Net: Unsupervised Polarization-Guided Two-Stage Decoupling for Underwater Low-Light Image Restoration</h1>

<p align="center">
  <a href="mailto:feixiaotong@tju.edu.cn">Xiaotong Fei</a> ·
  Linghao Shen<sup>*</sup> ·
  Yizhao Huang ·
  Haofeng Hu<sup>*</sup> ·
  Dagong Jia ·
  Binglu Wang
</p>

<p align="center">Tianjin University, Tianjin, 300072, China</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.optlastec.2026.116479"><em>Optics &amp; Laser Technology</em>, 204 (2026) 116479</a>
  ·
  <a href="paper/UPTD-Net.pdf">PDF</a>
</p>

**Abstract:** Underwater images captured under low-light conditions suffer from the coupled effects of scattering and illumination degradation, which reduce contrast, obscure details, and limit visibility. However, existing polarization-based methods often rely on accurate parameter estimation, making restoration sensitive to model assumptions and parameter errors. Supervised methods require paired datasets that are difficult to acquire in real underwater environments. To address these limitations, we formulate an underwater low-light imaging model inspired by Retinex theory and propose UPTD-Net, an unsupervised two-stage decoupling framework that optimizes each input without paired data or pretraining. UPTD-Net decomposes the coupled degradation into scattering removal and illumination recovery. Rather than using physical parameters estimated from polarization to directly determine the restored image, it incorporates polarization information into both stages as spatial weights for descattering refinement and a soft target for degraded illumination estimation. This design incorporates polarization information throughout restoration while reducing reliance on precise physical parameter estimates. Extensive experiments demonstrate that the proposed framework effectively restores underwater images under complex degradation conditions, achieving favorable quantitative performance and visual quality.

<p align="center">
  <img src="https://raw.githubusercontent.com/fxt-kby/UPTD-Net/main/Figs/framework.png" width="70%">
</p>

<p align="center">
  <em>BD-DeNet removes scattering. RI-DeNet recovers illumination. Polarization guides both stages.</em>
</p>

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/fxt-kby/UPTD-Net/main/Figs/result.png" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/fxt-kby/UPTD-Net/main/Figs/pool.png" width="100%">
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
input/illu1/
  s0/s0_1.bmp       S0 intensity, named with _<index>
  dop/1.bmp         DoP map, named <index>.bmp
  s_perp/s0_1.bmp   perpendicular component, same name as the S0 file
```

`S0` and `DoP` are paired by the index. `S⊥` follows the `S0` file name.

`illu1` to `illu3` are three illumination levels of the laboratory data. Each level contains scenes `s0_1` to `s0_15`. The default command uses `illu2`.

## Run

```bash
python run.py
```

```bash
python run.py --datasets input/illu1/s0 --datasets2 input/illu1/dop --datasets3 input/illu1/s_perp --output_folder output/illu1
```

| Output | Content |
| --- | --- |
| `output/` | restored image |
| `output/d/` | direct transmission |
| `output/d_enhanced/` | result after polarization-guided refinement |
| `output/b/` | backscattered light |

## Citation

If you find this project useful, please feel free to cite our paper.

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

## Contact

For questions, please contact [feixiaotong@tju.edu.cn](mailto:feixiaotong@tju.edu.cn).
