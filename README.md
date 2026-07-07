# GSA: Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment

[![arXiv](https://img.shields.io/badge/arXiv-2603.21936-b31b1b.svg)](https://arxiv.org/abs/2603.21936)
[![Project Page](https://img.shields.io/badge/Project-Page-1f8ceb.svg)](https://bgu-cs-vil.github.io/GSA-project)

Official code release for our CVPR paper. **Gaussian Splatting Alignment (GSA)**
aligns two independent 3D Gaussian Splatting (3DGS) models with a Sim(3)
similarity transformation (rotation, translation, and scale), even when they
are *different objects of the same category* (e.g. two different cars). GSA is
the first method to solve category-level 3DGS registration, and also sets the
state of the art on same-object registration.


## 1. Installation

Tested on Linux with an NVIDIA RTX 4090 (CUDA driver ≥ 11.8), Python 3.8,
PyTorch 2.4.0+cu118. Newer Python/PyTorch stacks are untested.

```bash
git clone https://github.com/BGU-CS-VIL/GSA.git && cd GSA
conda env create -f environment.yml
conda activate gsa

# Build the two CUDA extensions with the CUDA 11.8 toolchain installed inside
# the env (the system nvcc version does not matter).
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090; set your GPU's compute capability
pip install ./submodules/diff-gaussian-rasterization-feature ./submodules/simple-knn --no-build-isolation
```

## 2. Data & pre-trained models

All evaluation data and the paper's pre-trained 3DGS models ship as one
directory (~13 GB): ShapeNet, Objaverse, CO3D, and the real-car scans, each with
their images, masks, spherical feature maps, camera poses, ground-truth
transforms, and pre-trained feature-3DGS models.

**Download** (Google Drive, ~13 GB):

> [Google Drive](https://drive.google.com/drive/folders/1MgdrzI9fEWaQphQhUk1bbDtfP1zycQpC?usp=sharing)

Then point `GSA_DATA` at the extracted folder (the eval scripts read it, or pass
`--data_root`; nothing is ever written back to it):

```bash
export GSA_DATA=/path/to/GSA_release_data
```

```
GSA_release_data/
├── shapenet/  data/<category>/<object>/     images (train/), masks/, spherical_feature_maps/,
│   │                                        transforms_train.json, points3d.ply
│   │          data/<category>/pair_transforms.json    <- GT perturbations (paper)
│   └── models/<category>/<object>/          pre-trained 3DGS (point_cloud/iteration_7000/)
├── objaverse/ data/<object>/{block0,block1}/  two differnt multiviews frames blocks per object
│   │          data/<object>/world_frame_transforms.json  <- GT block alignment
│   └── models/<object>_block{0,1}/
├── co3d/      data/<category>/<camera_seq>/   images/, sparse/ (COLMAP), spherical_feature_maps/
│   └── models/<sequence_id>/                  30 pre-trained models (10 per category)
└── real_car/  data/2024_04_23_15_41_09/       reference scan (images/, sparse/, spherical_feature_maps/)
    │          data/pair_transforms.json       <- GT perturbations (paper)
    └── models/<scan_id>/                      5 pre-trained models
```

See `README_DATA.md` inside the data directory for details.

> **Important — do not retrain the CO3D and real-car models.** Their ground
> truth comes from a *manual alignment* of every trained model to a reference
> model (reference car scan `2024_04_23_15_41_09`; reference CO3D sequences
> `403_53208_103810` toyplane / `350_36826_69026` chair / `62_4324_10701`
> bicycle). Retraining from scratch produces models in arbitrary frames without
> that alignment, which invalidates the evaluation. Always use the released
> pre-trained models for these two datasets. ShapeNet (canonically aligned) and
> Objaverse (two view-blocks of the same object) can be retrained with
> `scripts/eval_train_shapenet.sh` / `scripts/eval_train_objaverse.sh`.

## 3. Running on your own object pair

You need, for each of the two objects: 360° multi-view images of the *masked*
object plus the binary masks (same filenames), and camera poses (either
Blender-style `transforms_train.json` with a `train/` folder, or a COLMAP
`sparse/` folder with `images/`).

Overall pipeline:
```
images + masks ──▶ sphere_extractor/extract_features.py ──▶ spherical_feature_maps/
                                                                  │
                   train.py (Feature-3DGS)         ◀──────────────┘
                        │ (×2: source + target)
                        ▼
                   register_pair.py  ──▶  aligned models + estimated (R, t, s)
```

### Step 1 — extract spherical-map features (both objects)

```bash
python sphere_extractor/extract_features.py \
    --images_dir data/objA/train \
    --masks_dir  data/objA/masks \
    --output_dir data/objA/spherical_feature_maps
```

This writes `<image>_sphere_fmap_CxHxW.pt` (3×H×W, values in [0,1], background
= 0) plus a `.jpg` visualization per image. The default checkpoint
(`checkpoints/sphere_mapper_gsa.pth`) is the one used for all
paper results.


### Step 2 — train a feature-3DGS model per object

```bash
scripts/train_object.sh data/objA output/objA
scripts/train_object.sh data/objB output/objB
# equivalent to: python train.py -s data/objA -m output/objA --iterations 7000
```

### Step 3 — register the pair

```bash
python register_pair.py \
    --source_model output/objB \
    --target_model output/objA \
    --target_data  data/objA \
    --output_dir   results/objB_to_objA
```

Use `--images images` if your target data is COLMAP-style (`images/` folder)
instead of Blender-style (`train/` folder). The output directory receives
`results.json` (estimated `R, t, s` as a 4×4 `T_est`), `combined_icp.ply` and
`combined_final.ply` (target + transformed source, for inspection), and
`full_align.ply` (the aligned source alone). Add `--random_transform` to first
perturb the source with a random SE(3)+scale transform and report ATE/RRE
against the known ground truth, this is the evaluation setting used for
already-aligned model pairs.

## 4. Reproducing the paper results

All eval scripts read `GSA_DATA` (or `--data_root`) and write everything to a
local `./results/` directory; the data drop is never modified.

The simplest way to reproduce every table is the driver script, which runs all
datasets with the paper settings (and a fixed CO3D seed):

```bash
export GSA_DATA=/path/to/GSA_release_data
scripts/reproduce_all.sh
```

Or run each dataset individually:

```bash
# ShapeNet (cross-instance, 6 categories x 10 pairs)
python eval/eval_register_3dgs_shapenet.py

# Objaverse (same-object, build twice from different multi-views)
python eval/eval_register_3dgs_objaverse.py

# CO3D (cross-instance, 45 pairs per category)
python eval/eval_register_3dgs_co3d.py --category toyplane
python eval/eval_register_3dgs_co3d.py --category chair
python eval/eval_register_3dgs_co3d.py --category bicycle

# 3D real car scans (cross-instance, 10 pairs)
python eval/eval_register_3dgs_real_car.py
```

Reproduction notes:
- **ShapeNet / real car / objaverse**: the ground-truth perturbations ship with
  the data (`pair_transforms.json`, `world_frame_transforms.json`).
- **CO3D**: the random perturbation of each pair is drawn at run time, so individual pairs vary between runs
  while aggregate statistics reproduce. Pass `--seed` for a deterministic run.
- **Reference cameras**: for CO3D and the real cars, the refinement stage
  renders through the cameras of the fixed reference sequence/scan listed
  above, this is possible because we manually align all models of a category to a 
  reference object (manual alignment).
- Quick smoke test: every script accepts `--max_pairs 1` (or `--max_objects 1`)
  to run a single pair.
- **GPU memory**: the ShapeNet evaluation originally used 50000 point
  samples, which requires a ~48 GB GPU. On a 24 GB GPU pass
  `--icp_samples 30000` or `--icp_samples 10000` (results may differ marginally from the paper). The
  other evaluations fit in 24 GB with their paper settings.

## 5. Retraining the spherical mapper (optional)

`sphere_extractor/training/` contains the training code for the spherical
mapper (adapted from [SphericalMaps](https://github.com/VICO-UoE/SphericalMaps),
CVPR 2024). It is only needed if you want to train a new mapper checkpoint;
see `sphere_extractor/README.md`. If you want to use new categories that the Spair71K dataset
does not contain, you need to retrain the SphericalMap on them.

## Acknowledgements

Our feature-3DGS training and rendering code is built directly on the excellent
[Feature-3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs) implementation
(Zhou et al., CVPR 2024).
We are grateful to the authors for releasing their code.

## Citation

```bibtex
@inproceedings{amoyal2026gsa,
  title     = {Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment},
  author    = {Amoyal, Roy and Freifeld, Oren and Baskin, Chaim},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## License

This repository mixes original and third-party code:

- **Our original GSA code** — `register_pair.py`, `eval_register_3dgs_*.py`, and
  `scripts/` — is released under the **MIT license** (see `LICENSE-MIT`).
- The **3DGS training/rendering base** (`train.py`, `scene/`,
  `gaussian_renderer/`, `arguments/`, `utils/`, `submodules/`) inherits the
  Gaussian-Splatting research license of Inria & MPII (see `LICENSE.md`).
- The `sphere_extractor/` subtree derives from SphericalMaps and is licensed
  **CC BY-NC-SA 4.0** (see `sphere_extractor/LICENSE`).
