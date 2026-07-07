# Spherical-map feature extractor

Extracts 3-channel spherical-map semantic features from masked object images.
The features are DINOv2 ViT-B/14 patch tokens mapped through a small trained
"sphere mapper" head, giving viewpoint-aware semantic coordinates on the
sphere. The 3DGS trainer (`train.py`) distills them into per-Gaussian
features that drive the registration.

This subtree is adapted from
[SphericalMaps](https://github.com/VICO-UoE/SphericalMaps) — *"Improving
Semantic Correspondence with Viewpoint-Guided Spherical Maps"*, Mariotti,
Mac Aodha, Bilen — CVPR 2024. Please cite them if you use this component. The
code in this directory is licensed CC BY-NC-SA 4.0 (see `LICENSE`).

## Usage

```bash
python sphere_extractor/extract_features.py \
    --images_dir <object>/train \
    --masks_dir  <object>/masks \
    --output_dir <object>/spherical_feature_maps
```

Per image `<stem>.png|.jpg` (with a mask of the same filename), this writes:
- `<stem>_sphere_fmap_CxHxW.pt` — float tensor `(3, H, W)` in `[0, 1]`,
  background = 0. This is exactly the format `scene/dataset_readers.py`
  expects inside an object's `spherical_feature_maps/` folder.
- `<stem>_sphere.jpg` — RGB visualization (skip with `--no_viz`).

By default the output resolution equals the input image; the paper data used
`--height 480 --width 640`.

## Checkpoint

`checkpoints/sphere_mapper_gsa.pth` is the GSA sphere-mapper — every released
feature map and pre-trained 3DGS model was built with it. It is the default
`--checkpoint`.

The DINOv2 backbone is fetched via `torch.hub` on first use (pinned to a
Python-3.8-compatible commit of the dinov2 repo; ~330 MB download, cached in
`~/.cache/torch/hub`).

## Retraining the mapper (optional)

`training/` contains the SphericalMaps training code (`train_sph.py`,
`configs/`, `datasets/`, `utils/`) with our local modifications for custom
data. It additionally requires the `timm` package (`pip install timm`) and the
corresponding correspondence datasets (SPair-71k / Freiburg cars — see the
upstream repo). Run from this directory with the parent on the path so
`dino_mapper` resolves:

```bash
cd sphere_extractor/training
PYTHONPATH=.. python train_sph.py --config configs/SPair_custom.yaml
```
