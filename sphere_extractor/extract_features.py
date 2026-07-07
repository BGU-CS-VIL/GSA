"""Spherical-map feature extraction for masked multi-view object images.

For every image in --images_dir with a matching mask in --masks_dir, extracts
DINOv2 patch tokens, maps them to 3-channel spherical features with the trained
sphere mapper, and writes into --output_dir:
    <image_stem>_sphere_fmap_CxHxW.pt   float tensor (3, H, W) in [0, 1]
    <image_stem>_sphere.jpg             RGB visualization (unless --no_viz)

Background pixels (mask == 0) are set to -1 before the (f + 1) / 2 shift, so
they end up at 0. The .pt files are exactly what scene/dataset_readers.py of
the 3DGS trainer expects inside an object's spherical_feature_maps/ directory.

Example (one object, features written next to its images):
    python sphere_extractor/extract_features.py \
        --images_dir data/my_car/train \
        --masks_dir  data/my_car/masks \
        --output_dir data/my_car/spherical_feature_maps
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from dino_mapper import DINOMapper

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
# The GSA sphere-mapper checkpoint that all released features and pre-trained
# 3DGS models were built with.
N_CATS = 18
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints', 'sphere_mapper_gsa.pth')


class FeatureExtractor:
    def __init__(self, sphere_ckpt_path, n_cats=N_CATS):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Pinned to a python-3.8-compatible commit; newer dinov2 revisions use
        # `X | None` annotations that need python >= 3.10.
        self.dinov2 = torch.hub.load(
            'facebookresearch/dinov2:e1277af2ba9496fbadf7aec6eba56e8d882d1e35',
            'dinov2_vitb14', skip_validation=True)
        self.dinov2.to(self.device)
        self.dinov2.eval()

        self.sphere_mapper = DINOMapper(n_cats=n_cats)
        self.sphere_mapper.load_checkpoint(sphere_ckpt_path, device=self.device)
        self.sphere_mapper.to(self.device)

    @torch.no_grad()
    def extract_features(self, img_path, mask_path):
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        transform = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        mask_transform = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor()
        ])

        img_tensor = transform(img).unsqueeze(0).to(self.device)
        mask_tensor = mask_transform(mask).to(self.device)

        dino_features = self.dinov2.forward_features(img_tensor)
        patch_tokens = dino_features['x_norm_patchtokens']
        if len(patch_tokens.shape) == 2:
            patch_tokens = patch_tokens.unsqueeze(0)

        spherical_features = self.sphere_mapper.sphere_mapper(patch_tokens)
        spherical_features = F.normalize(spherical_features, dim=-1)

        size = int(np.sqrt(spherical_features.size(1)))
        spherical_features = spherical_features.squeeze(0).view(size, size, 3)
        mask_resized = F.interpolate(mask_tensor.unsqueeze(0), size=(size, size), mode='nearest')
        mask_resized = mask_resized.squeeze().bool()

        spherical_features[~mask_resized] = -1

        return spherical_features.cpu(), img.size


def find_mask(masks_dir, img_name):
    """Mask with the same filename, else same stem with any known extension."""
    exact = os.path.join(masks_dir, img_name)
    if os.path.exists(exact):
        return exact
    stem = os.path.splitext(img_name)[0]
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(masks_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def process_directory(images_dir, masks_dir, output_dir, extractor, output_size, save_viz):
    os.makedirs(output_dir, exist_ok=True)
    image_names = sorted(f for f in os.listdir(images_dir) if f.endswith(IMAGE_EXTENSIONS))
    if not image_names:
        raise SystemExit(f"No images found in {images_dir}")

    processed = 0
    for img_name in tqdm(image_names, desc="Extracting features"):
        mask_path = find_mask(masks_dir, img_name)
        if mask_path is None:
            print(f"Warning: no mask for {img_name} in {masks_dir}, skipping")
            continue

        features, orig_size = extractor.extract_features(
            os.path.join(images_dir, img_name), mask_path)

        output_base = os.path.splitext(img_name)[0]
        features_normalized = (features + 1) / 2
        features_tensor = features_normalized.permute(2, 0, 1)
        out_size = output_size if output_size else orig_size[::-1]  # PIL size is (W, H)
        features_upsampled = F.interpolate(features_tensor.unsqueeze(0),
                                           size=out_size,
                                           mode='bilinear',
                                           align_corners=False)

        torch.save(features_upsampled.squeeze(),
                   os.path.join(output_dir, f"{output_base}_sphere_fmap_CxHxW.pt"))

        if save_viz:
            vis = (features_upsampled.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(output_dir, f"{output_base}_sphere.jpg"))

        processed += 1

    print(f"Done: {processed}/{len(image_names)} images -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract spherical-map semantic features for masked object images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images_dir', required=True,
                        help="Directory with the object's RGB images")
    parser.add_argument('--masks_dir', required=True,
                        help="Directory with object masks (same filenames as the images)")
    parser.add_argument('--output_dir', required=True,
                        help="Where to write the *_sphere_fmap_CxHxW.pt files "
                             "(use <object_dir>/spherical_feature_maps for 3DGS training)")
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help="Trained sphere-mapper checkpoint")
    parser.add_argument('--height', type=int, default=None,
                        help="Output feature-map height (default: image height)")
    parser.add_argument('--width', type=int, default=None,
                        help="Output feature-map width (default: image width)")
    parser.add_argument('--no_viz', action='store_true',
                        help="Skip writing the *_sphere.jpg visualizations")
    args = parser.parse_args()

    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be given together")
    output_size = (args.height, args.width) if args.height is not None else None

    extractor = FeatureExtractor(args.checkpoint, n_cats=N_CATS)
    process_directory(args.images_dir, args.masks_dir, args.output_dir,
                      extractor, output_size, save_viz=not args.no_viz)


if __name__ == "__main__":
    main()
