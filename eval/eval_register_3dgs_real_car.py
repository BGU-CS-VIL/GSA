# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roy Amoyal, Oren Freifeld, Chaim Baskin

"""3D real-car cross-instance 3DGS registration evaluation (Table 3).

Registers all 10 pairs among the real-car scans. Every model is manually aligned
to the reference scan (2024_04_23_15_41_09), whose cameras drive the fine stage.

    python eval/eval_register_3dgs_real_car.py
"""

# Allow running this script directly from eval/ (make the repo root importable).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from itertools import combinations
from scipy.spatial.transform import Rotation
from scene import Scene, GaussianModel
from gaussian_renderer import render, render_registration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from utils.math_utils import build_scaling_rotation, quaternion_to_rotation_matrix, rotation_matrix_to_quaternion, quaternion_multiply, quat_multiply
from torch import nn
import math
from utils.loss_utils import l1_loss, ssim, tv_loss 
import torch.optim as optim
import matplotlib.pyplot as plt

# All functions below are from the original code but adapted for 3D Real Car dataset

def transform_shs(shs_feat, rotation_matrix):
    from e3nn import o3
    import einops
    from einops import einsum
    # covert rotation matrix from torch to numpy
    rotation_matrix = rotation_matrix.cpu().numpy().astype(np.float32)
    # shs to cpu
    shs_feat = shs_feat.cpu()
    ## rotate shs
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(np.float32) # switch axes: yzx -> xyz
    # ensure everything is float
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    #rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
            D_3,
            three_degree_shs,
            "... i j, ... j -> ... i",
        )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat

def compute_ate(T_est, T_gt):
    """Compute Absolute Trajectory Error (ATE)"""
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]
    error = np.linalg.norm(t_est - t_gt)
    return error

def compute_rre(T_est, T_gt):
    """
    Compute Relative Rotation Error (RRE) using quaternions
    """
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    
    rot_est = Rotation.from_matrix(R_est)
    rot_gt = Rotation.from_matrix(R_gt)
    rot_diff = rot_gt * rot_est.inv()
    angle_rad = rot_diff.magnitude()
    
    return np.degrees(angle_rad)

def biased_random_angle(min_angle=45, max_angle=180, bias=0.7):
    """Generate a random angle biased towards higher values."""
    base = np.random.random()
    biased_base = base ** bias
    angle = min_angle + (max_angle - min_angle) * biased_base
    return angle

def create_random_transform(min_angle=45, max_angle=180, 
                          translation_range=1.0,
                          min_scale=0.5, max_scale=1.5):
    """Create random transformation with biased angles."""
    # Generate biased random angles
    angles = np.array([biased_random_angle(min_angle, max_angle) for _ in range(3)])
    
    # Create rotation matrix from angles
    R = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    
    # Random translation with controlled range
    t = np.random.uniform(-translation_range, translation_range, 3)
    
    # Random scale with beta distribution centered around 1.0
    scale_range = max_scale - min_scale
    s = min_scale + np.random.beta(2, 2) * scale_range
    
    # Create transformation matrix
    T = np.eye(4)
    T[:3, :3] = R 
    T[:3, 3] = t
    
    return {
        'transform': T.tolist(),
        'rotation': R.tolist(),
        'translation': t.tolist(),
        'scale': float(s),
        'angles': angles.tolist()
    }

def euler_to_rotation_matrix(angles, device='cuda'):
    """Convert Euler angles to rotation matrix."""
    # Ensure angles are on correct device
    angles = angles.to(device)
    
    Rx = torch.tensor([[1, 0, 0],
                      [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                      [0, torch.sin(angles[0]), torch.cos(angles[0])]], device=device)
    
    Ry = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
                      [0, 1, 0],
                      [-torch.sin(angles[1]), 0, torch.cos(angles[1])]], device=device)
    
    Rz = torch.tensor([[torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                      [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                      [0, 0, 1]], device=device)
    
    return Rz @ Ry @ Rx

def apply_transform_to_gaussian_model(gaussian_model, rotation, translation, scale):
    """Apply transformation to a Gaussian model."""
    with torch.no_grad():
        # Transform point cloud
        xyz = gaussian_model.get_xyz
        transformed_xyz = (scale * (rotation @ xyz.T)).T + translation
        
        # Transform spherical harmonics
        features = torch.cat([gaussian_model._features_dc.cpu(), 
                            gaussian_model._features_rest.cpu()], dim=1)
        transformed_features = transform_shs(features[:, 1:, :].clone().cpu(), 
                                          rotation.cpu())
        
        # Transform rotations using quaternions
        rotation_quat = torch.from_numpy(
            rotation_matrix_to_quaternion(rotation.cpu().numpy())
        ).cuda()
        transformed_rotations = torch.nn.functional.normalize(
            quat_multiply(gaussian_model.get_rotation, rotation_quat)
        )
        
        # Scale gaussian sizes
        transformed_scaling = gaussian_model._scaling + torch.log(
            torch.tensor(scale, device='cuda')
        )
        
        # Create transformed model
        transformed_model = GaussianModel(gaussian_model.active_sh_degree)
        transformed_model._xyz = nn.Parameter(transformed_xyz)
        transformed_model._features_dc = gaussian_model._features_dc
        transformed_model._features_rest = nn.Parameter(transformed_features)
        transformed_model._scaling = nn.Parameter(transformed_scaling)
        transformed_model._rotation = nn.Parameter(transformed_rotations)
        transformed_model._opacity = gaussian_model._opacity
        transformed_model._semantic_feature = gaussian_model._semantic_feature
        transformed_model.active_sh_degree = gaussian_model.active_sh_degree
        
        return transformed_model

def find_color_valid_correspondences(src_xyz, src_semantic, tgt_xyz, tgt_semantic, 
                                               semantic_thresh=0.1):
    """Find corresponding points based on semantic features - fully vectorized GPU version"""
    if src_xyz.device != tgt_xyz.device:
        tgt_xyz = tgt_xyz.to(src_xyz.device)
        tgt_semantic = tgt_semantic.to(src_xyz.device)
    
    if len(src_semantic.shape) > 2 or len(tgt_semantic.shape) > 2:
        src_semantic = src_semantic.view(src_semantic.shape[0], -1)
        tgt_semantic = tgt_semantic.view(tgt_semantic.shape[0], -1)
    
    semantic_distances = torch.cdist(src_semantic, tgt_semantic)
    semantic_mask = semantic_distances < semantic_thresh
    has_valid_matches = semantic_mask.any(dim=1)
    
    if not has_valid_matches.any():
        return []
    
    spatial_distances = torch.cdist(src_xyz, tgt_xyz)
    masked_spatial_distances = spatial_distances.clone()
    masked_spatial_distances[~semantic_mask] = float('inf')
    
    min_distances, closest_indices = torch.min(masked_spatial_distances, dim=1)
    valid_sources = min_distances != float('inf')
    
    valid_source_indices = torch.where(valid_sources)[0]
    valid_target_indices = closest_indices[valid_sources]
    
    correspondences = [(src_idx.item(), tgt_idx.item()) 
                      for src_idx, tgt_idx in zip(valid_source_indices, valid_target_indices)]
    
    return correspondences

def coarse_registration(source_xyz, source_semantic, target_xyz, target_semantic, 
                        n_samples=None, max_iter=6, semantic_thresh=0.01):
    """Iterative Closest Point (ICP) with semantic weighting"""
    source_xyz = source_xyz.cuda().float()
    source_semantic = source_semantic.cuda().float()
    target_xyz = target_xyz.cuda().float()
    target_semantic = target_semantic.cuda().float()
    
    n_samples = min(n_samples, source_xyz.shape[0], target_xyz.shape[0]) if n_samples else None
    
    if n_samples:
        src_indices = torch.randperm(source_xyz.shape[0])[:n_samples]
        tgt_indices = torch.randperm(target_xyz.shape[0])[:n_samples]
        source_xyz_sample = source_xyz[src_indices]
        source_semantic_sample = source_semantic[src_indices]
        target_xyz_sample = target_xyz[tgt_indices]
        target_semantic_sample = target_semantic[tgt_indices]
    else:
        source_xyz_sample = source_xyz
        source_semantic_sample = source_semantic
        target_xyz_sample = target_xyz
        target_semantic_sample = target_semantic
    
    R_final = torch.eye(3, device=source_xyz_sample.device)
    t_final = torch.zeros(3, device=source_xyz_sample.device)
    s_final = torch.tensor(1.0, device=source_xyz_sample.device)
    
    source_centroid = source_xyz_sample.mean(dim=0)
    target_centroid = target_xyz_sample.mean(dim=0)
    initial_translation = target_centroid - source_centroid
    
    source_xyz_sample = source_xyz_sample + initial_translation
    t_final = t_final + initial_translation
    
    pbar = tqdm(range(max_iter), desc="ICP Progress", leave=False)
    
    for iteration in pbar:
        correspondences = find_color_valid_correspondences(
            source_xyz_sample, source_semantic_sample,
            target_xyz_sample, target_semantic_sample,
            semantic_thresh=semantic_thresh
        )
        
        if not correspondences:
            pbar.set_description(f"ICP stopped: no correspondences")
            break
            
        src_corr = torch.stack([source_xyz_sample[i] for i, _ in correspondences])
        tgt_corr = torch.stack([target_xyz_sample[j] for _, j in correspondences])
        
        src_centroid = src_corr.mean(dim=0)
        tgt_centroid = tgt_corr.mean(dim=0)
        
        src_centered = src_corr - src_centroid
        tgt_centered = tgt_corr - tgt_centroid
        
        numerator = torch.sum(torch.norm(tgt_centered, dim=1) ** 2)
        denominator = torch.sum(torch.norm(src_centered, dim=1) ** 2)
        scale = torch.sqrt(numerator / denominator) if denominator > 0 else torch.tensor(1.0, device=src_centered.device)
        
        scale = torch.clamp(scale, 0.1, 10.0)
        
        H = src_centered.T @ tgt_centered
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = tgt_centroid - R @ src_centroid
        
        delta_transform = torch.norm(R - torch.eye(3, device=R.device))
        pbar.set_description(f"ICP delta: {delta_transform:.6f}")
        
        if delta_transform < 1e-4 or iteration == max_iter - 1:
            pbar.set_description(f"ICP converged")
            break
            
        R_final = R @ R_final
        t_final = scale * (R @ t_final) + t
        s_final = s_final * scale
        source_xyz_sample = (scale * (R @ source_xyz_sample.T)).T + t
    
    pbar.close()
    print("ICP finished.")
    print(f"Final rotation matrix:\n{R_final}")
    print(f"Final translation: {t_final}")
    print(f"Final scale: {s_final}")
    
    return R_final, t_final, s_final

def load_or_create_transforms(root_dir, object_ids, work_dir="./results/real_car"):
    """Load existing transforms or create new ones for all pairs"""
    transform_file = os.path.join(root_dir, "pair_transforms.json")

    if os.path.exists(transform_file):
        print(f"Loading existing transforms from {transform_file}")
        with open(transform_file, 'r') as f:
            transforms = json.load(f)
    else:
        print("=" * 78)
        print(f"WARNING: {transform_file} not found — generating NEW random pair")
        print("transforms. The paper numbers were computed with the pair_transforms.json")
        print("shipped with the release data; results from freshly generated transforms")
        print("are NOT comparable to the paper tables.")
        print("=" * 78)
        transforms = {}
        for obj1_id, obj2_id in combinations(object_ids, 2):
            pair_key = f"{min(obj1_id, obj2_id)}_{max(obj1_id, obj2_id)}"
            transforms[pair_key] = create_random_transform(
                min_angle=45,  # Minimum rotation angle
                max_angle=180,  # Maximum rotation angle
                translation_range=1.0,  # Translation range
                min_scale=0.5,  # Minimum scale
                max_scale=1.5   # Maximum scale
            )

        # The data drop is treated as read-only: keep generated transforms local.
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, "pair_transforms.json"), 'w') as f:
            json.dump(transforms, f, indent=4)

    return transforms

def save_combined_model(gs1, gs2, output_path, max_sh_degree=3):
    """Helper function to save combined models"""
    combined_model = GaussianModel(max_sh_degree)
    
    # Ensure all tensors are on CUDA before concatenation
    def ensure_cuda(tensor):
        return tensor.cuda() if tensor.device.type != 'cuda' else tensor
    
    # Get all parameters and ensure they're on CUDA
    xyz1 = ensure_cuda(gs1.get_xyz)
    xyz2 = ensure_cuda(gs2.get_xyz)
    features_dc1 = ensure_cuda(gs1._features_dc)
    features_dc2 = ensure_cuda(gs2._features_dc)
    features_rest1 = ensure_cuda(gs1._features_rest)
    features_rest2 = ensure_cuda(gs2._features_rest)
    scaling1 = ensure_cuda(gs1._scaling)
    scaling2 = ensure_cuda(gs2._scaling)
    rotation1 = ensure_cuda(gs1._rotation)
    rotation2 = ensure_cuda(gs2._rotation)
    opacity1 = ensure_cuda(gs1._opacity)
    opacity2 = ensure_cuda(gs2._opacity)
    semantic1 = ensure_cuda(gs1._semantic_feature)
    semantic2 = ensure_cuda(gs2._semantic_feature)
    
    # Concatenate all parameters
    combined_xyz = torch.cat([xyz1, xyz2])
    combined_features_dc = torch.cat([features_dc1, features_dc2])
    combined_features_rest = torch.cat([features_rest1, features_rest2])
    combined_scaling = torch.cat([scaling1, scaling2])
    combined_rotation = torch.cat([rotation1, rotation2])
    combined_opacity = torch.cat([opacity1, opacity2])
    combined_semantic = torch.cat([semantic1, semantic2])
    
    # Convert to numpy for saving
    combined_xyz_np = combined_xyz.detach().cpu().numpy()
    combined_features_dc_np = combined_features_dc.detach().cpu().numpy()
    combined_features_rest_np = combined_features_rest.detach().cpu().numpy()
    combined_opacity_np = combined_opacity.detach().cpu().numpy()
    combined_scaling_np = combined_scaling.detach().cpu().numpy()
    combined_rotation_np = combined_rotation.detach().cpu().numpy()
    combined_semantic_np = combined_semantic.detach().cpu().numpy()
    
    # Set combined model parameters
    combined_model._xyz = nn.Parameter(torch.from_numpy(combined_xyz_np).requires_grad_(True))
    combined_model._features_dc = nn.Parameter(torch.from_numpy(combined_features_dc_np).requires_grad_(True))
    combined_model._features_rest = nn.Parameter(torch.from_numpy(combined_features_rest_np).requires_grad_(True))
    combined_model._scaling = nn.Parameter(torch.from_numpy(combined_scaling_np).requires_grad_(True))
    combined_model._rotation = nn.Parameter(torch.from_numpy(combined_rotation_np).requires_grad_(True))
    combined_model._opacity = nn.Parameter(torch.from_numpy(combined_opacity_np).requires_grad_(True))
    combined_model._semantic_feature = nn.Parameter(torch.from_numpy(combined_semantic_np).requires_grad_(True))
    combined_model.active_sh_degree = max_sh_degree
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the combined model
    print(f"Saving combined model to: {output_path}")
    combined_model.save_ply(output_path)

def rotation_matrix_to_euler_angles(R):
    """Convert 3x3 rotation matrix to euler angles (x, y, z)"""
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.zeros_like(y)

    return torch.stack([x, y, z])

def fine_registration(target_model_path, source_model, target_data_path, initial_transform=None):
    """Training function for registration with geometric loss
    
    Args:
        target_model_path: Path to the target trained GS model (for loading PLY)
        source_model: Source GaussianModel object (already loaded and transformed)
        target_data_path: Path to dataset with images for training (cameras_model_name)
        initial_transform: Optional initial transformation from ICP (R, t, s)
    """
    print("\nStarting training with paths:")
    print(f"Target model path (PLY): {target_model_path}")
    print(f"Target data path (images): {target_data_path}")
    
    class DummyArgs:
        def __init__(self, source_path, model_path):
            self.source_path = source_path
            self.model_path = model_path
            self.images = "images"
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = False
            self.render_items = ['RGB', 'Depth', 'Edge', 'Normal', 'Curvature', 'Feature Map']
            self.sh_degree = 3
    
    # Note: source_path uses target_data_path (images), model_path uses target_model_path (PLY)
    dummy_args = DummyArgs(source_path=target_data_path, model_path=target_model_path)
    parser = ArgumentParser()
    dataset_model = ModelParams(parser).extract(dummy_args)
    pipe = PipelineParams(parser)
    
    gaussians_target = GaussianModel(3)
    gaussians_source = source_model
    gaussians_output = GaussianModel(3)
    
    target_ply = os.path.join(target_model_path, "point_cloud/iteration_7000/point_cloud.ply")
    gaussians_target.load_ply(target_ply)
    
    # Ensure source model parameters are float
    gaussians_source._xyz = nn.Parameter(gaussians_source._xyz.float())
    gaussians_source._features_dc = nn.Parameter(gaussians_source._features_dc.float())
    gaussians_source._features_rest = nn.Parameter(gaussians_source._features_rest.float())
    gaussians_source._scaling = nn.Parameter(gaussians_source._scaling.float())
    gaussians_source._rotation = nn.Parameter(gaussians_source._rotation.float())
    gaussians_source._opacity = nn.Parameter(gaussians_source._opacity.float())
    gaussians_source._semantic_feature = nn.Parameter(gaussians_source._semantic_feature.float())
    
    # Clone source to output
    gaussians_output._xyz = gaussians_source._xyz.clone()
    gaussians_output._features_dc = gaussians_source._features_dc.clone()
    gaussians_output._features_rest = gaussians_source._features_rest.clone()
    gaussians_output._scaling = gaussians_source._scaling.clone()
    gaussians_output._rotation = gaussians_source._rotation.clone()
    gaussians_output._opacity = gaussians_source._opacity.clone()
    gaussians_output._semantic_feature = gaussians_source._semantic_feature.clone()
    gaussians_output.active_sh_degree = gaussians_source.active_sh_degree
    
    scene = Scene(dataset_model, gaussians_target, load_iteration=-1, shuffle=False)
    # 5 diverse (evenly spread) views for the fine stage, matching the other datasets.
    all_cams = scene.getTrainCameras().copy()
    viewpoint_stack = all_cams[::max(1, len(all_cams) // 5)]

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    if initial_transform is not None:
        R_init, t_init, s_init = initial_transform
        
        R_init = torch.tensor(R_init, device="cuda", dtype=torch.float32)
        t_init = torch.tensor(t_init, device="cuda", dtype=torch.float32)
        s_init = torch.tensor([float(s_init)], device="cuda", dtype=torch.float32)
        
        euler_angles = rotation_matrix_to_euler_angles(R_init)
        print("Initial Euler angles:", euler_angles * (180/math.pi))
        
        gaussians_source.global_rotation.data = euler_angles.float()
        gaussians_source.global_translation.data = t_init.float()
        gaussians_source.global_scale.data = s_init.float()

    # Render feature maps for target model
    print("Rendering target feature maps...")
    target_feature_maps = []
    for viewpoint_cam in viewpoint_stack:
        viewpoint_cam = viewpoint_cam.cuda()
        render_pkg = render(viewpoint_cam, gaussians_target, pipe, background)
        target_feature_maps.append(render_pkg["feature_map"].float())
    
    optimizer = optim.AdamW([
        {'params': gaussians_source.global_translation, 'lr': 0.01},
        {'params': gaussians_source.global_rotation, 'lr': 0.01},
        {'params': gaussians_source.global_scale, 'lr': 0.01},        
    ])
    
    best_loss = float('inf')
    best_transform = {
        'rotation': None,
        'translation': None,
        'scale': None
    }
    os.makedirs("plot_3d_real_car", exist_ok=True)

    print("Starting optimization...")
    for iteration in range(60):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        render_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        
        for i, viewpoint_cam in enumerate(viewpoint_stack):
            render_pkg = render_registration(
                viewpoint_cam, 
                gaussians_source, 
                pipe, 
                background,
                override_color=gaussians_source.get_semantic_feature.squeeze(1).float()
            )
            source_feature_map = render_pkg["render"].float()
            
            loss = torch.nn.MSELoss()(source_feature_map, target_feature_maps[i])
            render_loss = render_loss + loss

        total_loss = render_loss

        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_transform['rotation'] = gaussians_source.global_rotation.data.clone()
            best_transform['translation'] = gaussians_source.global_translation.data.clone()
            best_transform['scale'] = gaussians_source.global_scale.data.clone()
            print(f"New best loss at iteration {iteration}: {best_loss}")
            print(f"Best rotation: {best_transform['rotation'].detach().cpu().numpy() * (180/math.pi)}°")
            print(f"Best translation: {best_transform['translation'].detach().cpu().numpy()}")
            print(f"Best scale: {best_transform['scale'].detach().cpu().numpy()}")

        print(f"Iteration {iteration}, Loss: {total_loss.item()}")
        print(f"Current rotation: {gaussians_source.global_rotation.detach().cpu().numpy() * (180/math.pi)}°")
        print(f"Current translation: {gaussians_source.global_translation.detach().cpu().numpy()}")
        print(f"Current scale: {gaussians_source.global_scale.detach().cpu().numpy()}")
        
        total_loss.backward(retain_graph=True)
        optimizer.step()

    # Apply best transformation for final output
    print("\nApplying best transformation found...")
    print(f"Best loss achieved: {best_loss}")
    
    final_angles = best_transform['rotation'].detach().cpu().numpy()
    final_rotation = Rotation.from_euler('xyz', final_angles).as_matrix()
    final_rotation_torch = torch.from_numpy(final_rotation).float().cuda()
    final_translation = best_transform['translation'].detach()
    final_scale = best_transform['scale'].detach().item()

    T_final = np.eye(4)
    T_final[:3, :3] = final_rotation_torch.detach().cpu().numpy()
    T_final[:3, 3] = final_translation.detach().cpu().numpy()

    gaussians_output = apply_transform_to_gaussian_model(
        gaussians_source,
        final_rotation_torch,
        final_translation,
        final_scale
    )

    return {
        'T_est': T_final,
        'scale': final_scale,
        'gaussians_output': gaussians_output,
        'best_loss': best_loss
    }

def save_ply_pair(obj1_id, obj2_id, base_dir, model_dir, work_dir, transforms):
    """Save PLY pair and transformation data for a given object pair"""
    pair_key = f"{min(obj1_id, obj2_id)}_{max(obj1_id, obj2_id)}"
    transform_data = transforms[pair_key]

    # Create ply_pairs directory (in the local work dir — model dirs stay read-only)
    ply_pairs_dir = os.path.join(work_dir, "ply_pairs")
    os.makedirs(ply_pairs_dir, exist_ok=True)

    # Create directory for this specific pair
    pair_dir = os.path.join(ply_pairs_dir, pair_key)
    os.makedirs(pair_dir, exist_ok=True)

    # Paths for source files
    target_ply_path = os.path.join(model_dir, obj1_id, "point_cloud/iteration_7000/point_cloud.ply")
    source_ply_path = os.path.join(model_dir, obj2_id, "point_cloud/iteration_7000/point_cloud.ply")
    
    if not os.path.exists(target_ply_path) or not os.path.exists(source_ply_path):
        print(f"Missing PLY files for pair {pair_key}")
        return False
    
    # Load models
    gs1 = GaussianModel(3)  # target
    gs2 = GaussianModel(3)  # source
    gs1.load_ply(target_ply_path)
    gs2.load_ply(source_ply_path)
    
    # Get transform data
    T_gt = np.array(transform_data['transform'])
    R_gt = torch.from_numpy(np.array(transform_data['rotation'])).float().cuda()
    t_gt = torch.from_numpy(np.array(transform_data['translation'])).float().cuda()
    s_gt = transform_data['scale']
    
    # Apply random transform to gs2
    gs2_random = apply_transform_to_gaussian_model(gs2, R_gt, t_gt, s_gt)
    print("Saving PLY pair in directory:", pair_dir)
    
    # Save files in pair directory
    target_save_path = os.path.join(pair_dir, f"{obj1_id}_target.ply")
    source_save_path = os.path.join(pair_dir, f"{obj2_id}_source_random.ply")
    transform_save_path = os.path.join(pair_dir, "transform.json")
    
    # Save PLY files
    gs1.save_ply(target_save_path)
    gs2_random.save_ply(source_save_path)
    
    # Save transformation data
    transform_info = {
        'target_id': obj1_id,
        'source_id': obj2_id,
        'transform': T_gt.tolist(),
        'rotation': R_gt.cpu().numpy().tolist(),
        'translation': t_gt.cpu().numpy().tolist(),
        'scale': float(s_gt)
    }
    
    with open(transform_save_path, 'w') as f:
        json.dump(transform_info, f, indent=4)
    
    return True

def process_object_pair(obj1_id, obj2_id, base_dir, model_dir, work_dir, transforms, cameras_model_name):
    """Process a single pair of objects with improved transform handling

    Args:
        obj1_id: Target object ID (for PLY model)
        obj2_id: Source object ID (for PLY model)
        base_dir: Base directory for dataset (camera images)
        model_dir: Directory with the pre-trained 3DGS models (read-only)
        work_dir: Local directory for all outputs
        transforms: Dictionary of transforms
        cameras_model_name: Model name to use for camera/training data (images)
    """
    pair_key = f"{min(obj1_id, obj2_id)}_{max(obj1_id, obj2_id)}"
    transform_data = transforms[pair_key]

    # Create combined models directory (all writes go to the local work dir)
    combined_models_dir = os.path.join(work_dir, "combined_models", pair_key)
    os.makedirs(combined_models_dir, exist_ok=True)

    # Construct paths - PLY models come from object IDs
    target_model_path = os.path.join(model_dir, obj1_id)
    source_model_path = os.path.join(model_dir, obj2_id)
    
    # Training images come from cameras_model_name (separate from object IDs)
    target_data_path = os.path.join(base_dir, cameras_model_name)
    
    print(f"\n{'='*60}")
    print(f"Processing pair: {obj2_id} -> {obj1_id}")
    print(f"{'='*60}")
    print(f"Target model path (PLY): {target_model_path}")
    print(f"Source model path (PLY): {source_model_path}")
    print(f"Target data path (images): {target_data_path}")
    
    target_ply_path = os.path.join(target_model_path, "point_cloud/iteration_7000/point_cloud.ply")
    source_ply_path = os.path.join(source_model_path, "point_cloud/iteration_7000/point_cloud.ply")
    
    if not os.path.exists(target_ply_path) or not os.path.exists(source_ply_path):
        print(f"Missing PLY files for pair {pair_key}")
        return None
    
    # Load models - initially aligned
    gs1 = GaussianModel(3)  # target
    gs2 = GaussianModel(3)  # source
    gs1.load_ply(target_ply_path)
    gs2.load_ply(source_ply_path)
    
    # Save initial combined model (models are aligned)
    gt_combined_path = os.path.join(combined_models_dir, "combined_gt.ply")
    save_combined_model(gs1, gs2, gt_combined_path)
    
    # Get random transform to apply
    T_gt = np.array(transform_data['transform'])
    R_gt = torch.from_numpy(np.array(transform_data['rotation'])).float().cuda()
    t_gt = torch.from_numpy(np.array(transform_data['translation'])).float().cuda()
    s_gt = transform_data['scale']
    
    # Apply random transform to gs2 to misalign it
    gs2_random = apply_transform_to_gaussian_model(gs2, R_gt, t_gt, s_gt)
    
    # Save combined model after random transformation
    random_combined_path = os.path.join(combined_models_dir, "combined_random.ply")
    save_combined_model(gs1, gs2_random, random_combined_path)
    
    # Run ICP between randomly transformed gs2 and target gs1
    R_init, t_init, s_init = coarse_registration(
        gs2_random.get_xyz, gs2_random._semantic_feature,  # Use randomly transformed source
        gs1.get_xyz, gs1._semantic_feature,  # Target
        n_samples=15000
    )
    
    # Create ICP transformation matrix
    T_icp = np.eye(4)
    T_icp[:3, :3] = R_init.detach().cpu().numpy()
    T_icp[:3, 3] = t_init.detach().cpu().numpy()
    
    # Compute inverse of ground truth transform
    T_gt_inv = np.linalg.inv(T_gt)
    
    # Compute ICP errors against inverse ground truth
    ate_icp = compute_ate(T_icp, T_gt_inv)
    rre_icp = compute_rre(T_icp, T_gt_inv)
    
    print(f"ICP Results - ATE: {ate_icp:.4f}, RRE: {rre_icp:.4f}°")
    
    # Apply ICP transform to the randomly transformed model
    gs2_icp = apply_transform_to_gaussian_model(
        gs2_random,  # Apply to randomly transformed model
        R_init.detach(), 
        t_init.detach(), 
        s_init.detach().item()
    )
    
    # Save combined model after ICP
    icp_combined_path = os.path.join(combined_models_dir, "combined_icp.ply")
    save_combined_model(gs1, gs2_icp, icp_combined_path)
    
    # Run full registration starting from ICP result
    final_results = fine_registration(
        target_model_path=target_model_path,
        source_model=gs2_random,  # Use the randomly transformed model
        target_data_path=target_data_path,  # Uses cameras_model_name for images
        initial_transform=(R_init.detach().cpu().numpy(),
                        t_init.detach().cpu().numpy(),
                        s_init.detach().cpu().numpy())
    )
    
    # Save combined model after final registration
    final_combined_path = os.path.join(combined_models_dir, "combined_final.ply")
    save_combined_model(gs1, final_results['gaussians_output'], final_combined_path)
    
    # Compute final errors against inverse ground truth
    ate_final = compute_ate(final_results['T_est'], T_gt_inv)
    rre_final = compute_rre(final_results['T_est'], T_gt_inv)
    
    print(f"Final Results - ATE: {ate_final:.4f}, RRE: {rre_final:.4f}°")
    
    # Detailed rotation analysis
    R_est = final_results['T_est'][:3, :3]
    R_gt_np = T_gt_inv[:3, :3]
    rot_est = Rotation.from_matrix(R_est)
    rot_gt = Rotation.from_matrix(R_gt_np)
    
    euler_est = rot_est.as_euler('xyz', degrees=True)
    euler_gt = rot_gt.as_euler('xyz', degrees=True)
    euler_diff = np.abs(euler_est - euler_gt)
    
    t_est = final_results['T_est'][:3, 3]
    t_gt = T_gt_inv[:3, 3]
    
    result = {
        'pair': pair_key,
        'source': obj2_id,
        'target': obj1_id,
        'cameras_model': cameras_model_name,
        'icp_ate': float(ate_icp),
        'icp_rre': float(rre_icp),
        'final_ate': float(ate_final),
        'final_rre': float(rre_final),
        'scale': final_results['scale'],
        'best_loss': final_results['best_loss'],
        'rotation_analysis': {
            'euler_angles': {
                'estimated': euler_est.tolist(),
                'ground_truth': euler_gt.tolist(),
                'differences': euler_diff.tolist()
            }
        },
        'translation_vectors': {
            'estimated': t_est.tolist(),
            'ground_truth': t_gt.tolist()
        },
        'T_gt': T_gt.tolist(),
        'T_est_icp': T_icp.tolist(),
        'T_est_final': final_results['T_est'].tolist()
    }
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return result

def main():
    """Main function to process 3D Real Car dataset"""
    cli = ArgumentParser(description="3D Real Car 3DGS registration evaluation")
    cli.add_argument("--data_root", default=os.environ.get("GSA_DATA", "./GSA_release_data"),
                     help="Release data root (contains real_car/data and real_car/models)")
    cli.add_argument("--results_dir", default="./results/real_car",
                     help="Where to write all outputs")
    cli.add_argument("--max_pairs", type=int, default=None,
                     help="Limit number of pairs (for smoke tests)")
    cli.add_argument("--skip_ply_pairs", action="store_true",
                     help="Skip exporting the per-pair target/source ply visualization files")
    cli_args = cli.parse_args()

    # Data and models come from the (read-only) release drop; all outputs are local.
    base_dir = os.path.join(cli_args.data_root, "real_car", "data")
    model_dir = os.path.join(cli_args.data_root, "real_car", "models")
    work_dir = cli_args.results_dir
    results_dir = os.path.join(work_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    # Fixed camera model name for training images (separate from object IDs).
    # All real-car models were manually aligned to this reference scan; its
    # cameras drive the differentiable refinement stage.
    cameras_model_name = "2024_04_23_15_41_09"

    print("Processing 3D Real Car dataset...")
    print(f"Base directory: {base_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Camera model for training images: {cameras_model_name}")

    # Get all object IDs (car directories)
    object_ids = [d for d in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, d))]

    print(f"Found {len(object_ids)} objects: {object_ids}")

    # Load or create transform pairs
    transforms = load_or_create_transforms(base_dir, object_ids, work_dir)

    pair_list = list(combinations(object_ids, 2))
    if cli_args.max_pairs is not None:
        pair_list = pair_list[:cli_args.max_pairs]

    # First, save all PLY pairs
    if not cli_args.skip_ply_pairs:
        print("\nSaving PLY pairs for all object combinations...")
        for obj1_id, obj2_id in tqdm(pair_list,
                                    desc=f"Saving PLY pairs"):
            save_ply_pair(obj1_id, obj2_id, base_dir, model_dir, work_dir, transforms)

    # Process all pairs for registration
    all_results = []
    all_ate_icp = []
    all_rre_icp = []
    all_ate_final = []
    all_rre_final = []

    print("\nProcessing all object pairs for registration...")
    for obj1_id, obj2_id in tqdm(pair_list,
                                desc=f"Processing car pairs"):
        result = process_object_pair(
            obj1_id, obj2_id, base_dir, model_dir, work_dir, transforms,
            cameras_model_name=cameras_model_name
        )
        
        if result is not None:
            all_results.append(result)
            all_ate_icp.append(result['icp_ate'])
            all_rre_icp.append(result['icp_rre'])
            all_ate_final.append(result['final_ate'])
            all_rre_final.append(result['final_rre'])
            
            tqdm.write(f"\nResults for {obj1_id}-{obj2_id}:")
            tqdm.write(f"ICP  - ATE: {result['icp_ate']:.4f} meters, RRE: {result['icp_rre']:.4f} degrees")
            tqdm.write(f"Final- ATE: {result['final_ate']:.4f} meters, RRE: {result['final_rre']:.4f} degrees")
    
    # Compute overall statistics
    if all_results:
        mean_ate_icp = np.mean(all_ate_icp)
        mean_rre_icp = np.mean(all_rre_icp)
        std_ate_icp = np.std(all_ate_icp)
        std_rre_icp = np.std(all_rre_icp)
        median_ate_icp = np.median(all_ate_icp)
        median_rre_icp = np.median(all_rre_icp)
        
        mean_ate_final = np.mean(all_ate_final)
        mean_rre_final = np.mean(all_rre_final)
        std_ate_final = np.std(all_ate_final)
        std_rre_final = np.std(all_rre_final)
        median_ate_final = np.median(all_ate_final)
        median_rre_final = np.median(all_rre_final)
        
        print("\n" + "="*60)
        print("BATCH REGISTRATION SUMMARY - 3D Real Car Dataset")
        print("="*60)
        print(f"Camera model for images: {cameras_model_name}")
        print(f"Total pairs: {len(list(combinations(object_ids, 2)))}")
        print(f"Successful: {len(all_results)}")
        
        print("\nICP Registration:")
        print(f"  Mean ATE: {mean_ate_icp:.4f} ± {std_ate_icp:.4f} meters")
        print(f"  Median ATE: {median_ate_icp:.4f} meters")
        print(f"  Mean RRE: {mean_rre_icp:.4f} ± {std_rre_icp:.4f} degrees")
        print(f"  Median RRE: {median_rre_icp:.4f} degrees")
        
        print("\nFinal Registration:")
        print(f"  Mean ATE: {mean_ate_final:.4f} ± {std_ate_final:.4f} meters")
        print(f"  Median ATE: {median_ate_final:.4f} meters")
        print(f"  Mean RRE: {mean_rre_final:.4f} ± {std_rre_final:.4f} degrees")
        print(f"  Median RRE: {median_rre_final:.4f} degrees")
        
        print("\nPer-pair Results:")
        print("-"*60)
        for r in all_results:
            print(f"  {r['source']} -> {r['target']}: RRE={r['final_rre']:.2f}°, ATE={r['final_ate']:.4f}")
        
        # Save results
        results_path = os.path.join(results_dir, "3d_real_car_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'cameras_model': cameras_model_name,
                'individual_results': all_results,
                'summary': {
                    'icp': {
                        'mean_ate': float(mean_ate_icp),
                        'std_ate': float(std_ate_icp),
                        'median_ate': float(median_ate_icp),
                        'mean_rre': float(mean_rre_icp),
                        'std_rre': float(std_rre_icp),
                        'median_rre': float(median_rre_icp)
                    },
                    'final': {
                        'mean_ate': float(mean_ate_final),
                        'std_ate': float(std_ate_final),
                        'median_ate': float(median_ate_final),
                        'mean_rre': float(mean_rre_final),
                        'std_rre': float(std_rre_final),
                        'median_rre': float(median_rre_final)
                    }
                }
            }, f, indent=4)
        
        print(f"\nResults saved to {results_path}")
    else:
        print("No valid results were obtained. Check that the dataset paths are correct.")

if __name__ == "__main__":
    main()