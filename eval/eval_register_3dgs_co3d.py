# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roy Amoyal, Oren Freifeld, Chaim Baskin

"""CO3D cross-instance 3DGS registration evaluation (Table 3).

Registers all 45 pairs within a CO3D category (toyplane, chair, or bicycle),
selected via --category. Each category has a fixed reference-camera sequence
whose views drive the fine stage, and 10 object sequences registered pairwise.
See CO3D_CONFIG below.

    python eval/eval_register_3dgs_co3d.py --category toyplane
"""

# Allow running this script directly from eval/ (make the repo root importable).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import numpy as np
import torch
from tqdm import tqdm
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
from itertools import combinations

def transform_shs(shs_feat, rotation_matrix):
    from e3nn import o3
    import einops
    from einops import einsum
    rotation_matrix = rotation_matrix.cpu().numpy().astype(np.float32)
    shs_feat = shs_feat.cpu()
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(np.float32)
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))

    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(D_1, one_degree_shs, "... i j, ... j -> ... i")
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(D_2, two_degree_shs, "... i j, ... j -> ... i")
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(D_3, three_degree_shs, "... i j, ... j -> ... i")
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
    """Compute Relative Rotation Error (RRE) using quaternions"""
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
    angles = np.array([biased_random_angle(min_angle, max_angle) for _ in range(3)])
    R = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    t = np.random.uniform(-translation_range, translation_range, 3)
    scale_range = max_scale - min_scale
    s = min_scale + np.random.beta(2, 2) * scale_range
    
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

def apply_transform_to_gaussian_model(gaussian_model, rotation, translation, scale):
    """Apply transformation to a Gaussian model."""
    with torch.no_grad():
        rotation = rotation.float()
        translation = translation.float()
        if isinstance(scale, torch.Tensor):
            scale = scale.float().item()
        
        xyz = gaussian_model.get_xyz.float()
        transformed_xyz = (scale * (rotation @ xyz.T)).T + translation
        
        features = torch.cat([gaussian_model._features_dc.cpu(), 
                            gaussian_model._features_rest.cpu()], dim=1)
        transformed_features = transform_shs(features[:, 1:, :].clone().cpu(), 
                                          rotation.cpu())
        
        rotation_quat = torch.from_numpy(
            rotation_matrix_to_quaternion(rotation.cpu().numpy())
        ).cuda().float()
        
        transformed_rotations = torch.nn.functional.normalize(
            quat_multiply(gaussian_model.get_rotation.float(), rotation_quat)
        )
        
        transformed_scaling = gaussian_model._scaling.float() + torch.log(
            torch.tensor(scale, device='cuda', dtype=torch.float32)
        )
        
        transformed_model = GaussianModel(gaussian_model.active_sh_degree)
        transformed_model._xyz = nn.Parameter(transformed_xyz.float())
        transformed_model._features_dc = nn.Parameter(gaussian_model._features_dc.float())
        transformed_model._features_rest = nn.Parameter(transformed_features.float())
        transformed_model._scaling = nn.Parameter(transformed_scaling.float())
        transformed_model._rotation = nn.Parameter(transformed_rotations.float())
        transformed_model._opacity = nn.Parameter(gaussian_model._opacity.float())
        transformed_model._semantic_feature = nn.Parameter(gaussian_model._semantic_feature.float())
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
    
    return R_final, t_final, s_final

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
    """Training function for registration with geometric loss"""
    
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
    
    dummy_args = DummyArgs(source_path=target_data_path, model_path=target_model_path)
    parser = ArgumentParser()
    dataset_model = ModelParams(parser).extract(dummy_args)
    pipe = PipelineParams(parser)
    
    gaussians_target = GaussianModel(3)
    gaussians_source = GaussianModel(3)
    gaussians_output = GaussianModel(3)
    
    target_ply = os.path.join(target_model_path, "point_cloud/iteration_7000/point_cloud.ply")
    
    gaussians_target.load_ply(target_ply)
    gaussians_source = source_model
    
    gaussians_source._xyz = nn.Parameter(gaussians_source._xyz.float())
    gaussians_source._features_dc = nn.Parameter(gaussians_source._features_dc.float())
    gaussians_source._features_rest = nn.Parameter(gaussians_source._features_rest.float())
    gaussians_source._scaling = nn.Parameter(gaussians_source._scaling.float())
    gaussians_source._rotation = nn.Parameter(gaussians_source._rotation.float())
    gaussians_source._opacity = nn.Parameter(gaussians_source._opacity.float())
    gaussians_source._semantic_feature = nn.Parameter(gaussians_source._semantic_feature.float())
    
    gaussians_output._xyz = gaussians_source._xyz.clone()
    gaussians_output._features_dc = gaussians_source._features_dc.clone()
    gaussians_output._features_rest = gaussians_source._features_rest.clone()
    gaussians_output._scaling = gaussians_source._scaling.clone()
    gaussians_output._rotation = gaussians_source._rotation.clone()
    gaussians_output._opacity = gaussians_source._opacity.clone()
    gaussians_output._semantic_feature = gaussians_source._semantic_feature.clone()
    gaussians_output.active_sh_degree = gaussians_source.active_sh_degree
    
    scene = Scene(dataset_model, gaussians_target, load_iteration=-1, shuffle=False)
    viewpoint_stack = scene.getTrainCameras().copy()
    # split equally get 5 viewpoints
    viewpoint_stack = viewpoint_stack[::len(viewpoint_stack)//5]
    print(f"Viewpoint stack length: {len(viewpoint_stack)}")
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    if initial_transform is not None:
        R_init, t_init, s_init = initial_transform
        
        R_init = torch.tensor(R_init, device="cuda", dtype=torch.float32)
        t_init = torch.tensor(t_init, device="cuda", dtype=torch.float32)
        s_init = torch.tensor([float(s_init)], device="cuda", dtype=torch.float32)
        
        euler_angles = rotation_matrix_to_euler_angles(R_init)
        
        gaussians_source.global_rotation.data = euler_angles.float()
        gaussians_source.global_translation.data = t_init.float()
        gaussians_source.global_scale.data = s_init.float()

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
        
        total_loss.backward(retain_graph=True)
        optimizer.step()

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

def save_combined_model(gs1, gs2, output_path, max_sh_degree=3):
    """Helper function to save combined models"""
    combined_model = GaussianModel(max_sh_degree)
    
    def ensure_cuda(tensor):
        return tensor.cuda() if tensor.device.type != 'cuda' else tensor
    
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
    
    combined_xyz = torch.cat([xyz1, xyz2])
    combined_features_dc = torch.cat([features_dc1, features_dc2])
    combined_features_rest = torch.cat([features_rest1, features_rest2])
    combined_scaling = torch.cat([scaling1, scaling2])
    combined_rotation = torch.cat([rotation1, rotation2])
    combined_opacity = torch.cat([opacity1, opacity2])
    combined_semantic = torch.cat([semantic1, semantic2])
    
    combined_xyz_np = combined_xyz.detach().cpu().numpy()
    combined_features_dc_np = combined_features_dc.detach().cpu().numpy()
    combined_features_rest_np = combined_features_rest.detach().cpu().numpy()
    combined_opacity_np = combined_opacity.detach().cpu().numpy()
    combined_scaling_np = combined_scaling.detach().cpu().numpy()
    combined_rotation_np = combined_rotation.detach().cpu().numpy()
    combined_semantic_np = combined_semantic.detach().cpu().numpy()
    
    combined_model._xyz = nn.Parameter(torch.from_numpy(combined_xyz_np).requires_grad_(True))
    combined_model._features_dc = nn.Parameter(torch.from_numpy(combined_features_dc_np).requires_grad_(True))
    combined_model._features_rest = nn.Parameter(torch.from_numpy(combined_features_rest_np).requires_grad_(True))
    combined_model._scaling = nn.Parameter(torch.from_numpy(combined_scaling_np).requires_grad_(True))
    combined_model._rotation = nn.Parameter(torch.from_numpy(combined_rotation_np).requires_grad_(True))
    combined_model._opacity = nn.Parameter(torch.from_numpy(combined_opacity_np).requires_grad_(True))
    combined_model._semantic_feature = nn.Parameter(torch.from_numpy(combined_semantic_np).requires_grad_(True))
    combined_model.active_sh_degree = max_sh_degree
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_model.save_ply(output_path)


def process_pair(target_trained_name, source_trained_name, cameras_model_name, 
                 base_output_dir, category="chair"):
    """Process a single pair of models and return RRE results"""
    
    print(f"\n{'='*60}")
    print(f"Processing pair: {source_trained_name} -> {target_trained_name}")
    print(f"{'='*60}")
    
    data_root = os.environ.get("GSA_DATA", "./GSA_release_data")
    TARGET_TRAINED = f"{data_root}/co3d/models/{target_trained_name}/"
    TARGET_DATA = f"{data_root}/co3d/data/{category}/{cameras_model_name}"
    SOURCE_PLY = f"{data_root}/co3d/models/{source_trained_name}/point_cloud/iteration_7000/point_cloud.ply"
    
    # Create output directory for this pair
    pair_output_dir = os.path.join(base_output_dir, f"{source_trained_name}_to_{target_trained_name}")
    os.makedirs(pair_output_dir, exist_ok=True)
    
    # Load models
    gs_target = GaussianModel(3)
    gs_source = GaussianModel(3)
    
    target_ply = os.path.join(TARGET_TRAINED, "point_cloud/iteration_7000/point_cloud.ply")
    gs_target.load_ply(target_ply)
    gs_source.load_ply(SOURCE_PLY)
    
    # Create random transform
    transform_data = create_random_transform(
        min_angle=45,
        max_angle=180,
        translation_range=1.0,
        min_scale=0.5,
        max_scale=1.5
    )
    
    # Save transform data
    with open(os.path.join(pair_output_dir, "transform_gt.json"), 'w') as f:
        json.dump(transform_data, f, indent=4)
    
    T_gt = np.array(transform_data['transform'])
    R_gt = torch.from_numpy(np.array(transform_data['rotation'])).float().cuda()
    t_gt = torch.from_numpy(np.array(transform_data['translation'])).float().cuda()
    s_gt = transform_data['scale']
    
    # Apply random transform to source
    gs_source_random = apply_transform_to_gaussian_model(gs_source, R_gt, t_gt, s_gt)
    
    # Run ICP
    R_init, t_init, s_init = coarse_registration(
        gs_source_random.get_xyz, gs_source_random._semantic_feature,
        gs_target.get_xyz, gs_target._semantic_feature,
        n_samples=35000
    )
    
    # Create ICP transformation matrix
    T_icp = np.eye(4)
    T_icp[:3, :3] = R_init.detach().cpu().numpy()
    T_icp[:3, 3] = t_init.detach().cpu().numpy()
    
    # Compute inverse of ground truth transform
    T_gt_inv = np.linalg.inv(T_gt)
    
    # Compute ICP errors
    ate_icp = compute_ate(T_icp, T_gt_inv)
    rre_icp = compute_rre(T_icp, T_gt_inv)
    
    print(f"ICP Results - ATE: {ate_icp:.4f}, RRE: {rre_icp:.4f}°")
    
    # Run full registration
    final_results = fine_registration(
        target_model_path=TARGET_TRAINED,
        source_model=gs_source_random,
        target_data_path=TARGET_DATA,
        initial_transform=(R_init.detach().cpu().numpy(),
                         t_init.detach().cpu().numpy(),
                         s_init.detach().cpu().numpy())
    )
    
    # Compute final errors
    ate_final = compute_ate(final_results['T_est'], T_gt_inv)
    rre_final = compute_rre(final_results['T_est'], T_gt_inv)
    
    print(f"Final Results - ATE: {ate_final:.4f}, RRE: {rre_final:.4f}°")
    
    # Save results for this pair
    pair_results = {
        'source': source_trained_name,
        'target': target_trained_name,
        'cameras': cameras_model_name,
        'ground_truth': {
            'transform': T_gt.tolist(),
            'scale': float(s_gt),
            'angles': transform_data['angles']
        },
        'icp': {
            'ate': float(ate_icp),
            'rre': float(rre_icp),
        },
        'final': {
            'ate': float(ate_final),
            'rre': float(rre_final),
            'scale': final_results['scale'],
            'best_loss': final_results['best_loss']
        }
    }
    
    with open(os.path.join(pair_output_dir, "results.json"), 'w') as f:
        json.dump(pair_results, f, indent=4)
    
    # Save combined model
    final_combined_path = os.path.join(pair_output_dir, "combined_final.ply")
    save_combined_model(gs_target, final_results['gaussians_output'], final_combined_path)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return {
        'source': source_trained_name,
        'target': target_trained_name,
        'rre_icp': rre_icp,
        'rre_final': rre_final,
        'ate_icp': ate_icp,
        'ate_final': ate_final
    }


# Per-category configuration: the fixed reference-camera sequence (supplies the
# shared rendering viewpoints for the fine stage) and the 10 object sequences
# that get registered pairwise (45 pairs per category).
CO3D_CONFIG = {
    "toyplane": {
        "cameras_model_name": "403_53208_103810",
        "object_ids": [
            "378_44304_88262", "190_20493_39384", "375_42675_85483", "386_46303_92328",
            "387_46731_93046", "404_53743_104802", "420_58343_112950", "420_58285_112763",
            "403_53208_103810", "407_54954_106155",
        ],
    },
    "chair": {
        "cameras_model_name": "350_36826_69026",
        "object_ids": [
            "246_26324_51990", "270_28805_57766", "306_32182_60434", "306_32198_61421",
            "341_35546_65424", "346_36038_66050", "350_36826_69026", "353_37317_70192",
            "372_40895_81174", "372_41207_82069",
        ],
    },
    "bicycle": {
        "cameras_model_name": "62_4324_10701",
        "object_ids": [
            "62_4324_10701", "149_16572_31999", "270_28814_57821", "372_40981_81625",
            "402_52546_102944", "402_52612_103052", "407_55006_106518", "410_55805_108471",
            "426_59713_115789", "430_60705_118821",
        ],
    },
}


def main():
    """Main function to process multiple pairs of CO3D models"""
    cli = ArgumentParser(description="CO3D cross-instance 3DGS registration evaluation")
    cli.add_argument("--data_root", default=None,
                     help="Release data root (contains co3d/data and co3d/models); "
                          "defaults to the GSA_DATA environment variable")
    cli.add_argument("--results_dir", default=None,
                     help="Where to write all outputs (default: ./results/co3d_<category>)")
    cli.add_argument("--seed", type=int, default=None,
                     help="Random seed for the per-pair perturbations. The paper numbers "
                          "were computed WITHOUT a seed; individual pairs vary run-to-run, "
                          "aggregate statistics reproduce.")
    cli.add_argument("--category", required=True, choices=sorted(CO3D_CONFIG),
                     help="CO3D category to evaluate")
    cli.add_argument("--max_pairs", type=int, default=None,
                     help="Limit number of pairs (for smoke tests)")
    cli_args = cli.parse_args()
    if cli_args.data_root is not None:
        os.environ["GSA_DATA"] = cli_args.data_root
    if cli_args.seed is not None:
        np.random.seed(cli_args.seed)
        torch.manual_seed(cli_args.seed)

    # Per-category fixed reference-camera sequence and the 10 object IDs to register.
    category = cli_args.category
    cameras_model_name = CO3D_CONFIG[category]["cameras_model_name"]
    object_ids = CO3D_CONFIG[category]["object_ids"]

    # Create all unique pairs
    pairs = list(combinations(object_ids, 2))
    if cli_args.max_pairs is not None:
        pairs = pairs[:cli_args.max_pairs]
    
    print(f"Processing {len(pairs)} pairs from {len(object_ids)} objects")
    print(f"Camera model: {cameras_model_name}")
    print(f"Category: {category}")
    
    # Create base output directory
    if cli_args.results_dir is not None:
        base_output_dir = cli_args.results_dir
    else:
        base_output_dir = f"./results/co3d_{category}_cam_{cameras_model_name}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    for target_name, source_name in pairs:
        try:
            result = process_pair(
                target_trained_name=target_name,
                source_trained_name=source_name,
                cameras_model_name=cameras_model_name,
                base_output_dir=base_output_dir,
                category=category
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing pair {source_name} -> {target_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'source': source_name,
                'target': target_name,
                'error': str(e),
                'rre_icp': None,
                'rre_final': None,
                'ate_icp': None,
                'ate_final': None
            })
    
    # Calculate statistics
    valid_results = [r for r in all_results if r.get('rre_final') is not None]
    
    if valid_results:
        rre_icp_values = [r['rre_icp'] for r in valid_results]
        rre_final_values = [r['rre_final'] for r in valid_results]
        ate_icp_values = [r['ate_icp'] for r in valid_results]
        ate_final_values = [r['ate_final'] for r in valid_results]
        
        summary = {
            'cameras_model': cameras_model_name,
            'category': category,
            'total_pairs': len(pairs),
            'successful_pairs': len(valid_results),
            'failed_pairs': len(pairs) - len(valid_results),
            'statistics': {
                'rre_icp': {
                    'mean': float(np.mean(rre_icp_values)),
                    'std': float(np.std(rre_icp_values)),
                    'min': float(np.min(rre_icp_values)),
                    'max': float(np.max(rre_icp_values)),
                    'median': float(np.median(rre_icp_values))
                },
                'rre_final': {
                    'mean': float(np.mean(rre_final_values)),
                    'std': float(np.std(rre_final_values)),
                    'min': float(np.min(rre_final_values)),
                    'max': float(np.max(rre_final_values)),
                    'median': float(np.median(rre_final_values))
                },
                'ate_icp': {
                    'mean': float(np.mean(ate_icp_values)),
                    'std': float(np.std(ate_icp_values)),
                    'min': float(np.min(ate_icp_values)),
                    'max': float(np.max(ate_icp_values)),
                    'median': float(np.median(ate_icp_values))
                },
                'ate_final': {
                    'mean': float(np.mean(ate_final_values)),
                    'std': float(np.std(ate_final_values)),
                    'min': float(np.min(ate_final_values)),
                    'max': float(np.max(ate_final_values)),
                    'median': float(np.median(ate_final_values))
                }
            },
            'pair_results': all_results
        }
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH REGISTRATION SUMMARY")
        print("="*60)
        print(f"Camera model: {cameras_model_name}")
        print(f"Category: {category}")
        print(f"Total pairs: {len(pairs)}")
        print(f"Successful: {len(valid_results)}, Failed: {len(pairs) - len(valid_results)}")
        print("\nRRE (Relative Rotation Error) Statistics:")
        print(f"  ICP:   Mean={np.mean(rre_icp_values):.2f}°, Std={np.std(rre_icp_values):.2f}°, Median={np.median(rre_icp_values):.2f}°")
        print(f"  Final: Mean={np.mean(rre_final_values):.2f}°, Std={np.std(rre_final_values):.2f}°, Median={np.median(rre_final_values):.2f}°")
        print("\nATE (Absolute Translation Error) Statistics:")
        print(f"  ICP:   Mean={np.mean(ate_icp_values):.4f}, Std={np.std(ate_icp_values):.4f}")
        print(f"  Final: Mean={np.mean(ate_final_values):.4f}, Std={np.std(ate_final_values):.4f}")
        print("\nPer-pair Results:")
        print("-"*60)
        for r in valid_results:
            print(f"  {r['source']} -> {r['target']}: RRE={r['rre_final']:.2f}°, ATE={r['ate_final']:.4f}")
        
    else:
        summary = {
            'cameras_model': cameras_model_name,
            'category': category,
            'total_pairs': len(pairs),
            'successful_pairs': 0,
            'failed_pairs': len(pairs),
            'pair_results': all_results
        }
        print("\nNo successful registrations!")
    
    # Save summary
    summary_path = os.path.join(base_output_dir, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nAll results saved to: {base_output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()