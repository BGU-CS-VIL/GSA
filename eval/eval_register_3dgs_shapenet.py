# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roy Amoyal, Oren Freifeld, Chaim Baskin

"""ShapeNet cross-instance 3DGS registration evaluation (Table 2).

Registers held-out object pairs within each of 6 ShapeNet categories
(airplane, boat, bus, car, chair, motorcycle), 10 pairs per category, using the
ground-truth perturbations shipped in the data (pair_transforms.json).

    python eval/eval_register_3dgs_shapenet.py
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
from utils.math_utils import build_scaling_rotation, quaternion_to_rotation_matrix, rotation_matrix_to_quaternion,quaternion_multiply,quat_multiply
from torch import nn
import math
from utils.loss_utils import l1_loss, ssim, tv_loss 
import torch.optim as optim
import matplotlib.pyplot as plt

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


import numpy as np
import torch
from scipy.spatial.transform import Rotation
import math

def biased_random_angle(min_angle=45, max_angle=180, bias=0.7):
    """Generate a random angle biased towards higher values.
    
    Args:
        min_angle (float): Minimum angle in degrees
        max_angle (float): Maximum angle in degrees
        bias (float): Bias towards higher values (0.0-1.0)
    
    Returns:
        float: Random angle in degrees
    """
    base = np.random.random()
    biased_base = base ** bias
    angle = min_angle + (max_angle - min_angle) * biased_base
    return angle

def create_random_transform(min_angle=45, max_angle=180, 
                          translation_range=1.0,
                          min_scale=0.5, max_scale=1.5):
    """Create random transformation with biased angles.
    
    Args:
        min_angle (float): Minimum rotation angle in degrees
        max_angle (float): Maximum rotation angle in degrees
        translation_range (float): Range for random translation [-range, range]
        min_scale (float): Minimum scale factor
        max_scale (float): Maximum scale factor
    
    Returns:
        dict: Transform data including matrices and parameters
    """
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
    # T[:3, :3] = R * s  # Apply scale to rotation
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
    """Convert Euler angles to rotation matrix.
    
    Args:
        angles (torch.Tensor): Euler angles in radians [rx, ry, rz]
        device (str): Device to put tensors on
        
    Returns:
        torch.Tensor: 3x3 rotation matrix
    """
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
    """Apply transformation to a Gaussian model.
    
    Args:
        gaussian_model (GaussianModel): Model to transform
        rotation (torch.Tensor): 3x3 rotation matrix
        translation (torch.Tensor): Translation vector
        scale (float): Scale factor
        
    Returns:
        GaussianModel: Transformed copy of input model
    """
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
    
    # Ensure same device
    if src_xyz.device != tgt_xyz.device:
        tgt_xyz = tgt_xyz.to(src_xyz.device)
        tgt_semantic = tgt_semantic.to(src_xyz.device)
    
    # Reshape semantic features if needed
    if len(src_semantic.shape) > 2 or len(tgt_semantic.shape) > 2:
        src_semantic = src_semantic.view(src_semantic.shape[0], -1)
        tgt_semantic = tgt_semantic.view(tgt_semantic.shape[0], -1)
    
    # Compute all pairwise semantic distances at once
    # src_semantic: [N, F], tgt_semantic: [M, F]
    # Result: [N, M] where result[i,j] = distance between src[i] and tgt[j]
    semantic_distances = torch.cdist(src_semantic, tgt_semantic)
    
    # Create mask for valid semantic correspondences
    semantic_mask = semantic_distances < semantic_thresh  # [N, M]
    
    # For each source point, check if it has any valid semantic matches
    has_valid_matches = semantic_mask.any(dim=1)  # [N]
    
    if not has_valid_matches.any():
        return []
    
    # Compute all pairwise spatial distances
    spatial_distances = torch.cdist(src_xyz, tgt_xyz)  # [N, M]
    
    # Set spatial distances to infinity where semantic match is invalid
    masked_spatial_distances = spatial_distances.clone()
    masked_spatial_distances[~semantic_mask] = float('inf')
    
    # Find the closest valid target for each source point
    min_distances, closest_indices = torch.min(masked_spatial_distances, dim=1)  # [N]
    
    # Filter out source points that have no valid matches (distance = inf)
    valid_sources = min_distances != float('inf')
    
    # Create correspondences list
    valid_source_indices = torch.where(valid_sources)[0]
    valid_target_indices = closest_indices[valid_sources]
    
    # Convert to list of tuples for compatibility with existing code
    correspondences = [(src_idx.item(), tgt_idx.item()) 
                      for src_idx, tgt_idx in zip(valid_source_indices, valid_target_indices)]
    
    return correspondences


def coarse_registration(source_xyz, source_semantic, target_xyz, target_semantic, 
                        n_samples=None, max_iter=3, semantic_thresh=0.01):
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

        
        # Clamp scale to reasonable range
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

def load_or_create_transforms(category_dir, object_ids, work_dir, category):
    """Load existing transforms or create new ones for all pairs"""
    transform_file = os.path.join(category_dir, "pair_transforms.json")

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
        os.makedirs(os.path.join(work_dir, category), exist_ok=True)
        with open(os.path.join(work_dir, category, "pair_transforms.json"), 'w') as f:
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
    
def process_object_pair(obj1_id, obj2_id, category_dir, model_dir, work_dir, transforms, category,
                        icp_samples=50000):
    """Process a single pair of objects with improved transform handling"""
    pair_key = f"{min(obj1_id, obj2_id)}_{max(obj1_id, obj2_id)}"
    transform_data = transforms[pair_key]

    # Create combined models directory (all writes go to the local work dir,
    # the model/data dirs of the release drop stay read-only)
    combined_models_dir = os.path.join(work_dir, category, "combined_models", pair_key)
    os.makedirs(combined_models_dir, exist_ok=True)

    # Construct paths
    target_model_path = os.path.join(model_dir, category, obj1_id)
    source_model_path = os.path.join(model_dir, category, obj2_id)
    target_data_path = os.path.join(category_dir, obj1_id)
    
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
        n_samples=icp_samples
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
    
    # Before running training, save the randomly transformed model
    # (into the work dir — the release model dirs are read-only)
    random_transformed_path = os.path.join(combined_models_dir, "random_transform.ply")
    gs2_random.save_ply(random_transformed_path)

    # Run full registration starting from ICP result
    final_results = fine_registration(
        target_model_path=target_model_path,
        source_model_path=source_model_path,  # Use the randomly transformed model
        target_data_path=target_data_path,
        parent_dir=category_dir,
        initial_transform=(R_init.detach().cpu().numpy(),
                        t_init.detach().cpu().numpy(),
                        s_init.detach().cpu().numpy()),
        source_random_ply=random_transformed_path,
        align_save_path=os.path.join(combined_models_dir, "full_align.ply")
    )
    
    # Load final transformed model and create combined version
    gs2_final = GaussianModel(3)
    gs2_final.load_ply(final_results['transformed_model_path'])
    final_combined_path = os.path.join(combined_models_dir, "combined_final.ply")
    save_combined_model(gs1, gs2_final, final_combined_path)
    
    # Compute final errors against inverse ground truth
    ate_final = compute_ate(final_results['T_est'], T_gt_inv)
    rre_final = compute_rre(final_results['T_est'], T_gt_inv)
    
    # Detailed rotation analysis
    R_est = final_results['T_est'][:3, :3]
    R_gt_np = T_gt_inv[:3, :3]
    rot_est = Rotation.from_matrix(R_est)
    rot_gt = Rotation.from_matrix(R_gt_np)
    
    euler_est = rot_est.as_euler('xyz', degrees=True)
    euler_gt = rot_gt.as_euler('xyz', degrees=True)
    euler_diff = np.abs(euler_est - euler_gt)
    
    t_est = final_results['T_est'][:3, 3]
    t_gt = T_gt[:3, 3]
    
    result = {
        'pair': pair_key,
        'icp_ate': float(ate_icp),
        'icp_rre': float(rre_icp),
        'final_ate': float(ate_final),
        'final_rre': float(rre_final),
        'scale': final_results['scale'],
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
    
    return result
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


def geman_mcclure_loss(point_diff, sigma=1.0):
    """
    Geman-McClure robust loss function
    
    Args:
        point_diff: Difference between transformed source and target points [N, 3]
        sigma: Scale parameter that controls the transition point between quadratic and linear behavior
    
    Returns:
        Robust loss value
    """
    # Compute squared residuals
    squared_residuals = torch.sum(point_diff ** 2, dim=1)  # [N]
    
    # Geman-McClure loss: r^2 / (sigma^2 + r^2)
    loss = squared_residuals / (sigma**2 + squared_residuals)
    
    return torch.mean(loss)

def fine_registration(target_model_path, source_model_path, target_data_path, parent_dir, initial_transform=None,
             source_random_ply=None, align_save_path=None, images="train", views_stride=-1,
             iteration=7000, n_fine_views=5):
    """Training function for registration with geometric loss"""
    print("\nStarting training with paths:")
    print(f"Target model path: {target_model_path}")
    print(f"Source model path: {source_model_path}")
    print(f"Target data path: {target_data_path}")

    # Setup arguments for dataset
    class DummyArgs:
        def __init__(self, source_path, model_path):
            self.source_path = source_path
            self.model_path = model_path
            self.images = images
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = False
            self.render_items = ['RGB', 'Depth', 'Edge', 'Normal', 'Curvature', 'Feature Map']
            self.sh_degree = 3
    
    # Create dataset model
    dummy_args = DummyArgs(source_path=target_data_path, model_path=target_model_path)
    parser = ArgumentParser()
    dataset_model = ModelParams(parser).extract(dummy_args)
    pipe = PipelineParams(parser)
    
    # Load models
    gaussians_target = GaussianModel(3)
    gaussians_source = GaussianModel(3)
    gaussians_output = GaussianModel(3)
    
    print("Loading models...")
    target_ply = os.path.join(target_model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    if source_random_ply is not None:
        source_ply = source_random_ply
    else:
        source_ply = os.path.join(source_model_path, f"point_cloud/iteration_{iteration}/random_transform.ply")

    gaussians_target.load_ply(target_ply)
    gaussians_source.load_ply(source_ply)
    gaussians_output.load_ply(source_ply)
    
    # Setup scene and cameras
    scene = Scene(dataset_model, gaussians_target, load_iteration=-1, shuffle=False)
    # Select n_fine_views diverse (evenly spread) views for the fine stage.
    # views_stride < 0 -> derive a stride that yields n_fine_views; otherwise use it directly.
    all_cams = scene.getTrainCameras().copy()
    if views_stride is None or views_stride < 0:
        stride = max(1, len(all_cams) // n_fine_views)
    else:
        stride = views_stride
    viewpoint_stack = all_cams[::stride]
    print(f"Viewpoint stack length: {len(viewpoint_stack)}")
    
    # Setup background color
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Apply initial transformation from ICP
    if initial_transform is not None:
        print("Applying initial ICP transformation...")
        R_init, t_init, s_init = initial_transform
        
        # Convert rotation matrix to euler angles
        from scipy.spatial.transform import Rotation
        euler_angles = rotation_matrix_to_euler_angles(torch.tensor(R_init, device="cuda"))
        print("Initial Euler angles:", euler_angles * (180/math.pi))
        
        # Set initial transformation
        gaussians_source.global_rotation.data = euler_angles
        gaussians_source.global_translation.data = torch.tensor(t_init, device="cuda", dtype=torch.float32)
        gaussians_source.global_scale.data = torch.tensor([float(s_init)], device="cuda", dtype=torch.float32)

    # Render feature maps for target model
    print("Rendering target feature maps...")
    target_feature_maps = []
    for viewpoint_cam in viewpoint_stack:
        viewpoint_cam = viewpoint_cam.cuda()
        render_pkg = render(viewpoint_cam, gaussians_target, pipe, background)
        target_feature_maps.append(render_pkg["feature_map"])
    
    # Setup optimization
    optimizer = optim.AdamW([
        {'params': gaussians_source.global_translation, 'lr': 0.01},
        {'params': gaussians_source.global_rotation, 'lr': 0.01},
        {'params': gaussians_source.global_scale, 'lr': 0.01},        
    ])
    
    # Track best transformation
    best_loss = float('inf')
    best_transform = {
        'rotation': None,
        'translation': None,
        'scale': None
    }
    
    # Parameters for geometric loss
    geometric_weight = 0.1  # Weight for geometric loss relative to render loss
    n_correspondence_samples = 50000  # Number of points to sample for correspondences
    semantic_thresh = 0.01  # Threshold for semantic correspondence matching
    
    # Training loop
    print("Starting optimization...")
    for iteration in range(60):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
        render_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
        geometric_loss = torch.tensor(0.0, device="cuda", requires_grad=True)

        # Render loss - multi-view feature matching
        for i, viewpoint_cam in enumerate(viewpoint_stack):
            render_pkg = render_registration(
                viewpoint_cam, 
                gaussians_source, 
                pipe, 
                background,
                override_color=gaussians_source.get_semantic_feature.squeeze(1)
            )
            source_feature_map = render_pkg["render"]
               # Simple mask - ignore near-zero pixels (background)
            # Create individual masks for non-background pixels
            source_mask = torch.sum(torch.abs(source_feature_map), dim=0) > 0.01
            target_mask = torch.sum(torch.abs(target_feature_maps[i]), dim=0) > 0.01

            # Get pixels that are valid in either source or target (union of masks)
            combined_mask = source_mask | target_mask

            if combined_mask.sum() > 0:
                # Extract pixels using the combined mask
                source_masked = source_feature_map[:, combined_mask]
                target_masked = target_feature_maps[i][:, combined_mask]
                
                # Set background pixels to zero in the masked arrays
                source_valid = source_mask[combined_mask]  # Which pixels in the masked array are valid for source
                target_valid = target_mask[combined_mask]  # Which pixels in the masked array are valid for target
                
                source_masked[:, ~source_valid] = 0
                target_masked[:, ~target_valid] = 0
                
                loss = torch.nn.SmoothL1Loss()(source_masked, target_masked)
            else:
                loss = torch.tensor(0.0, device=source_feature_map.device, requires_grad=True)
            render_loss = render_loss + loss
        
        
        # Combine losses
        total_loss = render_loss 

        # Check if this is the best loss so far
        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_transform['rotation'] = gaussians_source.global_rotation.data.clone()
            best_transform['translation'] = gaussians_source.global_translation.data.clone()
            best_transform['scale'] = gaussians_source.global_scale.data.clone()
            print(f"New best loss at iteration {iteration}: {best_loss:.6f}")
            print(f"  Render loss: {render_loss.item():.6f}")

        print(f"Iteration {iteration}, Total Loss: {total_loss.item():.6f}")
        print(f"  Render: {render_loss.item():.6f}, Geometric: {geometric_loss.item():.6f}")
        print(f"  Rotation: {gaussians_source.global_rotation.detach().cpu().numpy() * (180/math.pi)}°")
        print(f"  Translation: {gaussians_source.global_translation.detach().cpu().numpy()}")
        print(f"  Scale: {gaussians_source.global_scale.detach().cpu().numpy()}")
        
        total_loss.backward(retain_graph=True)
        optimizer.step()

    # Apply best transformation for final output
    print("\nApplying best transformation found...")
    print(f"Best loss achieved: {best_loss}")
    
    # Set the model to best transformation
    gaussians_source.global_rotation.data = best_transform['rotation']
    gaussians_source.global_translation.data = best_transform['translation']
    gaussians_source.global_scale.data = best_transform['scale']
    
    # Set scale to 1.0 for final output (if desired)
    gaussians_source.global_scale.data = torch.tensor([1.0], device="cuda")
    
    # Get final transformation with best parameters
    means3D, rotations, scaling, _, _ = gaussians_source.apply_global_transformation()
    gaussians_output._xyz = means3D
    gaussians_output._rotation = rotations
    gaussians_output._scaling = scaling
    
    # Save transformed model (into the work dir — model dirs stay read-only)
    if align_save_path is not None:
        save_path = align_save_path
    else:
        save_path = os.path.join(source_model_path, "point_cloud/iteration_7000/full_align.ply")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gaussians_output.save_ply(save_path)

    # Compute final transformation matrix using best transform
    final_angles = best_transform['rotation'].detach().cpu().numpy()
    final_rotation = Rotation.from_euler('xyz', final_angles).as_matrix()
    final_translation = best_transform['translation'].detach().cpu().numpy()
    final_scale = best_transform['scale'].detach().cpu().numpy().item()
    
    T_final = np.eye(4)
    T_final[:3, :3] = final_rotation
    T_final[:3, 3] = final_translation

    return {
        'T_est': T_final,
        'scale': final_scale,
        'transformed_model_path': save_path,
        'best_loss': best_loss
    }


def save_ply_pair(obj1_id, obj2_id, category_dir, model_dir, work_dir, transforms, category):
    """Save PLY pair and transformation data for a given object pair"""
    pair_key = f"{min(obj1_id, obj2_id)}_{max(obj1_id, obj2_id)}"
    transform_data = transforms[pair_key]

    # Create ply_pairs directory for the category (in the local work dir)
    ply_pairs_dir = os.path.join(work_dir, category, "ply_pairs")
    os.makedirs(ply_pairs_dir, exist_ok=True)

    # Create directory for this specific pair
    pair_dir = os.path.join(ply_pairs_dir, pair_key)
    os.makedirs(pair_dir, exist_ok=True)

    # Paths for source files
    target_ply_path = os.path.join(model_dir, category, obj1_id, "point_cloud/iteration_7000/point_cloud.ply")
    source_ply_path = os.path.join(model_dir, category, obj2_id, "point_cloud/iteration_7000/point_cloud.ply")
    
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



def main():
    """Main function to process ShapeNet dataset"""
    cli = ArgumentParser(description="ShapeNet cross-instance 3DGS registration evaluation")
    cli.add_argument("--data_root", default=os.environ.get("GSA_DATA", "./GSA_release_data"),
                     help="Release data root (contains shapenet/data and shapenet/models)")
    cli.add_argument("--results_dir", default="./results/shapenet",
                     help="Where to write all outputs (results json, combined models, ply pairs)")
    cli.add_argument("--category", default=None, help="Only run this category")
    cli.add_argument("--max_pairs", type=int, default=None,
                     help="Limit number of pairs per category (for smoke tests)")
    cli.add_argument("--skip_ply_pairs", action="store_true",
                     help="Skip exporting the per-pair target/source ply visualization files")
    cli.add_argument("--icp_samples", type=int, default=50000,
                     help="Gaussians sampled for the semantic ICP. The paper used 50000, "
                          "which needs a ~48 GB GPU; use 30000 on a 24 GB GPU")
    cli_args = cli.parse_args()

    # Data (read-only) and trained models (read-only) come from the release drop;
    # every output of this script goes under --results_dir.
    base_dir = os.path.join(cli_args.data_root, "shapenet", "data")
    model_dir = os.path.join(cli_args.data_root, "shapenet", "models")
    work_dir = cli_args.results_dir
    results_dir = os.path.join(work_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    print("All categories:", os.listdir(base_dir))

    # Process each category
    for category in sorted(os.listdir(base_dir)):
        category_dir = os.path.join(base_dir, category)
        if not os.path.isdir(category_dir):
            continue
        if cli_args.category is not None and category != cli_args.category:
            continue
        print(f"\nProcessing category: {category}")

        # Get object IDs
        object_ids = [d for d in os.listdir(category_dir)
                     if os.path.isdir(os.path.join(category_dir, d))]

        # Load or create transform pairs
        transforms = load_or_create_transforms(category_dir, object_ids, work_dir, category)

        pair_list = list(combinations(object_ids, 2))
        if cli_args.max_pairs is not None:
            pair_list = pair_list[:cli_args.max_pairs]

        # First, save all PLY pairs
        if not cli_args.skip_ply_pairs:
            print(f"Saving PLY pairs for {category}...")
            for obj1_id, obj2_id in tqdm(pair_list,
                                        desc=f"Saving PLY pairs for {category}"):
                save_ply_pair(obj1_id, obj2_id, category_dir, model_dir, work_dir, transforms, category)

        # Process all pairs for registration
        all_results = []
        all_ate_icp = []
        all_rre_icp = []
        all_ate_final = []
        all_rre_final = []

        for obj1_id, obj2_id in tqdm(pair_list,
                                    desc=f"Processing {category} pairs"):
            result = process_object_pair(
                obj1_id, obj2_id, category_dir, model_dir, work_dir, transforms, category,
                icp_samples=cli_args.icp_samples)
            
            if result is not None:
                all_results.append(result)
                all_ate_icp.append(result['icp_ate'])
                all_rre_icp.append(result['icp_rre'])
                all_ate_final.append(result['final_ate'])
                all_rre_final.append(result['final_rre'])
                
                tqdm.write(f"\nResults for {obj1_id}-{obj2_id}:")
                tqdm.write(f"ICP  - ATE: {result['icp_ate']:.4f} meters, RRE: {result['icp_rre']:.4f} degrees")
                tqdm.write(f"Final- ATE: {result['final_ate']:.4f} meters, RRE: {result['final_rre']:.4f} degrees")
        
        # Compute and save results as before
        if all_results:
            mean_ate_icp = np.mean(all_ate_icp)
            mean_rre_icp = np.mean(all_rre_icp)
            std_ate_icp = np.std(all_ate_icp)
            std_rre_icp = np.std(all_rre_icp)
            
            mean_ate_final = np.mean(all_ate_final)
            mean_rre_final = np.mean(all_rre_final)
            std_ate_final = np.std(all_ate_final)
            std_rre_final = np.std(all_rre_final)
            
            results_path = os.path.join(results_dir, f"{category}_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'individual_results': all_results,
                    'summary': {
                        'icp': {
                            'mean_ate': float(mean_ate_icp),
                            'std_ate': float(std_ate_icp),
                            'mean_rre': float(mean_rre_icp),
                            'std_rre': float(std_rre_icp)
                        },
                        'final': {
                            'mean_ate': float(mean_ate_final),
                            'std_ate': float(std_ate_final),
                            'mean_rre': float(mean_rre_final),
                            'std_rre': float(std_rre_final)
                        }
                    }
                }, f, indent=4)
            
            print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()