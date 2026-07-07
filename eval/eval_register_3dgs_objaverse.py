# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roy Amoyal, Oren Freifeld, Chaim Baskin

"""Objaverse same-object 3DGS registration evaluation.

Each object is captured as two disjoint view blocks (block0 / block1) that are
reconstructed independently; the two 3DGS models are then registered back into a
common frame (ground truth in world_frame_transforms.json).

    python eval/eval_register_3dgs_objaverse.py
"""

# Allow running this script directly from eval/ (make the repo root importable).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_registration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
def compute_ate(T_est, T_gt):
    """Compute Absolute Trajectory Error (ATE)"""
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]
    error = np.linalg.norm(t_est - t_gt)
    return error

def compute_rre(T_est, T_gt):
    from scipy.spatial.transform import Rotation

    """
    Compute Relative Rotation Error (RRE) between estimated and ground truth transforms
    using quaternion-based calculations for maximum numerical stability.
    
    This function uses scipy's Rotation class to handle the rotation calculations,
    which provides better numerical stability and mathematical correctness compared
    to the trace-based method. The error is computed as the geodesic distance
    between the two rotations on SO(3).
    
    Args:
        T_est: 4x4 estimated transformation matrix
        T_gt: 4x4 ground truth transformation matrix
    
    Returns:
        float: The angular difference in degrees between the two rotations
    """
    # Extract the 3x3 rotation matrices from the transformation matrices
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    
    # Convert rotation matrices to Rotation objects
    # This internally handles the conversion to quaternions and ensures proper normalization
    rot_est = Rotation.from_matrix(R_est)
    rot_gt = Rotation.from_matrix(R_gt)
    
    # Compute the relative rotation that transforms one orientation to the other
    # This is equivalent to rot_gt * rot_est^(-1) in quaternion multiplication
    rot_diff = rot_gt * rot_est.inv()
    
    # Get the magnitude of the rotation difference
    # This computes the geodesic distance on SO(3), which is the shortest possible
    # angle between the two rotations
    angle_rad = rot_diff.magnitude()
    
    # Convert to degrees for easier interpretation
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def find_color_valid_correspondences(src_xyz, src_semantic, tgt_xyz, tgt_semantic, 
                                  dist_thresh=0.5, semantic_thresh=0.1, semantic_weight=0.90):
    if src_xyz.device != tgt_xyz.device:
        tgt_xyz = tgt_xyz.to(src_xyz.device)
        tgt_semantic = tgt_semantic.to(src_xyz.device)
    
    if len(src_semantic.shape) > 2 or len(tgt_semantic.shape) > 2:
        src_semantic = src_semantic.view(src_semantic.shape[0], -1)
        tgt_semantic = tgt_semantic.view(tgt_semantic.shape[0], -1)
    
    correspondences = []
    for i in range(src_xyz.shape[0]):
        spatial_distances = torch.norm(tgt_xyz - src_xyz[i], dim=1)
        semantic_distances = torch.norm(tgt_semantic - src_semantic[i], dim=1)
        semantic_mask = semantic_distances < semantic_thresh
        
        if semantic_mask.any():
            valid_spatial_distances = spatial_distances[semantic_mask]
            closest_idx = torch.argmin(valid_spatial_distances)
            correspondences.append((i, torch.where(semantic_mask)[0][closest_idx].item()))
    
    return correspondences

def coarse_registration(source_xyz, source_semantic, target_xyz, target_semantic, 
                        n_samples=None, max_iter=6, semantic_thresh=0.01):
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
        
        numerator = torch.sum(torch.sum(tgt_centered * tgt_centered, dim=1))
        denominator = torch.sum(torch.sum(src_centered * src_centered, dim=1))
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

                # Update final transformation
        R_final = R @ R_final
        t_final = scale * (R @ t_final) + t
        s_final = s_final * scale
        
        # Transform source sample points
        source_xyz_sample = (scale * (R @ source_xyz_sample.T)).T + t
    
    print("ICP finished.")
    print(f"Final rotation matrix:\n{R_final}")
    print(f"Final translation: {t_final}")
    print(f"Final scale: {s_final}")
    
    
    pbar.close()
    return R_final, t_final, s_final

def loss_mse(render_img, gt):
    loss_fn = torch.nn.MSELoss()
    return loss_fn(render_img, gt)


def combine_rotations(angles):
    """Convert Euler angles to rotation matrix"""
    R_x = torch.stack([
        torch.stack([torch.ones_like(angles[0]), torch.zeros_like(angles[0]), torch.zeros_like(angles[0])]),
        torch.stack([torch.zeros_like(angles[0]), torch.cos(angles[0]), -torch.sin(angles[0])]),
        torch.stack([torch.zeros_like(angles[0]), torch.sin(angles[0]), torch.cos(angles[0])])
    ])
    
    R_y = torch.stack([
        torch.stack([torch.cos(angles[1]), torch.zeros_like(angles[1]), torch.sin(angles[1])]),
        torch.stack([torch.zeros_like(angles[1]), torch.ones_like(angles[1]), torch.zeros_like(angles[1])]),
        torch.stack([-torch.sin(angles[1]), torch.zeros_like(angles[1]), torch.cos(angles[1])])
    ])
    
    R_z = torch.stack([
        torch.stack([torch.cos(angles[2]), -torch.sin(angles[2]), torch.zeros_like(angles[2])]),
        torch.stack([torch.sin(angles[2]), torch.cos(angles[2]), torch.zeros_like(angles[2])]),
        torch.stack([torch.zeros_like(angles[2]), torch.zeros_like(angles[2]), torch.ones_like(angles[2])])
    ])
    
    return torch.matmul(torch.matmul(R_z, R_y), R_x)

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

def fine_registration(block0_path, block1_path, src_model_path, parent_dir, obj_dir, initial_transform=None,
             transformed_save_path=None):
    """
    Simplified training function that takes direct paths instead of args
    """
    print("Starting training with paths:")
    print(f"block0_path: {block0_path}")
    print(f"block1_path: {block1_path}")
    print(f"src_model_path: {src_model_path}")

    # Create a dataset model for block0 to load cameras
    class DummyArgs:
        def __init__(self, source_path, model_path):
            self.source_path = source_path       # Path to original images
            self.model_path = model_path         # Path to model/point cloud
            self.images = "train"               # Use default value from ModelParams
            self.resolution = -1                 # Default from ModelParams
            self.white_background = False        # Default from ModelParams
            self.data_device = "cuda"           # Default from ModelParams
            self.eval = False                   # Default from ModelParams
            self.render_items = ['RGB', 'Depth', 'Edge', 'Normal', 'Curvature', 'Feature Map']
            self.sh_degree = 3                  # Default from ModelParams
            
    # Correct source path for images
    source_path = os.path.join(parent_dir, obj_dir, "block0")
    dummy_args = DummyArgs(source_path=source_path, model_path=block0_path)
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pipe = PipelineParams(parser)
    dataset_model = lp.extract(dummy_args)
    
    
    gaussians_A = GaussianModel(3)  # Using sh_degree=3 directly
    gaussians_B = GaussianModel(3)
    gaussians_C = GaussianModel(3)
    
    print("Loading models...")
    # First load point cloud for gaussians_A
    gaussian_A_path = os.path.join(block0_path, "point_cloud/iteration_7000/point_cloud.ply")
    gaussians_A.load_ply(gaussian_A_path)
    gaussians_B.load_ply(src_model_path)
    gaussians_C.load_ply(src_model_path)

    # Create scene to get cameras
    scene_A = Scene(dataset_model, gaussians_A, load_iteration=-1, shuffle=False)
    viewpoint_stack = scene_A.getTrainCameras().copy()
    
    # Select every second frame
    viewpoint_stack = viewpoint_stack[::20]
    print(f"Loaded {len(viewpoint_stack)} camera views")
    print("Initial transformation:")
    print(f"R_init: {initial_transform[0]}")
    print(f"t_init: {initial_transform[1]}")
    print(f"s_init: {initial_transform[2]}")
    # Apply initial transformation if provided
    if initial_transform is not None:
        R_init, t_init, s_init = initial_transform
        # Convert rotation matrix to euler angles
        euler_angles = rotation_matrix_to_euler_angles(torch.tensor(R_init, device="cuda"))
        print("euler_angles",euler_angles)
        # print euler angles in degrees
        from numpy import pi
        print("euler_angles",euler_angles*(180/pi))
        print(gaussians_B.global_rotation.data)
        gaussians_B.global_rotation.data = euler_angles.data
        print("after",gaussians_B.global_rotation.data)

        gaussians_B.global_translation.data = torch.tensor(t_init, device="cuda")
        # Convert s_init to a single value if it's not already
        s_init_value = s_init.item() if torch.is_tensor(s_init) else float(s_init)
        gaussians_B.global_scale.data = torch.tensor([s_init_value], device="cuda")
        # Setup rendering parameters
        bg_color = [0, 0, 0]  # black background
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Render feature maps for target model
    feature_maps_queries = []
    original_rendered_images = []
    for viewpoint_cam in viewpoint_stack:
        viewpoint_cam = viewpoint_cam.cuda()
        render_pkg = render(viewpoint_cam, gaussians_A, pipe, background)
        feature_maps_queries.append(render_pkg["feature_map"])

        # original rendered images
        original_rendered_images.append(render_pkg["render"])
    # Setup optimizer
    optimizer = optim.Adam([gaussians_B.global_translation,
                          gaussians_B.global_rotation,
                          gaussians_B.global_scale], lr=0.01)
    # Track the best transformation across iterations (the loop can overshoot,
    # so we keep the lowest-loss transform rather than the last one).
    best_loss = float('inf')
    best_transform = {'rotation': None, 'translation': None, 'scale': None}
    # Training loop
    print("Starting optimization...")
    for iteration in range(80):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device="cuda", requires_grad=True)

        for i, viewpoint_cam in enumerate(viewpoint_stack):
            render_pkg = render_registration(viewpoint_cam, gaussians_B, pipe, background,
                                          override_color=gaussians_B.get_semantic_feature.squeeze(1))
            # The source feature map is the semantic feature rendered as
            # override-color UNDER the optimized global transform: render_pkg["render"].
            # (render_pkg["feature_map"] does not depend on the global transform,
            # which froze the loss and made the fine stage a no-op.)
            feature_map, image = render_pkg["render"], render_pkg["render"]
            # render depth image
            depth_image = render_pkg["depth"]


            # ORIGINAL LOSS ON FEATURES MAPS

            Ll1_feature = loss_mse(feature_map.cuda().float(), feature_maps_queries[i].cuda().float())
            total_loss = total_loss + Ll1_feature


        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_transform['rotation'] = gaussians_B.global_rotation.data.clone()
            best_transform['translation'] = gaussians_B.global_translation.data.clone()
            best_transform['scale'] = gaussians_B.global_scale.data.clone()

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {total_loss.item()}")
        total_loss.backward(retain_graph=True)
        optimizer.step()

    # Restore the best transformation found before producing the final output.
    if best_transform['rotation'] is not None:
        gaussians_B.global_rotation.data = best_transform['rotation']
        gaussians_B.global_translation.data = best_transform['translation']
        gaussians_B.global_scale.data = best_transform['scale']
    print(f"Best fine loss: {best_loss:.6f}")

    # Get final transformation
    means3D, rotations, scaling, feat_, feat2_ = gaussians_B.apply_global_transformation()
    gaussians_C._xyz = means3D
    gaussians_C._rotation = rotations
    gaussians_C._scaling = scaling
    
    # Save transformed model (to the local results dir — the release model dirs are read-only)
    if transformed_save_path is not None:
        save_path = transformed_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = src_model_path.replace('.ply', '_transformed.ply')
    gaussians_C.save_ply(save_path)

      # Get final transformation
    T_final = np.eye(4)
    final_angles = gaussians_B.global_rotation.detach().cpu().numpy()
    final_rotation = combine_rotations(torch.from_numpy(final_angles)).detach().cpu().numpy()
    final_translation = gaussians_B.global_translation.detach().cpu().numpy()
    final_scale = gaussians_B.global_scale.detach().cpu().numpy()

    T_final[:3, :3] = final_rotation
    T_final[:3, 3] = final_translation

    return {
        'T_est': T_final,
        'scale': final_scale.item(),
        'transformed_model_path': save_path
    }




def process_single_object(obj_dir, parent_dir, output_dir, results_dir="./results/objaverse"):
    print(f"Starting process_single_object for {obj_dir}")

    block0_path = os.path.join(output_dir, f"{obj_dir}_block0")
    block1_path = os.path.join(output_dir, f"{obj_dir}_block1/point_cloud/iteration_7000/point_cloud.ply")
    transforms_path = os.path.join(parent_dir, obj_dir, "world_frame_transforms.json")

    # Debug prints
    print(f"Checking paths:")
    print(f"block0_path: {block0_path} exists: {os.path.exists(block0_path)}")
    print(f"block1_path: {block1_path} exists: {os.path.exists(block1_path)}")
    print(f"transforms_path: {transforms_path} exists: {os.path.exists(transforms_path)}")

    if not all(os.path.exists(p) for p in [block0_path, block1_path, transforms_path]):
        print(f"Missing required files for {obj_dir}")
        return None

    # Run ICP first...
    print(f"\nRunning ICP for {obj_dir}")
    gs1 = GaussianModel(3)
    gs2 = GaussianModel(3)
    gs1.load_ply(os.path.join(block0_path, "point_cloud/iteration_7000/point_cloud.ply"))
    gs2.load_ply(block1_path)

    R_init, t_init, s_init = coarse_registration(
        gs2.get_xyz, gs2._semantic_feature,
        gs1.get_xyz, gs1._semantic_feature,
        n_samples=80000
    )

    # Load ground truth and compute ICP errors...
    with open(transforms_path, 'r') as f:
        gt_transforms = json.load(f)
    T_gt_0 = np.array(gt_transforms['0'][0])
    T_gt_1 = np.array(gt_transforms['1'][0])
    T_gt_rel = T_gt_0 @ np.linalg.inv(T_gt_1)

    T_icp = np.eye(4)
    T_icp[:3, :3] = R_init.detach().cpu().numpy()
    T_icp[:3, 3] = t_init.detach().cpu().numpy()

    ate_icp = compute_ate(T_icp, T_gt_rel)
    rre_icp = compute_rre(T_icp, T_gt_rel)


    print("Starting training")
    final_results = fine_registration(
        block0_path=block0_path,
        block1_path=block1_path,
        src_model_path=block1_path,
        parent_dir=parent_dir,
        obj_dir=obj_dir,
        initial_transform=(R_init.detach().cpu().numpy(),
                         t_init.detach().cpu().numpy(),
                         s_init.detach().cpu().numpy()),
        transformed_save_path=os.path.join(results_dir, "transformed_models",
                                           f"{obj_dir}_block1_transformed.ply")
    )
    print("Training completed")
    
    
    
    
    
    ate_final = compute_ate(final_results['T_est'], T_gt_rel)
    rre_final = compute_rre(final_results['T_est'], T_gt_rel)
    
     # Extract rotation components for detailed analysis
    from scipy.spatial.transform import Rotation
    
    # Create rotation objects for estimated and ground truth
    R_est = final_results['T_est'][:3, :3]
    R_gt = T_gt_rel[:3, :3]
    rot_est = Rotation.from_matrix(R_est)
    rot_gt = Rotation.from_matrix(R_gt)
    
    # Get Euler angles for both rotations
    euler_est = rot_est.as_euler('xyz', degrees=True)
    euler_gt = rot_gt.as_euler('xyz', degrees=True)
    euler_diff = np.abs(euler_est - euler_gt)
    
    # Extract translation vectors
    t_est = final_results['T_est'][:3, 3]
    t_gt = T_gt_rel[:3, 3]
    
    # Print detailed rotation and translation analysis
    print("\nRotation Error Analysis:")
    print("-" * 50)
    print(f"Total Rotation Error: {rre_final:.5f}°")
    print(f"                      {rre_final * np.pi / 180:.5f} radians")
    
    print("\nPer-axis Analysis (in degrees):")
    print(f"Roll  (X): Estimated = {euler_est[0]:7.5f}°  "
          f"Ground Truth = {euler_gt[0]:7.5f}°  "
          f"Difference = {euler_diff[0]:7.5f}°")
    print(f"Pitch (Y): Estimated = {euler_est[1]:7.5f}°  "
          f"Ground Truth = {euler_gt[1]:7.5f}°  "
          f"Difference = {euler_diff[1]:7.5f}°")
    print(f"Yaw   (Z): Estimated = {euler_est[2]:7.5f}°  "
          f"Ground Truth = {euler_gt[2]:7.5f}°  "
          f"Difference = {euler_diff[2]:7.5f}°")
    
    print("\nTranslation Vectors:")
    print(f"Estimated:    {t_est}")
    print(f"Ground Truth: {t_gt}")
    
    # Add the detailed analysis to the results dictionary
    result = {
        'object': obj_dir,
        'icp_ate': ate_icp,
        'icp_rre': rre_icp,
        'final_ate': ate_final,
        'final_rre': rre_final,
        'scale': final_results['scale'],
        'T_est_icp': T_icp,
        'T_est_final': final_results['T_est'],
        'T_gt': T_gt_rel,
        # Add new detailed analysis fields
        'rotation_analysis': {
            'euler_angles': {
                'estimated': euler_est.tolist(),
                'ground_truth': euler_gt.tolist(),
                'differences': euler_diff.tolist()
            },
            'radians': rre_final * np.pi / 180
        },
        'translation_vectors': {
            'estimated': t_est.tolist(),
            'ground_truth': t_gt.tolist()
        }
    }
    
    return result
    
    
    
    
    
def main():
    cli = ArgumentParser(description="Objaverse same-object 3DGS registration evaluation")
    cli.add_argument("--data_root", default=os.environ.get("GSA_DATA", "./GSA_release_data"),
                     help="Release data root (contains objaverse/data and objaverse/models)")
    cli.add_argument("--results_dir", default="./results/objaverse",
                     help="Where to write all outputs")
    cli.add_argument("--max_objects", type=int, default=None,
                     help="Limit number of objects (for smoke tests)")
    cli_args = cli.parse_args()

    parent_dir = os.path.join(cli_args.data_root, "objaverse", "data")
    output_dir = os.path.join(cli_args.data_root, "objaverse", "models")
    results_dir = cli_args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    all_ate_icp = []
    all_rre_icp = []
    all_ate_final = []
    all_rre_final = []
    
    # Get list of valid directories first
    obj_dirs = [d for d in os.listdir(parent_dir)
               if os.path.isdir(os.path.join(parent_dir, d))]
    print(f"Found {len(obj_dirs)} objects")
    if cli_args.max_objects is not None:
        obj_dirs = sorted(obj_dirs)[:cli_args.max_objects]

    # Process each object with progress bar
    for obj_dir in tqdm(obj_dirs, desc="Processing objects", unit="obj"):
        # exclude the object F_15_C_Jungle_Camo16c6
        if obj_dir == "F_15_C_Jungle_Camo16c6" or obj_dir == "Stylized_Skateboard10c7":
            continue

        result = process_single_object(obj_dir, parent_dir, output_dir, results_dir)
        
        if result is not None:
            all_results.append(result)
            all_ate_icp.append(result['icp_ate'])
            all_rre_icp.append(result['icp_rre'])
            all_ate_final.append(result['final_ate'])
            all_rre_final.append(result['final_rre'])
            
            tqdm.write(f"\nResults for {obj_dir}:")
            tqdm.write(f"ICP  - ATE: {result['icp_ate']:.4f} meters, RRE: {result['icp_rre']:.4f} degrees")
            tqdm.write(f"Final- ATE: {result['final_ate']:.4f} meters, RRE: {result['final_rre']:.4f} degrees")
            tqdm.write(f"Scale: {result['scale']:.4f}")

    # Compute overall statistics
    mean_ate_icp = np.mean(all_ate_icp)
    mean_rre_icp = np.mean(all_rre_icp)
    std_ate_icp = np.std(all_ate_icp)
    std_rre_icp = np.std(all_rre_icp)
    
    mean_ate_final = np.mean(all_ate_final)
    mean_rre_final = np.mean(all_rre_final)
    std_ate_final = np.std(all_ate_final)
    std_rre_final = np.std(all_rre_final)
    
    print("\nOverall Results:")
    print("ICP Registration:")
    print(f"Mean ATE: {mean_ate_icp:.4f} ± {std_ate_icp:.4f} meters")
    print(f"Mean RRE: {mean_rre_icp:.4f} ± {std_rre_icp:.4f} degrees")
    print("\nFinal Registration:")
    print(f"Mean ATE: {mean_ate_final:.4f} ± {std_ate_final:.4f} meters")
    print(f"Mean RRE: {mean_rre_final:.4f} ± {std_rre_final:.4f} degrees")
    
    # Save results
    results_path = os.path.join(results_dir, "registration_results_combined.json")
    with open(results_path, 'w') as f:
        json.dump({
            'individual_results': [
                {k: v.tolist() if isinstance(v, np.ndarray) else v 
                 for k, v in result.items()}
                for result in all_results
            ],
            'summary': {
                'icp': {
                    'mean_ate': mean_ate_icp,
                    'std_ate': std_ate_icp,
                    'mean_rre': mean_rre_icp,
                    'std_rre': std_rre_icp
                },
                'final': {
                    'mean_ate': mean_ate_final,
                    'std_ate': std_ate_final,
                    'mean_rre': mean_rre_final,
                    'std_rre': std_rre_final
                }
            }
        }, f, indent=4)
    
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()