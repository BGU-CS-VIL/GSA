#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

from torch import nn


# def calculate_selection_score(features, query_features, score_threshold=None, positive_ids=[0],model="lsseg"):
#     if model=="lsseg":
#         features /= features.norm(dim=-1, keepdim=True)
#         query_features /= query_features.norm(dim=-1, keepdim=True)
#         scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
#         if scores.shape[-1] == 1:
#             scores = scores[:, 0]  # (N_points,)
#             scores = (scores >= score_threshold).float()
#         else:
#             scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
#             if score_threshold is not None:
#                 scores = scores[:, positive_ids].sum(-1)  # (N_points, )
#                 scores = (scores >= score_threshold).float()
#             else:
#                 scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)  # (N_points, )
#                 scores = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda()).float()
#         return scores
#     elif model=="dino":
#         features /= features.norm(dim=-1, keepdim=True)
#         query_features /= query_features.norm(dim=-1, keepdim=True)
#         scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
#         if scores.shape[-1] == 1:
#             scores = scores[:, 0]  # (N_points,)
#             scores = (scores >= score_threshold).float()
#         else:
#             scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
#             if score_threshold is not None:
#                 scores = scores.sum(-1)  # (N_points, )
#                 scores = (scores >= score_threshold).float()
#             else:
#                 scores = scores.sum(-1)  # (N_points, )
#                 scores = torch.argmax(scores, dim=-1).cuda().float()
#         return scores


# def calculate_selection_score_delete(features, query_features, score_threshold=None, positive_ids=[0],model="lsseg"):
#         if model=="lsseg":
#             features /= features.norm(dim=-1, keepdim=True)
#             query_features /= query_features.norm(dim=-1, keepdim=True)
#             scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
#             if scores.shape[-1] == 1:
#                 scores = scores[:, 0]  # (N_points,)
#                 mask = (scores >= score_threshold).float()
#             else:
#                 scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
#                 scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)  # (N_points, )
#                 mask = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda())
                
#                 if score_threshold is not None:
#                     scores = scores[:, positive_ids].sum(-1)  # (N_points, )
#                     mask = torch.bitwise_or((scores >= score_threshold), mask).float()
            
#             return mask
#         elif model=="dino":
#             features /= features.norm(dim=-1, keepdim=True)
#             query_features /= query_features.norm(dim=-1, keepdim=True)
#             scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
#             if scores.shape[-1] == 1:
#                 scores = scores[:, 0]  # (N_points,)
#                 mask = (scores >= score_threshold).float()
#             else:
#                 scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
                
#                 scores = scores.sum(-1)  # (N_points, )
#                 mask = torch.argmax(scores, dim=-1).cuda()
                
#                 if score_threshold is not None:
#                     scores = scores.sum(-1)  # (N_points, )
#                     mask = torch.bitwise_or((scores >= score_threshold), mask).float()
            
#             return mask

def load_reference_features(reference_path):
    """Load reference DINO features from both classes."""
    import glob
    import torch
    
    reference_features = []
    path = "/home/amoyalr/Research/Thesis/3gs_tools/feature-3dgs/"
    reference_files = glob.glob(path + f"reference_dino_{reference_path}*.pt")
    
    for file_path in reference_files:
        features = torch.load(file_path)
        reference_features.append(features)
    
    # Concatenate all reference features
    if reference_features:
        return torch.cat(reference_features, dim=0)
    return None
# working partial good
# def compute_similarities_in_batches_with_pca(features, reference_features, batch_size=1000, n_components=3):
#     """Compute similarities between features and reference_features in batches after PCA reduction."""
#     import torch
#     from sklearn.decomposition import PCA
#     import numpy as np
    
#     n_points = features.shape[0]
#     max_similarities = torch.zeros(n_points, device=features.device)
    
#     # Move all data to CPU for PCA
#     features_cpu = features.cpu().numpy()
#     reference_features_cpu = reference_features.cpu().numpy()
    
#     # Combine features for PCA fitting
#     combined_features = np.vstack([features_cpu, reference_features_cpu])
    
#     # Fit PCA
#     pca = PCA(n_components=n_components)
#     pca.fit(combined_features)
    
#     # Transform reference features
#     reference_features_pca = torch.from_numpy(
#         pca.transform(reference_features_cpu)
#     ).to(features.device)
    
#     # Normalize reduced reference features
#     reference_features_pca /= reference_features_pca.norm(dim=-1, keepdim=True)
 
#     from tqdm import tqdm
    
#     for i in tqdm(range(0, n_points, batch_size), desc="Computing similarities", total=(n_points + batch_size - 1) // batch_size):
#         end_idx = min(i + batch_size, n_points)
#         batch_features = features_cpu[i:end_idx]
        
#         # Transform batch features
#         batch_features_pca = torch.from_numpy(
#             pca.transform(batch_features)
#         ).to(features.device)
        
#         # Normalize batch features
#         batch_features_pca /= batch_features_pca.norm(dim=-1, keepdim=True)
        
#         # Calculate similarity for this batch
#         similarities = batch_features_pca.half() @ reference_features_pca.T.half()
        
#         # Get maximum similarity for each point in batch
#         batch_max_similarities, _ = similarities.max(dim=1)
#         # print(batch_max_similarities)
#         # Store results
#         max_similarities[i:end_idx] = batch_max_similarities
        
#         # Clear GPU memory
#         del similarities, batch_features_pca
#         torch.cuda.empty_cache()
    
#     return max_similarities

# def calculate_selection_score(features, query_features, score_threshold=None, positive_ids=[0], model="lsseg", dino_reference=""):
#     if model == "lsseg":
#         # Original LSSEG code remains unchanged
#         features /= features.norm(dim=-1, keepdim=True)
#         query_features /= query_features.norm(d3gs_tools/feature-3dgs/output/OUTPUT_NAME_dino_base/train/ours_3000_extraction_car/renders/00000.pngim=-1, keepdim=True)
#         scores = features.half() @ query_features.T.half()
#         if scores.shape[-1] == 1:
#             scores = scores[:, 0]
#             scores = (scores >= score_threshold).float()
#         else:
#             scores = torch.nn.functional.softmax(scores, dim=-1)
#             if score_threshold is not None:
#                 scores = scores[:, positive_ids].sum(-1)
#                 scores = (scores >= score_threshold).float()
#             else:
#                 scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)
#                 scores = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda()).float()
#         return scores
    
#     elif model == "dino":
#         if dino_reference:
#             # Load and normalize reference features
#             reference_features = load_reference_features(dino_reference).cuda()
#             if reference_features is None:
#                 raise ValueError(f"No reference features found for path: {dino_reference}")
            
#             # Compute similarities in batches with PCA
#             max_similarities = compute_similarities_in_batches_with_pca(
#                 features, 
#                 reference_features,
#                 n_components=3
#             )
            
#             # Apply threshold
#             scores = (max_similarities >= score_threshold).float()
            
#             # Clear GPU memory
#             del reference_features
#             torch.cuda.empty_cache()
            
#             return scores

# def calculate_selection_score_delete(features, query_features, score_threshold=None, positive_ids=[0], model="lsseg", dino_reference=""):
#     if model == "lsseg":
#         # Original LSSEG code remains unchanged
#         features /= features.norm(dim=-1, keepdim=True)
#         query_features /= query_features.norm(dim=-1, keepdim=True)
#         scores = features.half() @ query_features.T.half()
#         if scores.shape[-1] == 1:
#             scores = scores[:, 0]
#             mask = (scores >= score_threshold).float()
#         else:
#             scores = torch.nn.functional.softmax(scores, dim=-1)
#             scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)
#             mask = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda())
            
#             if score_threshold is not None:
#                 scores = scores[:, positive_ids].sum(-1)
#                 mask = torch.bitwise_or((scores >= score_threshold), mask).float()
        
#         return mask
    
#     elif model == "dino":
#         if dino_reference:
#             # Load and normalize reference features
#             reference_features = load_reference_features(dino_reference).cuda()
#             if reference_features is None:
#                 raise ValueError(f"No reference features found for path: {dino_reference}")
            
#             # Compute similarities in batches with PCA
#             max_similarities = compute_similarities_in_batches_with_pca(
#                 features, 
#                 reference_features,
#                 n_components=3
#             )
            
#             # Apply threshold
#             mask = (max_similarities >= score_threshold).float()
            
#             # Clear GPU memory
#             del reference_features
#             torch.cuda.empty_cache()
            
#             return mask

# good
# def compute_similarities_in_batches(features, reference_features, batch_size=1000):
#     """Compute similarities between features and reference_features in batches."""
#     n_points = features.shape[0]
#     max_similarities = torch.zeros(n_points, device=features.device)

#     from tqdm import tqdm
    
#     for i in tqdm(range(0, n_points, batch_size), desc="Computing similarities", total=(n_points + batch_size - 1) // batch_size):
#         end_idx = min(i + batch_size, n_points)
#         batch_features = features[i:end_idx]
        
#         # Calculate similarity for this batch
#         similarities = batch_features.half() @ reference_features.T.half()  # (batch_size, N_reference)
        
#         # Get maximum similarity for each point in batch
#         batch_max_similarities, _ = similarities.max(dim=1)  # (batch_size,)
        

#         # Store results
#         max_similarities[i:end_idx] = batch_max_similarities

#         # Clear GPU memory
#         # del similarities
#         torch.cuda.empty_cache()
    
#     return max_similarities

def compute_similarities_in_batches(features, reference_features, batch_size=1000):
    """Compute similarities between features and reference_features in batches."""
    n_points = features.shape[0]
    similarities_mask = torch.zeros(n_points, device=features.device)
    
    from tqdm import tqdm
    
    for i in tqdm(range(0, n_points, batch_size), desc="Computing similarities"):
        end_idx = min(i + batch_size, n_points)
        batch_features = features[i:end_idx]
        
        # Calculate similarity for each point in batch individually
        for j in range(batch_features.shape[0]):
            point_feature = batch_features[j:j+1]  # Keep dimension for matmul
            point_similarities = point_feature.half() @ reference_features.T.half()
            # Get similarity score for this point
            point_max_similarity = point_similarities.max()
            similarities_mask[i+j] = point_max_similarity
        
        torch.cuda.empty_cache()
    
    return similarities_mask
def calculate_selection_score(features, query_features, score_threshold=None, positive_ids=[0], model="lsseg", dino_reference=""):
    if model == "lsseg":
        # Original LSSEG code remains unchanged
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()
        if scores.shape[-1] == 1:
            scores = scores[:, 0]
            scores = (scores >= score_threshold).float()
        else:
            scores = torch.nn.functional.softmax(scores, dim=-1)
            if score_threshold is not None:
                scores = scores[:, positive_ids].sum(-1)
                scores = (scores >= score_threshold).float()
            else:
                scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)
                scores = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda()).float()
        return scores
    elif model == "dino" and dino_reference:
        reference_features = load_reference_features(dino_reference).cuda()
        if reference_features is None:
            raise ValueError(f"No reference features found for path: {dino_reference}")
        
        # Normalize features
        features /= features.norm(dim=-1, keepdim=True)
        reference_features /= reference_features.norm(dim=-1, keepdim=True)
        
        # Compute individual similarities
        similarities_mask = compute_similarities_in_batches(features, reference_features)
        
        # Apply threshold
        scores = (similarities_mask >= score_threshold).float()
        
        del reference_features
        torch.cuda.empty_cache()
        
        return scores
    elif model == "dino_pca":
        scores = (features.abs().sum(dim=-1) > 0).float()
        return scores


# def calculate_selection_score(features, query_features, score_threshold=None, positive_ids=[0], model="lsseg", dino_reference=""):
#     if model == "lsseg":
#         # Original LSSEG code remains unchanged
#         features /= features.norm(dim=-1, keepdim=True)
#         query_features /= query_features.norm(dim=-1, keepdim=True)
#         scores = features.half() @ query_features.T.half()
#         if scores.shape[-1] == 1:
#             scores = scores[:, 0]
#             scores = (scores >= score_threshold).float()
#         else:
#             scores = torch.nn.functional.softmax(scores, dim=-1)
#             if score_threshold is not None:
#                 scores = scores[:, positive_ids].sum(-1)
#                 scores = (scores >= score_threshold).float()
#             else:
#                 scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)
#                 scores = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda()).float()
#         return scores
    
#     elif model == "dino":
#         if dino_reference:
#             # Load and normalize reference features
#             reference_features = load_reference_features(dino_reference).cuda()
#             if reference_features is None:
#                 raise ValueError(f"No reference features found for path: {dino_reference}")
            
#             # Normalize features
#             features /= features.norm(dim=-1, keepdim=True)
#             reference_features /= reference_features.norm(dim=-1, keepdim=True)
            
     
#             # Compute similarities in batches
#             max_similarities = compute_similarities_in_batches(features, reference_features)
            
#             # Apply threshold
#             scores = (max_similarities >= score_threshold).float()
            
#             # Clear GPU memory
#             del reference_features
#             torch.cuda.empty_cache()
            
#             return scores

def calculate_selection_score_delete(features, query_features, score_threshold=None, positive_ids=[0], model="lsseg", dino_reference=""):
    if model == "lsseg":
        # Original LSSEG code remains unchanged
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()
        if scores.shape[-1] == 1:
            scores = scores[:, 0]
            mask = (scores >= score_threshold).float()
        else:
            scores = torch.nn.functional.softmax(scores, dim=-1)
            scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)
            mask = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda())
            
            if score_threshold is not None:
                scores = scores[:, positive_ids].sum(-1)
                mask = torch.bitwise_or((scores >= score_threshold), mask).float()
        
        return mask
    
    elif model == "dino":
        if dino_reference:
            # Load and normalize reference features
            reference_features = load_reference_features(dino_reference).cuda()
            if reference_features is None:
                raise ValueError(f"No reference features found for path: {dino_reference}")
            
            # Normalize features
            features /= features.norm(dim=-1, keepdim=True)
            reference_features /= reference_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities in batches
            max_similarities = compute_similarities_in_batches(features, reference_features)
            
            # Apply threshold
            mask = (max_similarities >= score_threshold).float()
            
            # Clear GPU memory
            del reference_features
            torch.cuda.empty_cache()
            
            return mask

def save_selected_gaussians(pc, scores, save_path,model="lsseg"):
    """
    Save only the selected Gaussians to a PLY file based on semantic scores.
    
    Args:
        pc (GaussianModel): The Gaussian model containing all points
        scores (torch.Tensor): Binary selection mask (1 for selected, 0 for not selected)
        save_path (str): Path where to save the PLY file
    """
    # Create a new GaussianModel for selected points
    selected_model = GaussianModel(pc.max_sh_degree)
    
    # Get mask for selected points (scores > 0.5)
    selected_mask = scores > 0.5
    
    # Copy over only the selected points' data
    selected_model._xyz = nn.Parameter(pc._xyz[selected_mask].clone().detach())
    selected_model._features_dc = nn.Parameter(pc._features_dc[selected_mask].clone().detach())
    selected_model._features_rest = nn.Parameter(pc._features_rest[selected_mask].clone().detach())
    selected_model._opacity = nn.Parameter(pc._opacity[selected_mask].clone().detach())
    selected_model._scaling = nn.Parameter(pc._scaling[selected_mask].clone().detach())
    selected_model._rotation = nn.Parameter(pc._rotation[selected_mask].clone().detach())
    selected_model._semantic_feature = nn.Parameter(pc._semantic_feature[selected_mask].clone().detach())
    
    # Set other necessary attributes
    selected_model.active_sh_degree = pc.active_sh_degree
    selected_model.max_sh_degree = pc.max_sh_degree
    
    # Save the selected points to PLY
    selected_model.save_ply(save_path)
    
    return selected_model

def render_edit(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, queries_features : torch.Tensor, edit_dict : dict,
                scaling_modifier = 1.0, override_color = None,model_type="lsseg",dino_reference=""): 
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity



    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    semantic_feature = pc.get_semantic_feature


    positive_ids = edit_dict["positive_ids"]
    score_threshold = edit_dict["score_threshold"]
    op_dict = edit_dict["operations"]
    print("INITTTT MODEL TYPE",model_type)
    # edtiing
    if "deletion" in op_dict:
        scores = calculate_selection_score_delete(semantic_feature[:, 0, :], queries_features, 
                                       score_threshold=score_threshold, positive_ids=positive_ids,model=model_type,dino_reference=dino_reference) # # torch.Size([331617])
        opacity.masked_fill_(scores[:, None] >= 0.5, 0)
        # print(scores) # tensor(1., device='cuda:0') tensor(0., device='cuda:0')
    if "extraction" in op_dict:
        scores = calculate_selection_score(semantic_feature[:, 0, :], queries_features, 
                                       score_threshold=score_threshold, positive_ids=positive_ids,model=model_type,dino_reference=dino_reference)
        opacity.masked_fill_(scores[:, None] <= 0.5, 0)
    if "color_func" in op_dict:
        scores = calculate_selection_score(semantic_feature[:, 0, :], queries_features, 
                                       score_threshold=score_threshold, positive_ids=positive_ids,model=model_type,dino_reference=dino_reference)
        shs[:, 0, :] = shs[:, 0, :] * (1 - scores[:, None]) + op_dict["color_func"](shs[:, 0, :]) * scores[:, None]
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, feature_map, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        semantic_feature = semantic_feature, 
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    save_selected_gaussians(pc, scores, "./roy_ply.ply")

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'feature_map': feature_map,
            "depth": depth}


def render_registration(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    Render the scene with gradient flow for registration optimization.
    Background tensor (bg_color) must be on GPU!
    """
    
     
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass


    # Apply global transformation first
    means3D, rotations, new_scales, feat_, feat2_ = pc.apply_global_transformation()
    
    # Project 3D points to screen space
    viewmatrix = viewpoint_camera.world_view_transform
    projmatrix = viewpoint_camera.full_proj_transform
    full_proj = projmatrix @ viewmatrix
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Handle scales and rotations
    scales = pc.scaling_activation(new_scales)
    rotations = pc.rotation_activation(rotations)
    means2D = screenspace_points

    # Handle colors/features
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    semantic_feature = pc.get_semantic_feature
       # require grad for semantic feature
    # semantic_feature.requires_grad = True
    # retain grad for semantic feature
    # semantic_feature.retain_grad()
    # Rasterize
  
    rendered_image, feature_map, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        semantic_feature=semantic_feature,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)
    if feature_map is not None:
        feature_map.retain_grad()
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "feature_map": feature_map,
        "depth": depth
    }


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity



    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    semantic_feature = pc.get_semantic_feature
 
    var_loss = torch.zeros(1,viewpoint_camera.image_height,viewpoint_camera.image_width) ###d

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, feature_map, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        semantic_feature = semantic_feature, 
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'feature_map': feature_map,
            "depth": depth} ###d

