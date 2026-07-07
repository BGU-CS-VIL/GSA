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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self._semantic_feature = torch.empty(0) 
        
        
        # -----------------------------------
                # Global transformation parameters (trainable)
        self.global_translation = nn.Parameter(torch.zeros(3, requires_grad=True,device="cuda"))  # Global translation
        # self.global_rotation = nn.Parameter(torch.eye(3, requires_grad=True,device="cuda"))  # Global rotation (3x3 matrix)
        initial_rotation = torch.zeros(3,requires_grad=True, device="cuda")  # Random values in [0, 0.01]
        
        # Define learnable global rotation with small random perturbation
        self.global_rotation = nn.Parameter(initial_rotation)
        # self.global_scale = nn.Parameter(torch.ones(1, requires_grad=True))  # Global scaling (optional)
        self.global_rotation_matrix = None
        self.global_scale = nn.Parameter(torch.tensor([1.0],requires_grad=True,device="cuda"))


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._semantic_feature, 
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._semantic_feature) = model_args 
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    @property
    def get_semantic_feature(self):
        return self._semantic_feature 
    
    def rewrite_semantic_feature(self, x):
        self._semantic_feature = x

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, semantic_feature_size : int, speedup: bool):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        if speedup: # speed up for Segmentation
            semantic_feature_size = int(semantic_feature_size/4)
        self._semantic_feature = torch.zeros(fused_point_cloud.shape[0], semantic_feature_size, 1).float().cuda() 
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._semantic_feature = nn.Parameter(self._semantic_feature.transpose(1, 2).contiguous().requires_grad_(True))
        

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._semantic_feature], 'lr':training_args.semantic_feature_lr, "name": "semantic_feature"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add semantic features
        for i in range(self._semantic_feature.shape[1]*self._semantic_feature.shape[2]):  
            l.append('semantic_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        semantic_feature = self._semantic_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() 

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
   
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
        semantic_feature = np.stack([np.asarray(plydata.elements[0][f"semantic_{i}"]) for i in range(count)], axis=1) 
        semantic_feature = np.expand_dims(semantic_feature, axis=-1) 

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree



    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "semantic_feature": new_semantic_feature} 

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"] 

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N,1,1) 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantic_feature) 
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask] 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature) 

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        print("semantic_feature shape:", self._semantic_feature.shape)
        print("prune_mask shape:", prune_mask.shape)
        
        
        
        # ---------------------------- roy

        # Add condition for black semantic features using norm
        semantic_features = self._semantic_feature.transpose(1, 2).squeeze(-1)  # Make sure we have (N, D)
        feature_norms = torch.mean(semantic_features, dim=1)  # Take mean across semantic features
        is_black = (feature_norms < 0.1)  # Should now be shape (N,)
        
        print("semantic_features shape:", semantic_features.shape)
        print("feature_norms shape:", feature_norms.shape)
        print("is_black shape:", is_black.shape)
        prune_mask = torch.logical_or(prune_mask, is_black)
        # ----------------------------
        
        # Ensure dimensions match
        assert prune_mask.shape == is_black.shape, f"Shape mismatch: {prune_mask.shape} vs {is_black.shape}"
        # ---------------------------- roy

        

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
# ---------------------------------------------------------------------------------------------
# -------------------------- Gaussian Transformation ------------------------------------------
       
    def transform_shs(self, shs_feat, rotation_matrix):
        # Ensure tensors are on the same device (CUDA)
        shs_feat = shs_feat.detach().clone().to('cuda')
        rotation_matrix = rotation_matrix.to('cuda')

        # Rotate shs
        P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], device='cuda').float()  # Switch axes: yzx -> xyz
        permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P

        # Ensure that each rotation angle is on CUDA
        rot_angles = tuple(angle.to('cuda') for angle in rot_angles)  # Convert each angle to CUDA

        # Construction coefficients
        D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2])
        D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2])
        D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2])

        # Check the device of D_1, D_2, D_3
        print(f"D_1 device: {D_1.device}, D_2 device: {D_2.device}, D_3 device: {D_3.device}")


        # Rotation of the shs features
        for i, D in enumerate([D_1, D_2, D_3], start=1):
            shs_indices = slice((i - 1) * 5, i * 5) if i > 1 else slice(0, 3)  # 0-2 for one degree
            one_degree_shs = shs_feat[:, shs_indices]
            one_degree_shs = one_degree_shs.permute(0, 2, 1)  # Rearranging dimensions: n shs_num rgb -> n rgb shs_num
            one_degree_shs = torch.einsum('...ij,...j->...i', D, one_degree_shs)
            one_degree_shs = one_degree_shs.permute(0, 2, 1)  # Rearranging back: n rgb shs_num -> n shs_num rgb
            shs_feat[:, shs_indices] = one_degree_shs

        return shs_feat

    def quat_multiply(self, quaternion0, quaternion1):
        """Multiply two quaternions."""
        w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
        w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
        return torch.cat((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ), dim=-1)

    def rotation_x(self, angle_rad):
        cos = torch.cos(angle_rad)
        sin = torch.sin(angle_rad)
        return torch.stack([
            torch.stack([torch.ones_like(cos), torch.zeros_like(cos), torch.zeros_like(cos)]),
            torch.stack([torch.zeros_like(cos), cos, -sin]),
            torch.stack([torch.zeros_like(cos), sin, cos])
        ])

    def rotation_y(self, angle_rad):
        cos = torch.cos(angle_rad)
        sin = torch.sin(angle_rad)
        return torch.stack([
            torch.stack([cos, torch.zeros_like(cos), sin]),
            torch.stack([torch.zeros_like(cos), torch.ones_like(cos), torch.zeros_like(cos)]),
            torch.stack([-sin, torch.zeros_like(cos), cos])
        ])

    def rotation_z(self, angle_rad):
        cos = torch.cos(angle_rad)
        sin = torch.sin(angle_rad)
        return torch.stack([
            torch.stack([cos, -sin, torch.zeros_like(cos)]),
            torch.stack([sin, cos, torch.zeros_like(cos)]),
            torch.stack([torch.zeros_like(cos), torch.zeros_like(cos), torch.ones_like(cos)])
        ])

    # def combine_rotations(self, angles):
    #     rot_x = self.rotation_x(angles[0])
    #     rot_y = self.rotation_y(angles[1])
    #     rot_z = self.rotation_z(angles[2])
    #     return rot_z @ rot_y @ rot_x

    def combine_rotations(self, angles):
        # Clamp angles to prevent numerical instability
        angles = torch.clamp(angles, -2*np.pi, 2*np.pi)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-6
        
        rot_x = self.rotation_x(angles[0] + eps)
        rot_y = self.rotation_y(angles[1] + eps)
        rot_z = self.rotation_z(angles[2] + eps)
        
        # Use stable matrix multiplication
        combined = torch.matmul(rot_z, torch.matmul(rot_y, rot_x))
        
        # Ensure the result is a valid rotation matrix
        combined = torch.where(torch.isnan(combined), torch.eye(3, device=combined.device), combined)
        return combined

    def rotmat2qvec_torch(self, R):
        """
        Converts a rotation matrix to a quaternion using PyTorch.
        Args:
            R (torch.Tensor): Rotation matrix of shape (3, 3) or (B, 3, 3) if batched.
        Returns:
            qvec (torch.Tensor): Quaternion of shape (4,) if single matrix, (B, 4) if batched.
        """
        if len(R.shape) == 2:  # Single 3x3 matrix
            R = R.unsqueeze(0)  # Add batch dimension (B, 3, 3)
        
        # Extract the elements of the rotation matrix
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R[:, 0, 0], R[:, 1, 0], R[:, 2, 0], \
                                                        R[:, 0, 1], R[:, 1, 1], R[:, 2, 1], \
                                                        R[:, 0, 2], R[:, 1, 2], R[:, 2, 2]
        
        # Create a new tensor for K to avoid in-place modification
        K = torch.zeros((R.size(0), 4, 4), device=R.device, dtype=R.dtype, requires_grad=True)

        # Compute the elements of K without in-place operations
        K_0_0 = Rxx - Ryy - Rzz
        K_1_0 = Ryx + Rxy
        K_2_0 = Rzx + Rxz
        K_3_0 = Ryz - Rzy

        K_1_1 = Ryy - Rxx - Rzz
        K_2_1 = Rzy + Ryz
        K_3_1 = Rzx - Rxz

        K_2_2 = Rzz - Rxx - Ryy
        K_3_2 = Rxy - Ryx

        K_3_3 = Rxx + Ryy + Rzz

        # Now, update K using out-of-place operations
        K = K.clone()  # Create a fresh tensor to hold the updated values

        K[:, 0, 0] = K_0_0
        K[:, 1, 0] = K_1_0
        K[:, 0, 1] = K_1_0
        K[:, 2, 0] = K_2_0
        K[:, 0, 2] = K_2_0
        K[:, 3, 0] = K_3_0
        K[:, 0, 3] = K_3_0

        K[:, 1, 1] = K_1_1
        K[:, 2, 1] = K_2_1
        K[:, 1, 2] = K_2_1
        K[:, 3, 1] = K_3_1
        K[:, 1, 3] = K_3_1

        K[:, 2, 2] = K_2_2
        K[:, 3, 2] = K_3_2
        K[:, 2, 3] = K_3_2

        K[:, 3, 3] = K_3_3

        # Compute eigenvalues and eigenvectors of K
        eigvals, eigvecs = torch.linalg.eigh(K)  # (B, 4) and (B, 4, 4)
        
        # Get the eigenvector corresponding to the largest eigenvalue
        max_eigval_indices = torch.argmax(eigvals, dim=1)  # (B,)
        qvec = torch.stack([eigvecs[b, :, max_eigval_indices[b]] for b in range(eigvecs.size(0))], dim=0)  # (B, 4)
        
        # Reorder qvec to match the NumPy version (w, x, y, z)
        qvec = qvec[:, [3, 0, 1, 2]]

        # Ensure positive quaternion scalar (w)
        qvec = torch.where(qvec[:, 0:1] < 0, -qvec, qvec)  # Make sure w > 0

        # If the batch size is 1, return a single quaternion, otherwise return batch
        return qvec.squeeze(0) if qvec.size(0) == 1 else qvec

    def transform_shs(self, shs_feat, rotation_matrix):
        """
        Transform spherical harmonics features.
        Args:
            shs_feat: tensor of shape [N, C, F] where:
                N is number of points
                C is number of SH coefficients
                F is number of features
        """
        shs_feat = shs_feat.to("cuda")
        P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float().to('cuda')
        permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix.to('cuda') @ P
        
        rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
        rot_angles = [r.cpu() for r in rot_angles]
        
        result = torch.zeros_like(shs_feat)
        
        # Calculate the number of SH degrees based on number of coefficients
        # For degree l, number of coefficients is (2l + 1)
        # Total coefficients for up to degree L is sum(2l + 1) for l from 1 to L
        num_coeffs = shs_feat.shape[1]
        
        start_idx = 0
        current_degree = 1
        
        while start_idx < num_coeffs:
            # Number of coefficients for current degree
            degree_coeffs = 2 * current_degree + 1
            
            if start_idx + degree_coeffs > num_coeffs:
                break
                
            # Get Wigner-D matrix for current degree
            D_l = o3.wigner_D(current_degree, rot_angles[0], -rot_angles[1], rot_angles[2]).to('cuda')
            
            # Get features for current degree
            end_idx = start_idx + degree_coeffs
            l_features = shs_feat[:, start_idx:end_idx, :]
            
            # Apply rotation
            result[:, start_idx:end_idx, :] = torch.bmm(
                D_l.unsqueeze(0).expand(shs_feat.shape[0], -1, -1), 
                l_features
            )
            
            # Update indices for next degree
            start_idx = end_idx
            current_degree += 1
        
        return result

    # def apply_global_transformation(self):
    #     angles = self.global_rotation
        
    #     # Build rotation matrices (same as before)
    #     cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])
    #     cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
    #     cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])
        
    #     R_x = torch.stack([
    #         torch.tensor([1.0, 0.0, 0.0], device='cuda'),
    #         torch.stack([torch.zeros_like(cos_x), cos_x, -sin_x]),
    #         torch.stack([torch.zeros_like(cos_x), sin_x, cos_x])
    #     ])
        
    #     R_y = torch.stack([
    #         torch.stack([cos_y, torch.zeros_like(cos_y), sin_y]),
    #         torch.tensor([0.0, 1.0, 0.0], device='cuda'),
    #         torch.stack([-sin_y, torch.zeros_like(cos_y), cos_y])
    #     ])
        
    #     R_z = torch.stack([
    #         torch.stack([cos_z, -sin_z, torch.zeros_like(cos_z)]),
    #         torch.stack([sin_z, cos_z, torch.zeros_like(cos_z)]),
    #         torch.tensor([0.0, 0.0, 1.0], device='cuda')
    #     ])
        
    #     R = R_z @ R_y @ R_x
        
    #     # Transform positions
    #     center = self._xyz.mean(dim=0, keepdim=True)
    #     centered = self._xyz - center
    #     rotated = (R @ centered.T).T
    #     transformed_xyz = self.global_scale*(rotated + center) + self.global_translation

    #     # Transform rotations (quaternions)
    #     w = torch.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
    #     w4 = 4.0 * w
    #     x = (R[2,1] - R[1,2]) / w4
    #     y = (R[0,2] - R[2,0]) / w4
    #     z = (R[1,0] - R[0,1]) / w4
    #     quaternion = torch.stack([w, x, y, z])
        
    #     rotated_quaternions = self.quat_multiply(self._rotation, quaternion)
        
    #     # Transform SH features
    #     rotation_cpu = R.clone().detach()
        
       
    #     # rotated_features_dc = None
    #     # rotated_features_rest = None
    #     # Update scaling
    #     new_scaling = self._scaling + torch.log(self.global_scale)

    #     # the rotation isn't good, something with the colors
    #     # with torch.no_grad():
    #     #     features = torch.cat([self._features_dc.transpose(1,2), self._features_rest.transpose(1,2)], dim=2)

    #     #      # probabliy not working good
    #     #     rotated_features = self.transform_shs(features, rotation_cpu)
            
    #     #     # # Split back into DC and rest components
    #     #     rotated_features_dc = rotated_features[:, :, 0:1].transpose(1,2)
    #     #     rotated_features_rest = rotated_features[:, :, 1:].transpose(1,2)
    #     #     self._features_dc = rotated_features_dc.clone().to("cuda")
    #     #     self._features_rest = rotated_features_rest.clone().to("cuda")
          
    #     rotated_features_dc = None
    #     rotated_features_rest = None

    #     return transformed_xyz, rotated_quaternions, new_scaling, rotated_features_dc, rotated_features_rest        


    def rotmat2qvec_torch(self, R):
        """
        More numerically stable version of rotation matrix to quaternion conversion.
        """
        if len(R.shape) == 2:
            R = R.unsqueeze(0)
        
        # Add small epsilon to prevent numerical instability
        eps = 1e-7
        
        # Get trace
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        
        # Initialize quaternion
        qw = torch.zeros_like(tr)
        qx = torch.zeros_like(tr)
        qy = torch.zeros_like(tr)
        qz = torch.zeros_like(tr)
        
        # Case 1: trace > 0
        case_1 = tr > 0
        if case_1.any():
            S = torch.sqrt(tr[case_1] + 1.0) * 2
            qw[case_1] = 0.25 * S
            qx[case_1] = (R[case_1, 2, 1] - R[case_1, 1, 2]) / S
            qy[case_1] = (R[case_1, 0, 2] - R[case_1, 2, 0]) / S
            qz[case_1] = (R[case_1, 1, 0] - R[case_1, 0, 1]) / S
        
        # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
        case_2 = (~case_1) & (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2])
        if case_2.any():
            S = torch.sqrt(1.0 + R[case_2, 0, 0] - R[case_2, 1, 1] - R[case_2, 2, 2]) * 2
            qw[case_2] = (R[case_2, 2, 1] - R[case_2, 1, 2]) / S
            qx[case_2] = 0.25 * S
            qy[case_2] = (R[case_2, 0, 1] + R[case_2, 1, 0]) / S
            qz[case_2] = (R[case_2, 0, 2] + R[case_2, 2, 0]) / S
        
        # Case 3: R[1,1] > R[2,2]
        case_3 = (~case_1) & (~case_2) & (R[:, 1, 1] >= R[:, 2, 2])
        if case_3.any():
            S = torch.sqrt(1.0 + R[case_3, 1, 1] - R[case_3, 0, 0] - R[case_3, 2, 2]) * 2
            qw[case_3] = (R[case_3, 0, 2] - R[case_3, 2, 0]) / S
            qx[case_3] = (R[case_3, 0, 1] + R[case_3, 1, 0]) / S
            qy[case_3] = 0.25 * S
            qz[case_3] = (R[case_3, 1, 2] + R[case_3, 2, 1]) / S
        
        # Case 4: remaining cases
        case_4 = (~case_1) & (~case_2) & (~case_3)
        if case_4.any():
            S = torch.sqrt(1.0 + R[case_4, 2, 2] - R[case_4, 0, 0] - R[case_4, 1, 1]) * 2
            qw[case_4] = (R[case_4, 1, 0] - R[case_4, 0, 1]) / S
            qx[case_4] = (R[case_4, 0, 2] + R[case_4, 2, 0]) / S
            qy[case_4] = (R[case_4, 1, 2] + R[case_4, 2, 1]) / S
            qz[case_4] = 0.25 * S
        
        # Normalize quaternions
        norm = torch.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw = qw / (norm + eps)
        qx = qx / (norm + eps)
        qy = qy / (norm + eps)
        qz = qz / (norm + eps)
        
        # Stack quaternions
        q = torch.stack([qw, qx, qy, qz], dim=-1)
        
        # Ensure w is positive
        q = torch.where(q[..., 0:1] < 0, -q, q)
        
        return q.squeeze(0) if q.size(0) == 1 else q

    def apply_global_transformation(self):
        """Apply global transformation while maintaining gradient flow"""
        
        # Get angles from the parameter
        angles = self.global_rotation  # [rx, ry, rz]
        
        # Build rotation matrices with gradient flow
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
        
        # Combine rotations maintaining gradients
        R = torch.matmul(torch.matmul(R_z, R_y), R_x)
        
        # Transform positions with gradient flow
        # transformed_xyz = torch.matmul(self._xyz, R.T) + self.global_translation.unsqueeze(0)
        # transformed_xyz = self.global_scale * transformed_xyz
        
        # Transform positions with gradient flow
        transformed_xyz = torch.matmul(self._xyz, R.T)  # Apply rotation
        transformed_xyz = self.global_scale * transformed_xyz  # Apply scale
        transformed_xyz = transformed_xyz + self.global_translation.unsqueeze(0)  # Apply translation
        # For quaternions, derive from R
        quaternion = self.rotmat2qvec_torch(R)
        rotated_quaternions = self.quat_multiply(self._rotation, quaternion)
        
        # Update scaling
        new_scaling = self._scaling + torch.log(self.global_scale)
        
        # Keep features as is for now
        rotated_features_dc = self._features_dc
        rotated_features_rest = self._features_rest
        
        return transformed_xyz, rotated_quaternions, new_scaling, rotated_features_dc, rotated_features_rest