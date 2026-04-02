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
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.regulation import compute_plane_smoothness
from typing import Tuple

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

    def __init__(self, sh_degree : int, args, adaptive_motion_hierarchy):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        
        self.args = args
        self.adaptive_motion_hierarchy = adaptive_motion_hierarchy
        self._deformation_table = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._coefs = torch.empty(0)
        self._opacity = torch.empty(0)
        self._deform_mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation_table,
            self._deform_mask,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.percent_dense,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
            self._xyz, 
            self._deformation_table,
            self._deform_mask,
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
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
    def get_coef(self):
        return self._coefs, self.args.poly_order_num, self.args.fs_order_num
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1


        N = fused_point_cloud.shape[0]
        if self.adaptive_motion_hierarchy == True:
            weight_coefs = torch.zeros((N, self.args.ch_num, self.args.curve_num))
            position_coefs = torch.zeros((N, self.args.ch_num, self.args.curve_num)) + torch.linspace(0,1,self.args.curve_num)
            shape_coefs = torch.zeros((N, self.args.ch_num, self.args.curve_num)) + self.args.init_param
            _coefs = torch.stack((weight_coefs, position_coefs, shape_coefs), dim=2).reshape(N,-1).float().to("cuda")
            self._coefs = nn.Parameter(_coefs.requires_grad_(True))
        
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deform_mask = torch.gt(torch.zeros((self.get_xyz.shape[0]),device="cuda"),0)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._coefs], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "coefs"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps) 


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if param_group["name"] == "coefs":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

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
        for i in range(self._coefs.shape[1]):
            l.append('coefs_{}'.format(i))
        return l


    def load_model(self, path):
        print("loading model from exists{}".format(path))
        
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deform_mask = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
            
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


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
            
        coef_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("coefs_")]
        coef_names = sorted(coef_names, key = lambda x: int(x.split('_')[-1]))
        coefs = np.zeros((xyz.shape[0], len(coef_names)))
        for idx, attr_name in enumerate(coef_names):
            coefs[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._coefs = nn.Parameter(torch.tensor(coefs, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path)) 

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        coefs = self._coefs.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, coefs), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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
            if len(group["params"]) > 1:
                continue
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
        self._coefs = optimizable_tensors["coefs"]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self._deform_mask = self._deform_mask[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1 or group["name"]=='deformation':
                continue
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_coefs, new_deformation_table, new_deform_mask):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "coefs": new_coefs
       }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._coefs = optimizable_tensors["coefs"]
        self._deform_mask = torch.cat([self._deform_mask,new_deform_mask],-1)
        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def update_deform_mask_after_densification(self, deformation_threshold=0.05):
        """
        密集化后重新评估新点的变形掩码
        检查新点的变形累积量，根据变形量决定是否标记为变形点
        
        Args:
            deformation_threshold: 变形量阈值，超过此值则标记为变形点
        """
        if not hasattr(self, '_deformation_accum') or self._deformation_accum.shape[0] == 0:
            return
        
        # 检查所有点的变形累积
        deform_accum = self._deformation_accum.norm(dim=-1)
        
        # 如果变形量超过阈值，标记为变形点
        should_deform = deform_accum > deformation_threshold
        
        # 更新变形掩码（只更新新点，保留原有掩码）
        if len(should_deform) <= len(self._deform_mask):
            # 对于新点，如果变形量足够大，则标记为变形点
            # 这里采用"或"操作，保留原有掩码，同时添加新的变形点
            self._deform_mask[:len(should_deform)] = self._deform_mask[:len(should_deform)] | should_deform
            self._deformation_table[:len(should_deform)] = self._deformation_table[:len(should_deform)] | should_deform

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, region_mask=None):
        """
        密集化：对梯度较大的点做 split。支持 region_mask 限制作用区域。
        """
        n_init_points = self.get_xyz.shape[0]
        # 对齐 grads 长度到当前高斯数量
        padded_grad = torch.zeros((n_init_points), device="cuda")
        use_len = min(grads.shape[0], n_init_points)
        padded_grad[:use_len] = grads[:use_len].squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 仅在指定区域内进行 densify
        if region_mask is not None:
            if region_mask.shape[0] != selected_pts_mask.shape[0]:
                # 对齐长度
                padded_region = torch.zeros_like(selected_pts_mask)
                copy_len = min(region_mask.shape[0], selected_pts_mask.shape[0])
                padded_region[:copy_len] = region_mask[:copy_len]
                region_mask = padded_region
            selected_pts_mask = torch.logical_and(selected_pts_mask, region_mask)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_coefs = self._coefs[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        new_deform_mask = self._deform_mask[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_coefs,new_deformation_table, new_deform_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, region_mask=None):
        """
        密集化：对梯度较大的点做 clone。支持 region_mask 限制作用区域。
        """
        n_init_points = self.get_xyz.shape[0]
        # 对齐 grads 长度到当前高斯数量
        if grads.shape[0] != n_init_points:
            grads_aligned = torch.zeros((n_init_points, grads.shape[1]), device=grads.device, dtype=grads.dtype)
            use_len = min(grads.shape[0], n_init_points)
            grads_aligned[:use_len] = grads[:use_len]
        else:
            grads_aligned = grads

        selected_pts_mask = torch.where(torch.norm(grads_aligned, dim=-1) >= grad_threshold, True, False)
        if region_mask is not None:
            if region_mask.shape[0] != selected_pts_mask.shape[0]:
                padded_region = torch.zeros_like(selected_pts_mask)
                copy_len = min(region_mask.shape[0], selected_pts_mask.shape[0])
                padded_region[:copy_len] = region_mask[:copy_len]
                region_mask = padded_region
            selected_pts_mask = torch.logical_and(selected_pts_mask, region_mask)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_coefs    = self._coefs[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        new_deform_mask = self._deform_mask[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,new_coefs, new_deformation_table, new_deform_mask)
    
    def prune(self, max_grad, min_opacity, extent, max_screen_size, deformation_mask=None):
        """
        修剪高斯点，支持变形感知策略
        
        Args:
            max_grad: 最大梯度阈值
            min_opacity: 最小不透明度阈值
            extent: 场景范围
            max_screen_size: 最大屏幕尺寸
            deformation_mask: 变形掩码，如果提供则使用变形感知策略
        """
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        
        # 变形感知的修剪：检查变形异常的点
        if deformation_mask is not None and hasattr(self, '_deformation_accum'):
            # 确保掩码长度匹配
            if len(deformation_mask) <= len(self._deformation_accum):
                deform_accum = self._deformation_accum[:len(deformation_mask)].norm(dim=-1)
                # 移除变形量异常的点（过大：可能错误，过小：可能不需要变形）
                # 但只对非变形区域应用过小检查，变形区域允许小变形
                abnormal_deform_large = deform_accum > 0.5  # 变形过大
                abnormal_deform_small = (deform_accum < 0.001) & (~deformation_mask[:len(deform_accum)])  # 非变形区域变形过小
                abnormal_deform = abnormal_deform_large | abnormal_deform_small
                
                # 确保长度匹配
                if len(abnormal_deform) <= len(prune_mask):
                    prune_mask[:len(abnormal_deform)] = torch.logical_or(
                        prune_mask[:len(abnormal_deform)], abnormal_deform
                    )
        
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size, deformation_mask=None, deformation_scale=1.5):
        """
        密集化高斯点，支持变形感知策略
        
        Args:
            max_grad: 最大梯度阈值
            min_opacity: 最小不透明度阈值
            extent: 场景范围
            max_screen_size: 最大屏幕尺寸
            deformation_mask: 变形掩码，如果提供则使用变形感知策略
            deformation_scale: 变形区域的梯度阈值缩放因子（越小越容易密集化）
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # 对齐 grads 长度到当前高斯数量，避免与 self.get_scaling 等不一致
        n_init_points = self.get_xyz.shape[0]
        if grads.shape[0] != n_init_points:
            grads_aligned = torch.zeros((n_init_points, grads.shape[1]), device=grads.device, dtype=grads.dtype)
            use_len = min(grads.shape[0], n_init_points)
            grads_aligned[:use_len] = grads[:use_len]
            grads = grads_aligned

        if deformation_mask is not None and len(deformation_mask) > 0:
            # 变形感知的密集化策略
            if len(deformation_mask) > len(grads):
                deformation_mask = deformation_mask[:len(grads)]
            elif len(deformation_mask) < len(grads):
                padding = torch.zeros(len(grads) - len(deformation_mask), dtype=torch.bool, device=deformation_mask.device)
                deformation_mask = torch.cat([deformation_mask, padding])
            
            deform_grad_threshold = max_grad / deformation_scale
            static_grad_threshold = max_grad
            
            deform_mask = deformation_mask[:len(grads)]
            static_mask = ~deform_mask
            
            # 变形区域密集化（更激进）——使用 region_mask 保持长度一致
            if deform_mask.any():
                self.densify_and_clone(grads, deform_grad_threshold, extent, region_mask=deform_mask)
                self.densify_and_split(grads, deform_grad_threshold, extent, region_mask=deform_mask)
            
            # 静态区域密集化（保守）
            if static_mask.any():
                self.densify_and_clone(grads, static_grad_threshold, extent, region_mask=static_mask)
                self.densify_and_split(grads, static_grad_threshold, extent, region_mask=static_mask)
        else:
            # 原有逻辑（无变形感知）
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)
    
    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, deformation_mask=None, deform_weight=1.2):
        """
        添加密集化统计信息，支持变形感知策略
        
        Args:
            viewspace_point_tensor: 视图空间点张量
            update_filter: 更新过滤器
            deformation_mask: 变形掩码，如果提供则对变形区域加权
            deform_weight: 变形区域的梯度权重
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
        # 对变形区域进行额外加权
        if deformation_mask is not None and len(deformation_mask) > 0:
            # 确保掩码长度匹配
            if len(deformation_mask) > len(update_filter):
                deformation_mask = deformation_mask[:len(update_filter)]
            elif len(deformation_mask) < len(update_filter):
                padding = torch.zeros(len(update_filter) - len(deformation_mask), 
                                    dtype=torch.bool, device=deformation_mask.device)
                deformation_mask = torch.cat([deformation_mask, padding])
            
            # 变形区域的梯度需要更高权重
            deform_filter = update_filter & deformation_mask
            if deform_filter.any() and len(self.xyz_gradient_accum) > len(deform_filter):
                self.xyz_gradient_accum[deform_filter] *= deform_weight

    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)
        
    def gaussian_deformation(self, time, deformation_point, num_gaussians, ch_num=8, basis_num=20):
        """
        多尺度/分层时间基函数：
        - 将原来的 basis_num 个一维 Gaussian 基函数拆成「粗尺度 + 细尺度」两组
        - 粗尺度：更大的时间支撑（std 更大），建模整体/低频形变
        - 细尺度：更小的时间支撑（std 更小），建模局部/高频残差

        仍保持对外接口不变，只在内部对 std 做分组缩放。
        """

        num_points = len(self._xyz)
        coefficients = self._coefs.reshape(num_points, ch_num, 3, basis_num).contiguous()
        coefficients = coefficients[deformation_point, :, :, :]
        weights, means, std_devs = torch.chunk(coefficients, 3, -2)

        # --- 多尺度划分：前半为粗尺度，后半为细尺度 ---
        # 若为奇数，粗尺度多一个基函数
        coarse_basis_num = basis_num // 2 + basis_num % 2
        fine_basis_num = basis_num - coarse_basis_num

        # 避免极端情况下 fine_basis_num 为 0（很小的 basis_num）
        if fine_basis_num == 0:
            exponent = (time - means) ** 2 / (std_devs ** 2 + 1e-4)
            gaussians = torch.exp(-exponent ** 2)
            return (gaussians * weights).sum(-1).squeeze()

        weights_coarse = weights[..., :coarse_basis_num]
        means_coarse = means[..., :coarse_basis_num]
        std_coarse = std_devs[..., :coarse_basis_num]

        weights_fine = weights[..., coarse_basis_num:]
        means_fine = means[..., coarse_basis_num:]
        std_fine = std_devs[..., coarse_basis_num:]

        # --- 不同尺度的 std 缩放，形成明显的支撑差异 ---
        # 粗尺度：时间范围更大（std 放大）
        # 细尺度：时间范围更小（std 缩小），专注局部细节
        coarse_scale = 3.5
        fine_scale = 1.5

        std_coarse = std_coarse * coarse_scale
        std_fine = std_fine * fine_scale

        exponent_coarse = (time - means_coarse) ** 2 / (std_coarse ** 2 + 1e-4)
        exponent_fine = (time - means_fine) ** 2 / (std_fine ** 2 + 1e-4)

        gaussians_coarse = torch.exp(-exponent_coarse ** 2)
        gaussians_fine = torch.exp(-exponent_fine ** 2)

        deform_coarse = (gaussians_coarse * weights_coarse).sum(-1)
        deform_fine = (gaussians_fine * weights_fine).sum(-1)

        # 可以在这里添加额外的层级权重（目前设为 1 保持行为接近原版）
        return (deform_coarse + deform_fine).squeeze()


    def apply_deformations(
        self, 
        xyz: torch.Tensor, 
        scales: torch.Tensor, 
        rotations: torch.Tensor, 
        opacities: torch.Tensor, 
        deformation_point: torch.Tensor, 
        deformations: torch.Tensor, 
        time: float, 
        use_adaptive_hierarchy: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        deformation_config = {
            "xyz_scale": 4.0,
            "rotation_scale": 6.0,
            "opacity_scale": 20.0
        }

        if use_adaptive_hierarchy:
            xyz_deform = deformations[:, :3]
            rot_deform = deformations[:, 3:7]
            opacity_deform = deformations[:, 7:8]

            # Apply deformations using configuration
            xyz += deformation_config["xyz_scale"] * xyz_deform
            rotations += deformation_config["rotation_scale"] * rot_deform
            opacities += deformation_config["opacity_scale"] * opacity_deform

        return xyz, scales, rotations, opacities

    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    
    def compute_sparsity_regulation(self,):
        N = len(self._xyz)
        ch_num = self.args.ch_num
        coefs = self._coefs.reshape(N, ch_num, -1).contiguous() # [N, 7, ORDER_NUM + ORDER_NUM * 2 ]
        return (torch.sum(torch.abs(coefs), dim=-1, keepdim=True)\
            /torch.abs(coefs.max(dim=-1, keepdim = True)[0])).mean()   

    def compute_l1_regulation(self,):

        return (torch.abs(self._coefs)).mean()
    
    def compute_l2_regulation(self,):

        return (self._coefs**2).mean()
    
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()

    def save_ply_with_custom_data(self, path, xyz, scales, rotations, opacities):
        
        mkdir_p(os.path.dirname(path))
        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacities.detach().cpu().numpy()  
        scales = scales.detach().cpu().numpy()  
        coefs = self._coefs.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()  

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotations, coefs), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
