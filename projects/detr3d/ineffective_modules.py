import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, constant_init, xavier_init
from .detr3d_featsampler import GeoAwareFeatSampler
from .detr3d_transformer import Detr3DCrossAtten, inverse_sigmoid
from typing import Dict, List, Optional
from torch import Tensor
from .detr3d import DETR3D


@MODELS.register_module()
class CameraAwareFeatSampler(GeoAwareFeatSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.virtual_cam_dist = 0

    def get_scale_factor(self, ref_pt, pc_range, img_metas):
        cam2refpt, ego2refpt  = self.CamCenters2Objects(ref_pt, pc_range, 
                                                        img_metas, refpt_dist=True)
        scale_factor = (ego2refpt - self.virtual_cam_dist) / cam2refpt
        return scale_factor

@MODELS.register_module()
class Detr3DCrossAtten_CamEmb(Detr3DCrossAtten):
    def __init__(self,
                 num_cams = 1,
                 **kwargs):
        super().__init__(num_cams=1, **kwargs)

    def forward(self,
                query, 
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos
        query = query.permute(1, 0, 2)
        bs, num_query, _ = query.size()
        
        ego2cam = query.new_tensor([meta['lidar2cam'] for meta in kwargs['img_metas']]) # or maybe we can do cam2ego?
        num_view = ego2cam.shape[1]
        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, 1, 1, self.num_levels)
        attention_weights = F.softmax(attention_weights,dim=-1)
        attention_weights = attention_weights.repeat(1,1,1,num_view,1,1)
        # breakpoint()
        reference_points_3d = reference_points.clone()
        output, mask = self.feature_sampler.forward(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        # attention_weights = attention_weights.sigmoid() * mask
        attention_weights = attention_weights * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        # (num_query, bs, embed_dims)
        output = self.output_proj(output)
        pos_feat = self.position_encoder(
            inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        return self.dropout(output) + inp_residual + pos_feat

@MODELS.register_module()
class Detr3DCrossAtten_ManyCam(Detr3DCrossAtten):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()
        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        
        reference_points_3d = reference_points.clone()
        output, mask = self.feature_sampler.forward(
            value, reference_points, self.pc_range, kwargs['img_metas'])

        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        # argo: 7; nuscenes: 6; waymo:5
        num_img = mask.shape[3]
        if num_img == 7: 
            pre,suf = 0,11
        elif num_img == 6: 
            pre,suf = 7,5
        else: 
            pre,suf = 13,0

        if pre>0:
            mask_ext = torch.zeros_like(mask[:,:,:,0:1, ...]).repeat(1,1,1,pre,1,1)
            output_ext = torch.zeros_like(output[:,:,:,0:1, ...]).repeat(1,1,1,pre,1,1)
            mask = torch.cat((mask_ext, mask),dim=3)
            output = torch.cat((output_ext,output),dim=3)
        if suf>0:
            mask_ext = torch.zeros_like(mask[:,:,:,0:1, ...]).repeat(1,1,1,suf,1,1)
            output_ext = torch.zeros_like(output[:,:,:,0:1, ...]).repeat(1,1,1,suf,1,1)
            mask = torch.cat((mask, mask_ext),dim=3)
            output = torch.cat((output,output_ext),dim=3)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        # (num_query, bs, embed_dims)
        output = self.output_proj(output)
        pos_feat = self.position_encoder(
            inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        return self.dropout(output) + inp_residual + pos_feat

import torch.nn.functional as F
@MODELS.register_module()
class debugDETR3D(DETR3D):
    def extract_feat(self, batch_inputs_dict: Dict, batch_input_metas: List[dict]) -> List[Tensor]:
        feats = super().extract_feat(batch_inputs_dict, batch_input_metas)
        feats = self.rescale_feats(feats, batch_input_metas[0]['ksync_factor'][0])
        return feats
        
    def rescale_feats(self,feats, scale):
        ret = []
        for i in range(len(feats)):
            feat = feats[i]
            B,N,C,H,W = feat.shape
            ret.append(F.interpolate(feat.view(-1,C,H,W),scale_factor=scale).unsqueeze(0))
        return ret


# #https://github.com/open-mmlab/mmdetection3d/pull/2110
# update_info_BUG_FIX = True

# def get_lidar2img(cam2img, lidar2cam):
#     """Get the projection matrix of lidar2img.

#     Args:
#         cam2img (torch.Tensor): A 3x3 or 4x4 projection matrix.
#         lidar2cam (torch.Tensor): A 3x3 or 4x4 projection matrix.

#     Returns:
#         torch.Tensor: transformation matrix with shape 4x4.
#     """
#     if update_info_BUG_FIX == False:
#         lidar2cam_r = lidar2cam[:3, :3]
#         lidar2cam_t = lidar2cam[:3, 3]
#         lidar2cam_t = torch.matmul(lidar2cam_t, lidar2cam_r.T)
#         lidar2cam[:3, 3] = lidar2cam_t
#     if cam2img.shape == (3, 3):
#         temp = cam2img.new_zeros(4, 4)
#         temp[:3, :3] = cam2img
#         temp[3, 3] = 1
#         cam2img = temp

#     if lidar2cam.shape == (3, 3):
#         temp = lidar2cam.new_zeros(4, 4)
#         temp[:3, :3] = lidar2cam
#         temp[3, 3] = 1
#         lidar2cam = temp
#     return torch.matmul(cam2img, lidar2cam)