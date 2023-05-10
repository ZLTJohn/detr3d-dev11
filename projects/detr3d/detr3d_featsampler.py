from typing import Dict, List, Optional
from torch import Tensor
from mmdet3d.registry import MODELS
import numpy as np
import torch
import torch.nn.functional as F

@MODELS.register_module()
class DefaultFeatSampler:
    '''Default feature sampler of DETR3D'''
    def __init__(self):
        self.creator = 'zltjohn'
    
    def denormalize(self, ref_pt, pc_range):
        '''
        Denormalize reference points according to perception_range
        Args:
            ref_pt: ref points
            pc_range: List of number: [X0,Y0,Z0,X1,Y1,Z1]
        '''
        [X0,Y0,Z0,X1,Y1,Z1] = pc_range
        ref_pt[..., 0:1] = ref_pt[..., 0:1] * (X1 - X0) + X0
        ref_pt[..., 1:2] = ref_pt[..., 1:2] * (Y1 - Y0) + Y0
        ref_pt[..., 2:3] = ref_pt[..., 2:3] * (Z1 - Z0) + Z0
        return ref_pt

    def project_ego2cam(self,
                        ref_pt: Tensor,
                        pc_range: List,
                        img_metas: List[Dict],
                        return_depth = False):
        '''
        Project normalized reference points to images according to projection matrix
        Args:
            ref_pt: 3D normalized Reference points, in shape [Batch, Query, 3]
            pc_range: perception range
            img_metas: metainfos that contains 'lidar2img'
        Return:
            pt_cam: projected 2D points in nomalized image coordinate, 
                    in shape [Batch, num_camera, Query, 2]
            mask: whether pt_cam is inside of image,
                    in shape [Batch, num_camera, Query, 1]

        '''
        lidar2img = [meta['lidar2img'] for meta in img_metas]
        lidar2img = np.asarray(lidar2img)
        lidar2img = ref_pt.new_tensor(lidar2img)
        ref_pt = ref_pt.clone()

        B, Q = ref_pt.size()[:2]
        N = lidar2img.size(1)
        eps = 1e-5

        ref_pt = self.denormalize(ref_pt, pc_range)
        # (B num_q 3) -> (B num_q 4) -> (B 1 num_q 4) -> (B num_cam num_q 4 1)
        ref_pt = torch.cat((ref_pt, torch.ones_like(ref_pt[..., :1])), -1)
        ref_pt = ref_pt.view(B, 1, Q, 4)
        ref_pt = ref_pt.repeat(1, N, 1, 1).unsqueeze(-1)
        # (B num_cam 4 4) -> (B num_cam num_q 4 4)
        lidar2img = lidar2img.view(B, N, 1, 4, 4)\
                           .repeat(1, 1, Q, 1, 1)
        # (... 4 4) * (... 4 1) -> (B num_cam num_q 4)
        pt_cam = torch.matmul(lidar2img, ref_pt).squeeze(-1)

        # (B num_cam num_q)
        z = pt_cam[..., 2:3]
        eps = eps * torch.ones_like(z)
        mask = (z > eps)
        pt_cam = pt_cam[..., 0:2] / torch.maximum(z, eps)  # prevent zero-division
        # padded nuscene image: 928*1600, 
        # nevermind mask in waymo cam3~4 since padded regions are 0
        (h, w) = img_metas[0]['pad_shape']
        pt_cam[..., 0] /= w
        pt_cam[..., 1] /= h
        mask = (mask & (pt_cam[..., 0:1] > 0.0)
                & (pt_cam[..., 0:1] < 1.0)
                & (pt_cam[..., 1:2] > 0.0)
                & (pt_cam[..., 1:2] < 1.0))
        if return_depth:
            return pt_cam, mask, z
        else:
            return pt_cam, mask
    
    def forward(self,
                mlvl_feats,
                ref_pt,
                pc_range,
                img_metas):
        pt_cam, mask = self.project_ego2cam(ref_pt, pc_range, img_metas)
        B, N, Q, _ = mask.shape
        # mask[:, 3:5, :] &= (pt_cam[:, 3:5, :, 1:2] < 0.7)
        # (B num_cam num_q) -> (B 1 num_q num_cam 1 1)
        mask = mask.view(B, N, 1, Q, 1, 1)\
                .permute(0, 2, 3, 1, 4, 5)
        mask = torch.nan_to_num(mask)

        pt_cam = (pt_cam - 0.5) * 2  # [0,1] to [-1,1] to do grid_sample
        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            B, N, C, H, W = feat.size()
            feat = feat.view(B * N, C, H, W)
            pt_cam_lvl = pt_cam.view(B * N, Q, 1, 2)
            sampled_feat = F.grid_sample(feat, pt_cam_lvl)
            # (B num_cam C num_query 1) -> List of (B C num_q num_cam 1)
            sampled_feat = sampled_feat.view(B, N, C, Q, 1)
            sampled_feat = sampled_feat.permute(0, 2, 3, 1, 4)
            sampled_feats.append(sampled_feat)

        sampled_feats = torch.stack(sampled_feats, -1)
        # (B C num_q num_cam fpn_lvl)
        sampled_feats = sampled_feats.view(B, C, Q, N, 1, len(mlvl_feats))
        return sampled_feats, mask

@MODELS.register_module()
class GeoAwareFeatSampler(DefaultFeatSampler):
    '''
    Geometry aware feature sampler, we consider:
            pixel_size = object_size * focal_length_k / (q_j-t_i)
        where 'q_j' is the coordinate of j-th reference point,
              't_i' means the coordinate of i-th camera center,
        all in ego frame
    Args:
        offset_2d:  sampling points relative to the projected 2d query points
        base_dist:  default (q_j-t_i) 
        base_fxfy:  vitual camera focal length
        pooling:    how to fuse sampling points per query
        debug:      for visualization
    '''
    def __init__(self,
                 offset_2d = [[0,0],[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5],[0.5,0.5]],
                 base_dist = 51.2,
                 base_fxfy = -1,
                 pooling = 'avgpool',
                 debug = False):
        self.offset_2d = offset_2d
        self.base_dist = base_dist
        self.base_fxfy = base_fxfy # currently assumes fx==fy
        if pooling == 'avgpool':
            self.pooling = torch.mean
        elif pooling == 'maxpool':
            self.pooling = torch.max
        self.debug = debug

    def CamCenters2Objects(self, ref_pt, pc_range, img_metas, refpt_dist=False):
        '''
        Calculate distance between ref_pt and different camera centers
        Args:
            ref_pt: Normalized reference points
            pc_range: perception range
            img_metas: metainfos that contains 'lidar2img'
            refpt_dist: whether to return distance from ego to refpt,
                default to False
        Return:
            cam2refpt: in shape [B,N,Q]
            ego2refpt (optional): in shape [B,N,Q]
        '''
        lidar2cam = [meta['lidar2cam'] for meta in img_metas]
        lidar2cam = np.asarray(lidar2cam)
        cam2lidar = np.linalg.inv(lidar2cam)
        CamCenters = ref_pt.new_tensor(cam2lidar[:,:,:3,-1]).unsqueeze(2)   # B N 1 3
        N = CamCenters.shape[1]
        ref_pt = self.denormalize(ref_pt, pc_range)
        ref_pt = ref_pt.unsqueeze(1).repeat(1,N,1,1)    # B 1 Q 3
        cam2refpt = torch.norm(ref_pt - CamCenters,dim=-1)
        if refpt_dist == True:
            ego2refpt = torch.norm(ref_pt,dim=-1)
            return cam2refpt, ego2refpt
        else:
            return cam2refpt
    
    def get_fxfy(self, img_metas):
        # [B, N, 3,3]
        cam2img = [meta['cam2img'] for meta in img_metas]
        cam2img = np.asarray(cam2img)
        fxfy = (cam2img[:,:,0:1,0] + cam2img[:,:,1:2,1])/2
        return fxfy

    def get_scale_factor(self, ref_pt, pc_range, img_metas):
        cam2refpt = self.CamCenters2Objects(ref_pt, pc_range, img_metas)
        scale_factor = torch.ones_like(cam2refpt)# B N Q
        eps = 1e-5 * torch.ones_like(cam2refpt)
        if self.base_dist != -1:
            scale_factor *= self.base_dist / torch.maximum(cam2refpt,eps)
        if self.base_fxfy != -1:
            # fxfy focals in [B N,1]
            cams_fxfy = ref_pt.new_tensor(self.get_fxfy(img_metas))
            scale_factor *= cams_fxfy / self.base_fxfy
        return scale_factor

    def forward(self,
                mlvl_feats,
                ref_pt,
                pc_range,
                img_metas):
        ref_pt = ref_pt.clone()
        pt_cam, mask = self.project_ego2cam(ref_pt, pc_range, img_metas)
        offset_2d = self.offset_2d
        K = len(offset_2d)
        B, N, Q, _ = mask.shape

        scale_factor = self.get_scale_factor(ref_pt, pc_range, img_metas)
        scale_factor = scale_factor.view(B,N,Q,1,1)
        offset_2d = ref_pt.new_tensor(offset_2d).view(1,1,1,K,2)
        offset_pt_cam = offset_2d * scale_factor
        pt_cam = pt_cam.view(B,N,Q,1,2)

        vis0,vis1=[],[]
        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            B, N, C, H, W = feat.size()
            feat = feat.view(B * N, C, H, W)
            offset_pt_cam_lvl = offset_pt_cam.clone()
            offset_pt_cam_lvl[...,0] /= W
            offset_pt_cam_lvl[...,1] /= H
            pt_cam_lvl = pt_cam + offset_pt_cam_lvl
            
            if self.debug == True:
                vis0.append(pt_cam_lvl.view(B,N,Q*K,2))
                vis1.append(mask.view(B,N,Q,1).repeat(1,1,1,K).view(B,N,Q*K))
                # 1,N,Q,K,2
                if lvl == len(mlvl_feats)-1:
                    vis0 = torch.stack(vis0,2).view(B,N,-1,2)
                    vis1 = torch.stack(vis1,2).view(B,N,-1)
                    return vis0,vis1
                continue

            pt_cam_lvl = (pt_cam_lvl-0.5)*2 
            pt_cam_lvl = pt_cam_lvl.view(B * N, Q*K, 1, 2)
            sampled_feat = F.grid_sample(feat, pt_cam_lvl)
            # (B num_cam C num_query 1) -> List of (B C num_q num_cam 1)
            sampled_feat = sampled_feat.view(B, N, C, Q*K, 1)\
                                    .permute(0, 2, 3, 1,   4)
            sampled_feats.append(sampled_feat)

        sampled_feats = torch.stack(sampled_feats, -1)
        # (B C num_q num_cam fpn_lvl)
        sampled_feats = sampled_feats.view(B, C, Q, K, N, 1, len(mlvl_feats))
        sampled_feats = self.pooling(sampled_feats, dim=3)
        mask = mask.view(B, N, 1, Q, 1, 1)\
                .permute(0, 2, 3, 1, 4, 5)
        mask = torch.nan_to_num(mask)
        return sampled_feats, mask

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

# TODO: make it compatible with Multi-view. Actually it is easy since you can just use 'z' in project_ego_to_cam()
@MODELS.register_module()
class FrontCameraPoseAwareFeatSampler(GeoAwareFeatSampler):
    '''
    Currently only support front camera modeling.
    '''
    def CamCenters2Objects(self, ref_pt, pc_range, img_metas, refpt_dist=False):
        '''
        Calculate distance between ref_pt and different camera centers
        Args:
            ref_pt: Normalized reference points
            pc_range: perception range
            img_metas: metainfos that contains 'lidar2img'
            refpt_dist: whether to return distance from ego to refpt,
                default to False
        Return:
            cam2refpt: in shape [B,N,Q]
            ego2refpt (optional): in shape [B,N,Q]
        '''
        lidar2cam = [meta['lidar2cam'] for meta in img_metas]
        lidar2cam = np.asarray(lidar2cam)
        cam2lidar = np.linalg.inv(lidar2cam)
        CamCenters = ref_pt.new_tensor(cam2lidar[:,:,0,-1]).unsqueeze(2)   # B N 1
        N = CamCenters.shape[1]
        ref_pt = self.denormalize(ref_pt, pc_range)
        ref_pt = ref_pt[...,0]
        ref_pt = ref_pt.unsqueeze(1).repeat(1,N,1)    # B 1 Q
        cam2refpt = (ref_pt - CamCenters)  # B N Q
        if refpt_dist == True:
            ego2refpt = torch.norm(ref_pt,dim=-1)
            return cam2refpt, ego2refpt
        else:
            return cam2refpt

# def CamCenters2Objects(ref_pt, img_metas):
#     lidar2cam = [meta['lidar2cam'] for meta in img_metas]
#     lidar2cam = np.asarray(lidar2cam)
#     cam2lidar = np.linalg.inv(lidar2cam)
#     CamCenters = ref_pt.new_tensor(cam2lidar[:,:,:3,-1]).unsqueeze(2)   # B N 1 3
#     ref_pt = ref_pt.unsqueeze(1)    # B 1 Q 3
#     cam2refpt = ref_pt - CamCenters
#     return torch.norm(cam2refpt,dim=-1)

# def feature_sampling_RoiSimple(mlvl_feats,
#                      ref_pt,
#                      pc_range,
#                      img_metas,
#                      offset_2d = [[0,0],[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5],[0.5,0.5]],
#                      vis_only = False):
#     ref_pt_3d = ref_pt.clone()
#     ref_pt = ref_pt.clone()
#     pt_cam, mask = feature_sampling(mlvl_feats, ref_pt, pc_range, img_metas, no_sampling=True)
#     K = len(offset_2d)
#     B, N, Q, _ = mask.shape
#     ref_pt[..., 0:1] = ref_pt[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]  # x
#     ref_pt[..., 1:2] = ref_pt[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]  # y
#     ref_pt[..., 2:3] = ref_pt[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]  # z
#     ref_pt_dist = CamCenters2Objects(ref_pt, img_metas)
#     # ref_pt_dist = torch.norm(ref_pt,dim=-1)# wrong,you should use extrinsics to get something shaped in [B,N,Q] but not [B,Q]
#     base_dist = 51.2
#     scale_factor = base_dist / ref_pt_dist
#     scale_factor = scale_factor.view(B,N,Q,1,1)
#     offset_2d = ref_pt.new_tensor(offset_2d).view(1,1,1,K,2)
#     offset_pt_cam = offset_2d * scale_factor
#     pt_cam = pt_cam.view(B,N,Q,1,2)
    
#     # pt_cam = (pt_cam - 0.5) * 2  # [0,1] to [-1,1] to do grid_sample
#     vis0,vis1=[],[]
#     sampled_feats = []
#     for lvl, feat in enumerate(mlvl_feats):
#         B, N, C, H, W = feat.size()
#         feat = feat.view(B * N, C, H, W)
#         offset_pt_cam_lvl = offset_pt_cam.clone()
#         offset_pt_cam_lvl[...,0] /= W
#         offset_pt_cam_lvl[...,1] /= H
#         pt_cam_lvl = pt_cam + offset_pt_cam_lvl
#         if vis_only == True:
#             vis0.append(pt_cam_lvl.view(B,N,Q*K,2))
#             vis1.append(mask.view(B,N,Q,1).repeat(1,1,1,K).view(B,N,Q*K))
#             continue
#             # 1,N,Q,K,2
#         pt_cam_lvl = (pt_cam_lvl-0.5)*2 
#         pt_cam_lvl = pt_cam_lvl.view(B * N, Q*K, 1, 2)
#         sampled_feat = F.grid_sample(feat, pt_cam_lvl)
#         # (B num_cam C num_query 1) -> List of (B C num_q num_cam 1)
#         sampled_feat = sampled_feat.view(B, N, C, Q*K, 1)
#         sampled_feat = sampled_feat.permute(0, 2, 3, 1, 4)
#         sampled_feats.append(sampled_feat)
#     if vis_only:
#         vis0 = torch.stack(vis0,2).view(B,N,-1,2)
#         vis1 = torch.stack(vis1,2).view(B,N,-1)
#         return vis0,vis1

#     sampled_feats = torch.stack(sampled_feats, -1)
#     # (B C num_q num_cam fpn_lvl)
#     sampled_feats = sampled_feats.view(B, C, Q, K, N, 1, len(mlvl_feats))
#     sampled_feats = sampled_feats.mean(3)
#     mask = mask.view(B, N, 1, Q, 1, 1).permute(0, 2, 3, 1, 4, 5)
#     mask = torch.nan_to_num(mask)
#     return ref_pt_3d, sampled_feats, mask
#     # pt_cam[..., 0] *= W
#     # pt_cam[..., 1] *= H
#     # pt_cam.view(B,N,)


# def feature_sampling(mlvl_feats,
#                      ref_pt,
#                      pc_range,
#                      img_metas,
#                      no_sampling=False):
#     """ sample multi-level features by projecting 3D reference points
#             to 2D image
#         Args:
#             mlvl_feats (List[Tensor]): Image features from
#                 different level. Each element has shape
#                 (B, N, C, H_lvl, W_lvl).
#             ref_pt (Tensor): The normalized 3D reference
#                 points with shape (bs, num_query, 3)
#             pc_range: perception range of the detector
#             img_metas (list[dict]): Meta information of multiple inputs
#                 in a batch, containing `lidar2img`.
#             no_sampling (bool): If set 'True', the function will return
#                 2D projected points and mask only.
#         Returns:
#             ref_pt_3d (Tensor): A copy of original ref_pt
#             sampled_feats (Tensor): sampled features with shape \
#                 (B C num_q N 1 fpn_lvl)
#             mask (Tensor): Determine whether the reference point \
#                 has projected outsied of images, with shape \
#                 (B 1 num_q N 1 1)
#     """
#     lidar2img = [meta['lidar2img'] for meta in img_metas]
#     lidar2img = np.asarray(lidar2img)
#     lidar2img = ref_pt.new_tensor(lidar2img)
#     ref_pt = ref_pt.clone()
#     ref_pt_3d = ref_pt.clone()

#     B, num_query = ref_pt.size()[:2]
#     num_cam = lidar2img.size(1)
#     eps = 1e-5

#     ref_pt[..., 0:1] = \
#         ref_pt[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]  # x
#     ref_pt[..., 1:2] = \
#         ref_pt[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]  # y
#     ref_pt[..., 2:3] = \
#         ref_pt[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]  # z

#     # (B num_q 3) -> (B num_q 4) -> (B 1 num_q 4) -> (B num_cam num_q 4 1)
#     ref_pt = torch.cat((ref_pt, torch.ones_like(ref_pt[..., :1])), -1)
#     ref_pt = ref_pt.view(B, 1, num_query, 4)
#     ref_pt = ref_pt.repeat(1, num_cam, 1, 1).unsqueeze(-1)
#     # (B num_cam 4 4) -> (B num_cam num_q 4 4)
#     lidar2img = lidar2img.view(B, num_cam, 1, 4, 4)\
#                          .repeat(1, 1, num_query, 1, 1)
#     # (... 4 4) * (... 4 1) -> (B num_cam num_q 4)
#     pt_cam = torch.matmul(lidar2img, ref_pt).squeeze(-1)

#     # (B num_cam num_q)
#     z = pt_cam[..., 2:3]
#     eps = eps * torch.ones_like(z)
#     mask = (z > eps)
#     pt_cam = pt_cam[..., 0:2] / torch.maximum(z, eps)  # prevent zero-division
#     # padded nuscene image: 928*1600
#     (h, w) = img_metas[0]['pad_shape']
#     pt_cam[..., 0] /= w
#     pt_cam[..., 1] /= h
#     # else:
#     # (h,w,_) = img_metas[0]['ori_shape'][0]          # waymo image
#     # pt_cam[..., 0] /= w # cam0~2: 1280*1920
#     # pt_cam[..., 1] /= h # cam3~4: 886 *1920 padded to 1280*1920
#     # mask[:, 3:5, :] &= (pt_cam[:, 3:5, :, 1:2] < 0.7) # filter pt_cam_y > 886
#     # !!!!!!!????
#     mask = (mask & (pt_cam[..., 0:1] > 0.0)
#             & (pt_cam[..., 0:1] < 1.0)
#             & (pt_cam[..., 1:2] > 0.0)
#             & (pt_cam[..., 1:2] < 1.0))

#     if no_sampling:
#         return pt_cam, mask

#     # (B num_cam num_q) -> (B 1 num_q num_cam 1 1)
#     mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
#     mask = torch.nan_to_num(mask)

#     pt_cam = (pt_cam - 0.5) * 2  # [0,1] to [-1,1] to do grid_sample
#     sampled_feats = []
#     for lvl, feat in enumerate(mlvl_feats):
#         B, N, C, H, W = feat.size()
#         feat = feat.view(B * N, C, H, W)
#         pt_cam_lvl = pt_cam.view(B * N, num_query, 1, 2)
#         sampled_feat = F.grid_sample(feat, pt_cam_lvl)
#         # (B num_cam C num_query 1) -> List of (B C num_q num_cam 1)
#         sampled_feat = sampled_feat.view(B, N, C, num_query, 1)
#         sampled_feat = sampled_feat.permute(0, 2, 3, 1, 4)
#         sampled_feats.append(sampled_feat)

#     sampled_feats = torch.stack(sampled_feats, -1)
#     # (B C num_q num_cam fpn_lvl)
#     sampled_feats = \
#         sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
#     return ref_pt_3d, sampled_feats, mask

# def feature_sampling_hotfix(mlvl_feats,
#                      ref_pt,
#                      pc_range,
#                      img_metas):
#     B, num_query = ref_pt.size()[:2]
#     num_cam = 5
#     ref_pt_3d = ref_pt.clone()
#     pt_cam, mask = feature_sampling(mlvl_feats, ref_pt, pc_range, img_metas, no_sampling=True)
#     mask[:, 3:5, :] &= (pt_cam[:, 3:5, :, 1:2] < 0.7)
#     mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
#     mask = torch.nan_to_num(mask)

#     pt_cam = (pt_cam - 0.5) * 2  # [0,1] to [-1,1] to do grid_sample
#     sampled_feats = []
#     for lvl, feat in enumerate(mlvl_feats):
#         B, N, C, H, W = feat.size()
#         feat = feat.view(B * N, C, H, W)
#         pt_cam_lvl = pt_cam.view(B * N, num_query, 1, 2)
#         sampled_feat = F.grid_sample(feat, pt_cam_lvl)
#         # (B num_cam C num_query 1) -> List of (B C num_q num_cam 1)
#         sampled_feat = sampled_feat.view(B, N, C, num_query, 1)
#         sampled_feat = sampled_feat.permute(0, 2, 3, 1, 4)
#         sampled_feats.append(sampled_feat)

#     sampled_feats = torch.stack(sampled_feats, -1)
#     # (B C num_q num_cam fpn_lvl)
#     sampled_feats = \
#         sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
#     return ref_pt_3d, sampled_feats, mask