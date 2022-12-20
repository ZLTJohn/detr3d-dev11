import os
import time
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torchvision.utils as vutils
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, bbox3d2result
from mmengine.structures import InstanceData
from torch import Tensor

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.old_env import auto_fp16, force_fp32


# from mmdet3d.structures.bbox_3d.utils import get_lidar2img #!!!
# from .visualizer_zlt import *
@MODELS.register_module()
class Detr3D(MVXTwoStageDetector):
    """Detr3D."""

    def __init__(self,
                 data_preprocessor=None,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 debug_name=None,
                 gtvis_range=[0, 105],
                 vis_count=None):
        super(Detr3D, self).__init__(img_backbone=img_backbone,
                                     img_neck=img_neck,
                                     pts_bbox_head=pts_bbox_head,
                                     train_cfg=train_cfg,
                                     test_cfg=test_cfg,
                                     data_preprocessor=data_preprocessor)
        self.grid_mask = GridMask(True,
                                  True,
                                  rotate=1,
                                  offset=False,
                                  ratio=0.5,
                                  mode=1,
                                  prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.debug_name = debug_name
        self.gtvis_range = gtvis_range
        self.vis_count = vis_count

    def extract_img_feat(self, img: Tensor,
                         input_metas: List[dict]) -> List[Tensor]:
        """Extract features of images."""

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]  #bs nchw
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)  # mask out some grids
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, batch_inputs_dict: Dict,
                     batch_input_metas: List[dict]) -> List[Tensor]:
        """Extract features from images and points."""
        imgs = batch_inputs_dict.get('imgs', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        return img_feats

    def _forward(self):
        raise NotImplementedError(f'tensor mode is yet to add')

    ##original forward_train
    def loss(self, batch_inputs_dict: Dict[List, Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        outs = self.pts_bbox_head(img_feats, batch_input_metas, **kwargs)

        batch_gt_instances = [
            item.gt_instances_3d for item in batch_data_samples
        ]
        loss_inputs = [batch_gt_instances, outs]
        losses_pts = self.pts_bbox_head.loss_by_feat(*loss_inputs)

        return losses_pts

    #original simple_test
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        outs = self.pts_bbox_head(img_feats, batch_input_metas)

        results_list_3d = self.pts_bbox_head.predict_by_feat(
            outs, batch_input_metas, **kwargs)  #rescale in kwargs

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d)
        return detsamples

    #may need speed-up
    def add_lidar2img(self, batch_input_metas: List[Dict]) -> List[Dict]:
        for meta in batch_input_metas:
            l2i = list()
            for i in range(len(meta['cam2img'])):
                c2i = torch.tensor(meta['cam2img'][i]).double()
                l2c = torch.tensor(meta['lidar2cam'][i]).double()
                l2i.append(get_lidar2img(c2i, l2c).float().numpy())
            meta['lidar2img'] = l2i
        return batch_input_metas

#https://github.com/open-mmlab/mmdetection3d/pull/2110
update_info_BUG_FIX=True 
def get_lidar2img(cam2img, lidar2cam):
    """Get the projection matrix of lidar2img.

    Args:
        cam2img (torch.Tensor): A 3x3 or 4x4 projection matrix.
        lidar2cam (torch.Tensor): A 3x3 or 4x4 projection matrix.

    Returns:
        torch.Tensor: transformation matrix with shape 4x4.
    """
    if update_info_BUG_FIX == False:
        lidar2cam_r = lidar2cam[:3, :3]
        lidar2cam_t = lidar2cam[:3, 3]
        lidar2cam_t = torch.matmul(lidar2cam_t, lidar2cam_r.T)
        lidar2cam[:3, 3] = lidar2cam_t
    if cam2img.shape == (3, 3):
        temp = cam2img.new_zeros(4, 4)
        temp[:3, :3] = cam2img
        temp[3, 3] = 1
        cam2img = temp

    if lidar2cam.shape == (3, 3):
        temp = lidar2cam.new_zeros(4, 4)
        temp[:3, :3] = lidar2cam
        temp[3, 3] = 1
        lidar2cam = temp
    return torch.matmul(cam2img, lidar2cam)
