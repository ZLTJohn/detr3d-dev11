import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d.utils import get_lidar2img
from torch import Tensor

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

from .vis_zlt import visualizer_zlt


@MODELS.register_module()
class Detr3D(MVXTwoStageDetector):
    """DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

    Args:
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
        use_grid_mask (bool) : Data augmentation. Whether to mask out some
            grids during extract_img_feat. Defaults to False.
        img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        pts_bbox_head (dict, optional): Bboxes head of
            detr3d. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor=None,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 debug_vis_cfg=None):
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
        if debug_vis_cfg is not None:
            self.vis = visualizer_zlt(**debug_vis_cfg)
        else:
            self.vis = None

    def extract_img_feat(self, img: Tensor,
                         batch_input_metas: List[dict]) -> List[Tensor]:
        """Extract features from images.

        Args:
            img (tensor): Batched multi-view image tensor with shape (B, N, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             list[tensor]: multi-level image features.
        """

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]  # bs nchw
            # update real input shape of each single img
            for img_meta in batch_input_metas:
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
        """Extract features from images.

        Refer to self.extract_img_feat()
        """
        imgs = batch_inputs_dict.get('imgs', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        return img_feats

    def _forward(self):
        raise NotImplementedError(f'tensor mode is yet to add')

    # original forward_train
    def loss(self, batch_inputs_dict: Dict[List, Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `imgs` keys.
                - imgs (torch.Tensor): Tensor of batched multi-view  images, has shape (B, N, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data Samples. It usually includes information such as `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        outs = self.pts_bbox_head(img_feats, batch_input_metas, **kwargs)

        batch_gt_instances_3d = [
            item.gt_instances_3d for item in batch_data_samples
        ]
        loss_inputs = [batch_gt_instances_3d, outs]
        losses_pts = self.pts_bbox_head.loss_by_feat(*loss_inputs)

        if self.vis is not None:
            self.vis.visualize(batch_gt_instances_3d, batch_input_metas, batch_inputs_dict.get('imgs', None))
        return losses_pts

    # original simple_test
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `imgs` keys.

                - imgs (torch.Tensor): Tensor of batched multi-view images, has shape (B, N, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 9).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        outs = self.pts_bbox_head(img_feats, batch_input_metas)

        results_list_3d = self.pts_bbox_head.predict_by_feat(
            outs, batch_input_metas, **kwargs)
        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d)
        if self.vis is not None:
            self.vis.visualize(results_list_3d, batch_input_metas, batch_inputs_dict.get('imgs', None))
        return detsamples

    # may need speed-up
    def add_lidar2img(self, batch_input_metas: List[Dict]) -> List[Dict]:
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            l2i = list()
            for i in range(len(meta['cam2img'])):
                c2i = torch.tensor(meta['cam2img'][i]).double()
                l2c = torch.tensor(meta['lidar2cam'][i]).double()
                l2i.append(get_lidar2img(c2i, l2c).float().numpy())
            meta['lidar2img'] = l2i
        return batch_input_metas


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
