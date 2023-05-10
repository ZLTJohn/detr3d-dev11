from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet.models import DETECTORS
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.centerpoint import CenterPoint
from typing import Dict, List, Optional
from torch import Tensor
from mmdet3d.structures import Det3DDataSample
from projects.detr3d.vis_zlt import visualizer_zlt
@MODELS.register_module()
class BEVDet(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.
    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_
    Args:
        view_transformer (dict): Configuration dict of view transformer.
        bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, debug_vis_cfg=None, **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.img_view_transformer = MODELS.build(img_view_transformer)
        self.img_bev_encoder_backbone = MODELS.build(
            img_bev_encoder_backbone)
        self.img_bev_encoder_neck = MODELS.build(img_bev_encoder_neck)

        if debug_vis_cfg is not None:
            self.vis = visualizer_zlt(**debug_vis_cfg)
        else:
            self.vis = None

    def image_encoder(self, imgs):
        """Image-view feature encoder."""
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
        assert len(x) == 1
        _, out_C, out_H, out_W = x[0].shape
        x[0] = x[0].view(B, N, out_C, out_H, out_W)
        return x

    def bev_encoder(self, x):
        """Bird-Eye-View feature encoder."""
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        return x

    def extract_img_feat(self, img_inputs, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img_inputs)
        x = self.img_view_transformer(x, img_metas)
        x = self.bev_encoder(x)
        return x

    def extract_feat(self, img_inputs, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img_inputs, img_metas)
        return img_feats

    def loss(self, batch_inputs_dict: Dict[List, Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs):
        img_metas = [item.metainfo for item in batch_data_samples]
        # batch_input_metas = self.add_lidar2img(batch_input_metas)
        img_inputs = batch_inputs_dict.get('imgs', None)
        img_feats = self.extract_feat(img_inputs, img_metas)
        losses = dict()
        losses_pts = self.pts_bbox_head.loss(img_feats, batch_data_samples,
                                             **kwargs)
        if self.vis is not None:
            batch_gt_instances_3d = [
                item.gt_instances_3d for item in batch_data_samples
            ]
            self.vis.visualize(batch_gt_instances_3d, img_metas,
                               batch_inputs_dict.get('imgs', None))
        losses.update(losses_pts)
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs):
        """Test function without augmentaiton."""
        img_metas = [item.metainfo for item in batch_data_samples]
        img_inputs = batch_inputs_dict.get('imgs', None)
        img_feats = self.extract_feat(img_inputs, img_metas)
        # breakpoint()
        results_list_3d = self.pts_bbox_head.predict(img_feats, batch_data_samples, **kwargs)
        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d)
        if self.vis is not None:
            batch_gt_instances_3d = [
                item.gt_instances_3d for item in batch_data_samples]
            if len(batch_gt_instances_3d[0])!=0:
                self.vis.visualize(batch_gt_instances_3d, img_metas,name_suffix='_gt')
                self.vis.visualize(results_list_3d, img_metas,
                               batch_inputs_dict.get('imgs', None))
            else:
                print('skipping one frame since no gt left')
        return detsamples

    # def forward_dummy(self,
    #                   points=None,
    #                   img_metas=None,
    #                   img_inputs=None,
    #                   **kwargs):
    #     """Dummy forward function."""
    #     img_feats, _ = self.extract_feat(img_inputs, img_metas)
    #     img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
    #     bbox_list = [dict() for _ in range(1)]
    #     assert self.with_pts_bbox
    #     bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=False)
    #     for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
    #         result_dict['pts_bbox'] = pts_bbox
    #     return bbox_list