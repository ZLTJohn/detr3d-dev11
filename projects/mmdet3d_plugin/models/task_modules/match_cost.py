from typing import List, Optional, Union

import torch
from mmdet3d.registry import TASK_UTILS
from torch import Tensor


@TASK_UTILS.register_module()
class BBox3DL1Cost(object):

    def __init__(self, weight: Union[float, int] = 1.):
        self.weight = weight

    def __call__(self, bbox_pred: Tensor, gt_bboxes: Tensor) -> Tensor:
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
