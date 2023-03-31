from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.loading import LoadMultiViewImageFromFiles
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
import copy
import mmengine
from mmengine.fileio import get
import mmcv
import numpy as np
import os.path as osp
import torch
import random
import time

@TRANSFORMS.register_module()
class GetBEVDetInputs():
    """Generate the inputs of BEVDet including the images and the
    transformation information for the Lift-Splat-Shoot view transformer."""
    def get_rot(self, rad):
        """Generate 2D rotation matrix according to the input radian.
        Args:
            rad (float): Ratation magnitude in radian.
        Returns:
            torch.Tensor: The 2D rotation matrix in shape of (2, 2).
        """
        return torch.Tensor([
            [np.cos(rad), np.sin(rad)],
            [-np.sin(rad), np.cos(rad)],
        ])

    def get_post_transform(self, resize, crop, flip, rotate):
        """Generate 3D translation and rotation matrix according to the image
        view data transformation.
        Args:
            resize (float): Scale of resize.
            crop (tuple(int)): Range of cropping in format of (lower_w,
                lower_h, upper_w, upper_h).
            flip (bool): Flag of flip operation.
            rotate (float): Magnitude of rotation in angle.
        Returns:
            tuple(torch.Tensor): The 3D translation and rotation matrix.
        """
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        post_tran_3d = torch.zeros(3).float()
        post_rot_3d = torch.eye(3).float()
        post_tran_3d[:2] = post_tran
        post_rot_3d[:2, :2] = post_rot
        return post_rot_3d, post_tran_3d

    def __call__(self, results):
        assert 'crop' in results
        imgs = torch.tensor(np.stack(results['img'])). \
            permute((0, 3, 1, 2)).contiguous()
        cam2lidar = torch.tensor(np.stack(results['cam2lidar'])).float()
        intrins = torch.tensor(np.stack(results['cam_intrinsic'])).float()

        # post_rots and post_trans for image view data augmentation
        post_rots = []
        post_trans = []
        for img_id in range(len(results['img'])):
            crop = results['crop'][img_id]
            crop = (crop[0], crop[1],
                    crop[0] + results['pad_shape'][img_id][1],
                    crop[1] + results['pad_shape'][img_id][0])
            flip = False if 'flip' not in results else results['flip'][img_id]
            rotate = 0.0 if 'rotate' not in results \
                else results['rotate'][img_id]
            post_rot, post_tran = self.get_post_transform(
                results['scale_factor'][img_id][0], crop, flip, rotate)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
        post_rots, post_trans = torch.stack(post_rots), torch.stack(post_trans)

        # update cam2lidar according to the augmentation in Bird-Eye-View
        aug_transform = torch.zeros((imgs.shape[0], 4, 4)).float()
        aug_transform[:, -1, -1] = 1.0
        # update for GlobalRotScaleTrans
        # rotate
        if 'pcd_rotation' in results:
            rotation = results['pcd_rotation'].T
        else:
            rotation = torch.eye(3).view(1, 3, 3)
        # scale
        if 'pcd_scale_factor' in results:
            rotation = rotation * results['pcd_scale_factor']
        aug_transform[:, :3, :3] = rotation
        # translate
        if 'pcd_trans' in results:
            aug_transform[:, :3, -1] = \
                torch.from_numpy(results['pcd_trans']).reshape(1, 3)

        # update for RandomFlip3D
        if 'pcd_horizontal_flip' in results and results['pcd_horizontal_flip']:
            aug_transform[:, 1, :] = aug_transform[:, 1, :] * -1
        if 'pcd_vertical_flip' in results and results['pcd_vertical_flip']:
            aug_transform[:, 0, :] = aug_transform[:, 0, :] * -1
        cam2lidar = aug_transform.matmul(cam2lidar)

        rots = cam2lidar[:, :3, :3]
        trans = cam2lidar[:, :3, 3]
        results['img_inputs'] = \
            (imgs, rots, trans, intrins, post_rots, post_trans)
        return results