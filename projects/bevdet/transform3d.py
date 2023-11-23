from typing import Any
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.loading import LoadMultiViewImageFromFiles
from mmdet.datasets.transforms import Rotate
from mmdet3d.datasets.transforms.transforms_3d import RandomCrop
# from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from projects.cross_dataset.transform_3d import Pack3DDetInputsExtra
from mmengine.fileio import get
import numpy as np
import torch

@TRANSFORMS.register_module()
class Pack3DDetInputsBEVDet(Pack3DDetInputsExtra):
    def __init__(self, **kwargs) -> None:
        extra_keys = [
            'bevdet_input'
        ]
        super().__init__(**kwargs)
        self.meta_keys = tuple(list(self.meta_keys)+extra_keys)

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
    def add_cam2lidar(self, results):
        l2c = np.array(results['lidar2cam'])
        c2l = np.linalg.inv(l2c)
        results['cam2lidar'] = c2l

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
    def to4x4(self, mat):
        temp = mat.new_zeros(mat.shape[0],4,4)
        temp[:,:3,:3] = mat[:,:3,:3]
        temp[:,3,3] = 1
        return temp

    def __call__(self, results):
        assert 'crop' in results
        self.add_cam2lidar(results)
        results['cam_intrinsic'] = results['cam2img']
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
        aug_transform = torch.zeros((len(results['img']), 4, 4)).float()
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
        intrins = self.to4x4(intrins)
        results['bevdet_input'] = [rots, trans, intrins, post_rots, post_trans]
        return results
    

@TRANSFORMS.register_module()
class RangeLimitedRandomCrop(RandomCrop):
    """Randomly crop image-view objects under a limitation of range.
    Args:
        relative_x_offset_range (tuple[float]): Relative range of random crop
            in x direction. (x_min, x_max) in [0, 1.0]. Default to (0.0, 1.0).
        relative_y_offset_range (tuple[float]): Relative range of random crop
            in y direction. (y_min, y_max) in [0, 1.0]. Default to (0.0, 1.0).
    """

    def __init__(self,
                 relative_x_offset_range=(0.0, 1.0),
                 relative_y_offset_range=(0.0, 1.0),
                 **kwargs):
        super(RangeLimitedRandomCrop, self).__init__(**kwargs)
        for range in [relative_x_offset_range, relative_y_offset_range]:
            assert 0 <= range[0] <= range[1] <= 1
        self.relative_x_offset_range = relative_x_offset_range
        self.relative_y_offset_range = relative_y_offset_range

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images.
        Modified from RandomCrop in mmdet==2.25.0
        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_range_h = (margin_h * self.relative_y_offset_range[0],
                              margin_h * self.relative_y_offset_range[1] + 1)
            offset_h = np.random.randint(*offset_range_h)
            offset_range_w = (margin_w * self.relative_x_offset_range[0],
                              margin_w * self.relative_x_offset_range[1] + 1)
            offset_w = np.random.randint(*offset_range_w)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
            results['crop'] = (crop_x1, crop_y1, crop_x2, crop_y2)
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results
    

@TRANSFORMS.register_module()
class RandomRotate(Rotate):
    """Randomly rotate images.
    The ratation angle is selected uniformly within the interval specified by
    the 'range'  parameter.
    Args:
        range (tuple[float]): Define the range of random rotation.
            (angle_min, angle_max) in angle.
    """

    def __init__(self, range, **kwargs):
        super(RandomRotate, self).__init__(**kwargs)
        self.range = range

    def __call__(self, results):
        self.angle = np.random.uniform(self.range[0], self.range[1])
        super(RandomRotate, self).__call__(results)
        results['rotate'] = self.angle
        return results