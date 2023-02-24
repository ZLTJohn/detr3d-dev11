from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.loading import LoadMultiViewImageFromFiles
import copy
import mmengine
import mmcv
import numpy as np
@TRANSFORMS.register_module()
class filename2img_path:

    def __call__(self, results):
        results['img_path'] = results['filename']
        return results

    def __repr__(self):
        return 'maybe we need to fix this bug'

@TRANSFORMS.register_module()
class evalann2ann:

    def __call__(self, results):
        results['ann_info'] = results['eval_ann_info']
        return results

    def __repr__(self):
        return 'maybe we need to fix this bug'

@TRANSFORMS.register_module()
class PermuteImages:
    def __init__(self,new2old = [0,2,1,4,5,3]):
        self.new2old = new2old
    def __call__(self, results):
        keys = ['images', 'filename', 'cam2img', 'lidar2cam', 'ori_cam2img', 'img', 'img_path']
        for key in keys:
            if type(results[key]) == dict:
                ks,vs = [], []
                for k in results[key]:
                    ks.append(k)
                    vs.append(results[key][k])
                results[key] = {}
                for i in range(len(self.new2old)):
                    results[key][ks[self.new2old[i]]] = vs[self.new2old[i]]
            else:
                new_list = []
                for i in range(len(self.new2old)):
                    new_list.append(results[key][self.new2old[i]])
                results[key] = new_list
        return results

@TRANSFORMS.register_module()
class RotateScene_neg90:
    '''rotate whole scene by 90 degree, revese clockwise'''
    def __init__(self):
        # A = -90 
        # Rotation matrix = [
        #  cosA, -sinA 0
        #  sinA, cosA 0
        #  0     0    1
        # ]
        A = -np.pi/2
        self.R = np.array(
            [[0,1,0,0],
            [-1,0,0,0],
            [0,0,1,0],
            [0,0,0,1]]
        )
        self.invR = np.array([
            [0,-1,0,0],
            [1,0,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])
        # -np.pi/2

    def __call__(self, results):
        # breakpoint()
        box = results['gt_bboxes_3d'].tensor
        box[:,[0,1]] = box[:,[1,0]]
        box[:,1] = -box[:,1]   
        box[:,-1] = box[:,-1] - np.pi/2
        results['gt_bboxes_3d'].limit_yaw
        l2c = np.array(results['lidar2cam'])
        l2c = l2c @ self.invR
        results['lidar2cam'] = l2c
        results['ego_old2new'] = self.R
        # labels: x, y switch, old_yaw+new_yaw = pi/4 *2
        # ego2cam:
        return results
    # TODO: parse ego_old2new to img_meta so that visualizer can deal with it

@TRANSFORMS.register_module()
class ProjectLabelToWaymoClass(object):
    NUSC_NAME_MAP = {
            'car': 'Car',
            'truck': 'Car',
            'construction_vehicle': 'Car',
            'bus': 'Car',
            'trailer': 'Car',
            'motorcycle': 'Car',
            'bicycle': 'Cyclist',
            'pedestrian': 'Pedestrian'}

    def __init__(self,
                 class_names=None,
                 waymo_name=['Car', 'Pedestrian', 'Cyclist'],
                 name_map=NUSC_NAME_MAP):
        self.class_names = class_names
        self.waymo_name = waymo_name
        self.name_map = name_map
        ind_N2W = []
        for i, n_name in enumerate(self.class_names):
            if n_name in self.name_map:
                w_ind = waymo_name.index(self.name_map[n_name])
            else:
                w_ind = -1
            ind_N2W.append(w_ind)
        self.ind_N2W = np.array(ind_N2W)
    # class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    #                'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    # nusc2waymo = [0, 0, 0, 0, 0, -1, 0, 2, 1, -1]

    def __call__(self, results):
        labels = results['gt_labels_3d']
        if len(labels) > 0:
            labels = self.ind_N2W[labels]
            mask = (labels != -1)
            results['gt_labels_3d'] = labels[mask]
            results['gt_bboxes_3d'] = results['gt_bboxes_3d'][mask]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class Argo2LoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):

    def __init__(self, flip_front_cam = True, **kwargs):
        self.flip_front_cam = flip_front_cam
        super().__init__(**kwargs)

    def transform(self, results: dict):
        if self.num_ref_frames > 0:
           assert NotImplementedError

        filename, cam2img, lidar2cam = [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            cam2img.append(cam_item['cam2img'])
            lidar2cam.append(cam_item['lidar2cam'])
        results['filename'] = filename
        results['cam2img'] = cam2img
        results['lidar2cam'] = lidar2cam

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [self.file_client.get(name) for name in filename]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # Argo2 specific--->
        if self.flip_front_cam:
            # flip (x,y) to (y,x)
            imgs[0] = imgs[0].transpose(1,0,2)
            results['flip'] = [(i==0) for i in range(len(imgs))]
            results['cam2img'][0][[0,1]] = results['cam2img'][0][[1,0]]
        # <--- Argo2 specific

        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results

from typing import Any, Dict, Union

from mmengine.registry import MODEL_WRAPPERS
from mmengine.model.wrappers import MMDistributedDataParallel
@MODEL_WRAPPERS.register_module()
class CustomMMDDP(MMDistributedDataParallel):
    # for argo2 flip front cam image in feature extraction in backbone
    def __init__(self,
                 module,
                 static_graph = True,
                 **kwargs):
        super().__init__(module=module, **kwargs)
        self.static_graph = static_graph
        if static_graph:
            self._set_static_graph()