from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.loading import LoadMultiViewImageFromFiles, LoadPointsFromFile
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
import copy
import mmengine
from mmengine.fileio import get
import mmcv
import numpy as np
import os.path as osp
import random
import time
@TRANSFORMS.register_module()
class filename2img_path:

    def __call__(self, results):
        results['img_path'] = results['filename']
        return results

    def __repr__(self):
        return 'maybe we need to fix this bug'

@TRANSFORMS.register_module()
class Pack3DDetInputsExtra(Pack3DDetInputs):
    def __init__(self, **kwargs) -> None:
        extra_keys = [
            'dataset_name',
            'city_name',
            'timestamp',
            'token',
            'num_pts_feats',
            'ksync_factor',
            'lidar2ego',
            'img_flip'
        ]
        super().__init__(**kwargs)
        self.meta_keys = tuple(list(self.meta_keys)+extra_keys)
    def transform(self, results):
        lp = results.get('lidar_points',None)
        if lp is not None:
            results['lidar2ego'] = lp.get('lidar2ego',None)
        return super().transform(results)

@TRANSFORMS.register_module()
class evalann2ann:
    '''
    put eval_ann_info into ann_info, in this way, we can evaluate more easily
    also put location info into eval_ann_info
    '''
    def __call__(self, results):
        results['ann_info'] = results['eval_ann_info']
        return results

    def __repr__(self):
        return 'maybe we need to fix this bug'
from mmdet3d.datasets.transforms.transforms_3d import Resize3D, MultiViewWrapper
@TRANSFORMS.register_module()
class Ksync:
    def __init__(self, fx = -1, fy = -1, dont_resize = False):
        self.resize3d = MultiViewWrapper(transforms = [Resize3D(scale_factor = (1.0, 1.0))])
        self.dont_resize = dont_resize
        self.fx = fx
        self.fy = fy
    
    def __call__(self, results):
        K = results['cam2img'][0]
        flip = results.get('flip')
        if flip is not None and flip[0] is True:
            fx = K[0][1]
            fy = K[1][0]
        else:
            fx = K[0][0]
            fy = K[1][1]
        scalex = self.fx / fx
        scaley = self.fy / fy
        if self.fy==-1:
            scaley = scalex
        self.resize3d.transforms.transforms[0].scale_factor = (scalex, scaley)
        results['ksync_factor'] = (scalex, scaley)
        if self.dont_resize:
            return results
        else:
            return self.resize3d(results)


from mmcv.image.geometric import _scale_size
@TRANSFORMS.register_module()
class SyncFocalLength:
    '''
    Sync back camera focal length for NuScenes
    '''
    def __init__(self, specific_idx = -1):
        self.specific_idx = specific_idx
    def __call__(self, results):
        num_view = results['num_views']
        # get the back cam
        for idx in range(num_view):
            if results['cam2img'][idx][0][0] < results['cam2img'][0][0][0] - 100:
                break
        if self.specific_idx != -1:
            idx = self.specific_idx
        # scale_factor = results['cam2img'][0][0][0] / results['cam2img'][idx][0][0]
        scale_factor = 1.5649831353955652
        # resize
        img = mmcv.imrescale(results['img'][idx],scale_factor)
        new_h, new_w = img.shape[:2]
        H, W = results['img'][0].shape[:2]
        w_scale = new_w / W
        h_scale = new_h / H
        results['cam2img'][idx][0] *= np.array(w_scale)
        results['cam2img'][idx][1] *= np.array(h_scale)
        results['img'][idx] = img
        for i in range(num_view):
            if i != idx:
                img = mmcv.impad(results['img'][i], shape=(new_h,new_w))
                results['img'][i] = img
        results['pad_shape'] = (new_h,new_w)
        results['img_shape'] = (new_h,new_w)
        return results

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
        self.A = -np.pi/2
        self.R = np.array(
            [[0,1,0,0],
            [-1,0,0,0],
            [0,0,1,0],
            [0,0,0,1]]
        )
        self.new2old = np.linalg.inv(self.R)

    def __call__(self, results):
        # (R.T @ pt.T).T = pt @ R
        # we are left mult Rotation matrix, so we have to transpose it
        results['gt_bboxes_3d'].rotate((self.R[:3,:3]).T)
        results['lidar2cam'] = np.array(results['lidar2cam']) @ self.new2old # @ pt_new

        trans = results.get('trans_mat',np.identity(4))
        results['trans_mat'] = self.R @ trans # @ pt_origin
        # labels: x, y switch, old_yaw+new_yaw = pi/4 *2
        # ego2cam:
        return results
    # TODO: parse ego_old2new to img_meta so that visualizer can deal with it
    # parse it to eval ann info
@TRANSFORMS.register_module()
class ego_transform:
    '''
    Do ego transformation for NuScenes and Lyft
    '''
    def __init__(self, trans='lidar2ego'):
        self.T = trans
    def transformation(self,results, Tr):
        inv_Tr = np.linalg.inv(Tr)
        results['gt_bboxes_3d'].rotate(Tr[:3,:3].T) # LidarInstance.rotate use right-side matmul
        results['gt_bboxes_3d'].translate(Tr[:3,3])
        results['lidar2cam'] = np.array(results['lidar2cam']) @ inv_Tr # @ pt_new
        trans = results.get('trans_mat',np.identity(4))
        results['trans_mat'] = Tr @ trans # @ pt_origin
        return results

    def __call__(self, results):
        if type(self.T) is not str:
            Tr = np.array(self.T)
        else:
            Tr = np.array(results['lidar_points'][self.T])
        if self.T == 'Tr_imu_to_velo':
            Tr = np.linalg.inv(Tr)
        return self.transformation(results, Tr)
        
# Tr_K360=[1,-1,-1,1], too big change for K360 to able mmdet3d work properly
@TRANSFORMS.register_module()
class ego_transform_K360(ego_transform):
    def __call__(self, results):
        Tr = np.array(results['lidar_points'][self.T])
        rectify = np.diag([1,-1,-1,1.])
        Tr = rectify @ Tr
        return self.transformation(results, Tr)

@TRANSFORMS.register_module()
class EgoTranslate:
    # TODO: add info for evaluation in case of mis-configured
    def __init__(self, Tr_range = [0,0,0,0,0,0], eval = True, frame_wise = False, trans = None):
        self.Tr_range = Tr_range
        self.frame_wise = frame_wise
        self.eval= eval
        self.trans = trans
        random.seed(time.time())

    def get_scene_token(self, results):
        '''
            Examples:
                'data/argo2/val/02678d04-cc9f-3148-9f95-1ba66347dff9/sensors/cameras/ring_front_center/315969904449927217.jpg'
                'data/nus_v2/samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460.jpg'
                'data/waymo_dev1x/kitti_format/training/image_0/1000000.jpg'
        '''
        if self.frame_wise == False:
            path = results['img_path'][0]
            imgfile = osp.basename(path)
            if  'waymo' in path:
                # ABBBCCC for split, scene, frame
                scene_token = imgfile[:4]
            elif 'nus' in path:
                scene_token = imgfile[:29]# n015-2018-08-02-17-16-37+0800
            else:
                scene_token = results['log_id']
        else:
            scene_token = results['timestamp']
        return scene_token

    def __call__(self, results):
        # support nusc only
        if self.eval and self.trans == None:
            scene_token = self.get_scene_token(results)
            random.seed(scene_token)
        if self.trans is None:
            [X0,Y0,Z0,X1,Y1,Z1] = self.Tr_range
            x = random.random() * (X1 - X0) + X0#-0.15789807484957175
            y = random.random() * (Y1 - Y0) + Y0
            z = random.random() * (Z1 - Z0) + Z0
            t = np.array([x,y,z])
        elif self.trans == 'FrontCam':
            t = np.linalg.inv(results['lidar2cam'])[0][:3,3]
        else:
            t = np.array(self.trans)
        # ego += t, pts -= t
        new2old = np.identity(4)
        new2old[:3,3] = t
        old2new = np.identity(4)
        old2new[:3,3] = -t
        results['gt_bboxes_3d'].translate(-t)
        results['lidar2cam'] = np.array(results['lidar2cam']) @ new2old # @ pt_new
        trans = results.get('trans_mat',np.identity(4))
        results['trans_mat'] = old2new @ trans # @ pt_origin
        return results
    
@TRANSFORMS.register_module()
class HorizontalFlipAll(object):
    '''
    NuScenes only!
    '''
    def __init__(self, location = 'singapore', new2old=[0,2,1,3,5,4]):
        self.location = location
        self.R = np.identity(4)
        self.R[1,1] = -1
        # new2old is equal to old2new
        # We assert that img1/2 and img4/5 is front[left/right], back[left/right]
        self.imgPermute = PermuteImages(new2old=new2old)
    def __call__(self, results):
        if self.location not in results['city_name']:
            return results
        # step 1: flip lidar points, lidar2cam, gt box, add trans_mat
        results['gt_bboxes_3d'].flip('horizontal')
        results['lidar2cam'] = np.array(results['lidar2cam']) @ self.R
        results['trans_mat'] = self.R @ results.get('trans_mat',np.identity(4))
        # step 2: flip image, flip cam2img
        assert type(results['img']) == list
        imgs = results['img']
        c2i = np.array(results['cam2img'])
        for i in range(len(imgs)):
            imgs[i] = mmcv.imflip(imgs[i],direction='horizontal')
            # xfx' + zcx' = z(1600-u)
            # xfx  + zcx  = zu
            # ==> 1600 = x/z(fx'+fx) + cx' + cx
            # ==> fx' = -fx, cx' = 1600 - cx
            c2i[i][0,0] = -c2i[i][0,0]
            c2i[i][0,2] = imgs[i].shape[1] - c2i[i][0,2]
        results['cam2img'] = c2i
        results['img_flip'] = True
        # step 3: permute image order, but skip it if under mono detection
        if len(results['img']) > 1:
            results = self.imgPermute(results)
        return results

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

from mmdet3d.datasets.transforms.transforms_3d import BaseTransform
@TRANSFORMS.register_module()
class ObjectRangeFilter3D(BaseTransform):
    """Filter objects by the range.

    Required Keys:

    - gt_bboxes_3d

    Modified Keys:

    - gt_bboxes_3d

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict):
        """Transform function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
            keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_3d(self.pcd_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

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
            if 'img_path' not in cam_item: # for kitti-mono
                continue
            filename.append(cam_item['img_path'])
            cam2img.append(cam_item['cam2img'])
            lidar2cam.append(cam_item['lidar2cam'])
        results['filename'] = filename
        results['cam2img'] = cam2img
        results['lidar2cam'] = lidar2cam

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
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


from av2.structures.sweep import Sweep
from pathlib import Path
@TRANSFORMS.register_module()
class Argo2LoadPointsFromFile(LoadPointsFromFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.load_dim == 3 # not support except for xyz
    def _load_points(self, pts_filename: str) -> np.ndarray:
        sweep = Sweep.from_feather(Path(pts_filename))
        points = sweep.xyz
        return points
    
@TRANSFORMS.register_module()
class fillzeros:
    def __init__(self, hpre, hsuf) -> None:
        self.hpre = hpre
        self.hsuf = hsuf
    
    def __call__(self, results):
        for i in range(len(results['img'])):
            img = results['img'][i]
            img[:self.hpre] = 0
            img[self.hsuf:] = 0
            results['img'][i]
        return results


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