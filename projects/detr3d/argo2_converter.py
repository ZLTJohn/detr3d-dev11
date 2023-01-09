# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp
from pathlib import Path
from time import time

import mmengine
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.evaluation.detection.constants import CompetitionCategories
from av2.evaluation.detection.utils import DetectionCfg
from av2.structures.sweep import Sweep
from pyquaternion import Quaternion

CLASSES = [x.value for x in CompetitionCategories]
CAM_NAMES = [
    'ring_front_center',
    'ring_front_right',
    'ring_front_left',
    'ring_rear_right',
    'ring_rear_left',
    'ring_side_right',
    'ring_side_left',
    # 'stereo_front_left', 'stereo_front_right',
]

# FAIL_LOGS = [
#     '01bb304d-7bd8-35f8-bbef-7086b688e35e',
#     '453e5558-6363-38e3-bf9b-42b5ba0a6f1d'
# ]


# TODO: filter out num_lidar_pts too small
class argo2_info_gatherer:

    def __init__(self, root_path, split, out_dir, prefix):
        self.root_path = root_path
        root_path = osp.join(root_path, split)
        _ = time()
        self.loader = AV2SensorDataLoader(Path(root_path), Path(root_path))
        print('initialize argo2 dataloader time:', time() - _)
        self.log_ids = self.loader.get_log_ids()
        self.out_dir = out_dir
        self.out_name = f'argo2_infos_{split}.pkl'
        # self.out_name = f'debug_{split}.pkl'
        self.split = split
        self.prefix = prefix
        self.use_lidar = True
        # self.lidar_bin_dir = osp.join(out_dir, 'lidar_bins/')
        # mmengine.mkdir_or_exist(self.lidar_bin_dir)

    def gather_all(self):
        print('\nConverting {} split...'.format(self.split))
        self.prog_bar = mmengine.ProgressBar(len(self.log_ids))
        infos_all = []
        for i in range(len(self.log_ids)):
            infos_one_scene = self.gather_single(i)
            infos_all.extend(infos_one_scene)
            self.prog_bar.update()
            # break
        print('\nsaving {} to {}'.format(self.out_name, self.out_dir))
        categories = {}
        for i, name in enumerate(CLASSES):
            categories[name] = i
        infos_all = {
            'data_list': infos_all,
            'metainfo': {
                'categories': categories,
                'dataset': 'argoverse2',
                'info_version': '1.1'
            }
        }
        mmengine.dump(infos_all, osp.join(self.out_dir, self.out_name))

    def gather_single(self, scene_idx):
        infos_one_scene = []
        log_id = self.log_ids[scene_idx]
        loader = self.loader
        # lidar timestamps, annotation provided
        timestamps = loader.get_ordered_log_lidar_timestamps(log_id)
        # prog_bar1 = mmengine.ProgressBar(len(timestamps))
        for frame_id, ts in enumerate(timestamps):
            # sample_idx
            sample_idx = f'{self.prefix}{scene_idx:03d}{frame_id:03d}'
            # ego2global
            city_SE3_ego = loader.get_city_SE3_ego(log_id, ts)
            ego2global = city_SE3_ego.transform_matrix
            # images
            images = {}
            for cam in CAM_NAMES:
                # img_path
                img_path = loader.get_closest_img_fpath(log_id, cam, ts)
                if img_path == None:
                    images = None
                    break
                pinhole_cam = loader.get_log_pinhole_camera(log_id, cam)
                intrinsics = pinhole_cam.intrinsics
                extrinsics = pinhole_cam.extrinsics

                # get cam2img according to intrinsics
                cam2img = np.eye(4, dtype=float)
                cam2img[:3, :3] = intrinsics.K
                # lidar2cam is ego2cam since lidar points are given in ego frame
                lidar2cam = extrinsics
                # we don't save lidar2img anymore
                images[cam] = {
                    'img_path':
                    str(img_path.relative_to(self.root_path)
                        ),  # add {split}/{log_id}/sensors/cameras/{cam_name}
                    'height': intrinsics.height_px,
                    'width': intrinsics.width_px,
                    'cam2img': cam2img,
                    'lidar2cam': lidar2cam,
                    'timestamp': img_path.stem
                }

            if images == None:
                print('Not complete camera data at', ts)
                continue

            # lidar_points:
            lidar_path = loader.get_lidar_fpath_at_lidar_timestamp(log_id, ts)
            # save lidar
            sweep = Sweep.from_feather(lidar_path)
            data = np.column_stack((sweep.xyz, sweep.intensity,
                                    sweep.laser_number, sweep.offset_ns))

            save_path = lidar_path
            # save_path = Path(osp.join(self.lidar_bin_dir, f'{sample_idx}.bin'))
            # data.astype(np.float32).tofile(save_path)

            lidar2ego_down = sweep.ego_SE3_down_lidar.transform_matrix
            lidar2ego_up = sweep.ego_SE3_up_lidar.transform_matrix
            lidar_info = {
                'num_pts_feats': data.shape[1],
                'lidar_path': str(osp.relpath(save_path, self.root_path)),
                'lidar2ego_down': lidar2ego_down,
                'lidar2ego_up': lidar2ego_up,
                'timestamp': sweep.timestamp_ns
            }

            # annotations
            instances = []
            if self.split != 'test':
                labels = loader.get_labels_at_lidar_timestamp(log_id, ts)
                for label in labels:
                    # some catergory not interested
                    if not label.category in CLASSES:
                        continue
                    # all in egovehicle frame coordinate
                    (x, y, z) = label.xyz_center_m
                    (l, w, h) = label.dims_lwh_m  # mmdet3d needs lhw
                    rot = Quaternion(matrix=label.dst_SE3_object.rotation)
                    yaw = rot.yaw_pitch_roll[0]
                    instance = {
                        # we leave bbox_3d coordinate unchange, and proceed it in class argo2_dataset
                        'bbox_3d': [x, y, z, l, w, h, yaw],
                        'bbox_label_3d':
                        CLASSES.index(label.category),
                        # TODO: use existing info in *.feather to speedup
                        'num_lidar_pts':
                        label.compute_interior_points(sweep.xyz)[0].shape[0]
                    }
                    instances.append(instance)

            # TODO: cam_instances
            # cam_instances = {}

            info = {
                'sample_idx': sample_idx,
                'log_id': log_id,
                'city_name': loader.get_city_name(log_id),
                'timestamp': ts,
                'ego2global': ego2global,
                'images': images,
                'lidar_points': lidar_info,
                'instances': instances
            }
            infos_one_scene.append(info)
            # prog_bar1.update()

        return infos_one_scene


def argo2_data_prep(root_path, out_dir):
    splits = ['train', 'val', 'test']
    for i, split in enumerate(splits):
        gatherer = argo2_info_gatherer(root_path, split, out_dir, i)
        gatherer.gather_all()


if __name__ == '__main__':
    # from mmdet3d.utils import register_all_modules
    # register_all_modules()
    # # Set to spawn mode to avoid stuck when process dataset creating
    # import multiprocessing
    # multiprocessing.set_start_method('spawn')
    root_path = 'data/argo2/'
    out_dir = 'data/argo2/'
    argo2_data_prep(root_path=root_path, out_dir=out_dir)
