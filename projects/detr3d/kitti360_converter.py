import numpy as np
import torch
import os
import os.path as osp
import mmengine
from glob import glob
from kitti360scripts.helpers import data, annotation, project
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid
from pyquaternion import Quaternion
# from kitti360scripts.devkits.convertOxtsPose.python.data import loadTimestamps
CLASSES = ['bicycle', 'box', 'bridge', 'building', 'bus', 'car',
           'caravan', 'garage', 'lamp', 'motorcycle', 'person', 
           'pole', 'rider', 'smallpole', 'stop', 'traffic light', 
           'traffic sign', 'trailer', 'train', 'trash bin', 'truck', 
           'tunnel', 'unknown construction', 'unknown object', 
           'unknown vehicle', 'vending machine']
SELECTED_CLASS = ['bicycle', 'bus', 'car',
           'caravan', 'motorcycle', 'person', 
           'rider', 'trailer', 'train', 'truck', 
           'unknown vehicle']   # static objects are too many, about 2/3 of objects
categories = {}
for i, name in enumerate(CLASSES):
    categories[name] = i
METAINFO = {
    'categories': categories,
    'dataset': 'kitti-360',
    'info_version': '1.1',
    'version': 'v0.0-trainval'}
class CustomCameraPerspective(project.CameraPerspective):
    def projectCenter(self, obj3d, frameId):
        vertices = obj3d.vertices.copy()
        vertices = vertices.mean(0,keepdims=True)
        uv, depth = self.project_vertices(vertices, frameId)
        obj3d.vertices_proj = uv
        obj3d.vertices_depth = depth 

# TODO: filter out occluded objects via tracing lidar ray to bbox points
class kitti360Dataloader(object):
    def __init__(self, data_root, gt_in_cam_only = True, out_dir = None):
        self.data_root = data_root
        self.gt_in_cam_only = gt_in_cam_only
        if out_dir is None:
            out_dir = data_root
        self.out_dir = data_root
        self.scenes = os.listdir(self.data_root+'/data_2d_raw')
        self.loaders = []
        for scene in self.scenes:
            Loader = kitti360Dataloader_OneScene(self.data_root, scene, gt_in_cam_only = self.gt_in_cam_only)
            self.loaders.append(Loader)

    def get_classes(self):
        self.class_names = {}
        for Loader in self.loaders:
            self.class_names.update(Loader.class_names)
        self.class_names = sorted(list(self.class_names.keys()))
        return self.class_names

    def make_data_list(self):
        self.data_list = []
        cnt=0
        for Loader in self.loaders:
            Loader.project_gt_to_cam0()
            Loader.make_data_list()
            Loader.save()
            for i in Loader.data_list:
                i['sample_idx'] = cnt
                cnt += 1
                self.data_list.append(i)

    def save(self):
        infos_all = {
            'data_list': self.data_list,
            'metainfo': METAINFO,
            }
        out_name = osp.join(self.out_dir, 'kitti360_infos_all.pkl')
        mmengine.dump(infos_all, out_name)
        print('save to',out_name)

class kitti360Dataloader_OneScene(object):
    def __init__(self, data_root, scene, out_dir = None, cam_ids = [0,1], gt_in_cam_only = True, calc_num_lidar_pts=True):# cam_ids must be ascending, must have '0'
        self.kitti360Path = data_root
        self.data_root = data_root
        self.gt_in_cam_only = gt_in_cam_only
        if out_dir is None:
            out_dir = data_root
        self.out_dir = data_root
        self.sequence = scene
        self.cam_ids = cam_ids # no fisheye now
        self.cameras = []
        self.class_names = {}
        self.calc_num_lidar_pts = calc_num_lidar_pts
        for cam_id in cam_ids:
            self.cameras.append(CustomCameraPerspective(self.kitti360Path, 
                                                          self.sequence, cam_id))
        # self.kitti360_mono = data.KITTI360(data_root,scene)
        self.label3DBboxPath = osp.join(self.kitti360Path, 'data_3d_bboxes')
        self.annotation3D = annotation.Annotation3D(self.label3DBboxPath, self.sequence)
        self.object_per_frame = self.parse_anno_to_frame()
        self.load_lidar()
        self.load_images(cam_ids)

    def loadTimestamps(self, ts_path):
        with open(os.path.join(ts_path, 'timestamps.txt')) as f:
            data=f.read().splitlines()
        ts = [l.replace(' ','_') for l in data]
        return ts

    def load_lidar(self):
        '''load velodyne as lidar'''
        # calibration/calib_cam_to_velo.txt
        self.cam0_to_velo = loadCalibrationRigid(osp.join(self.kitti360Path, 
                                'calibration/calib_cam_to_velo.txt'))
        # self.cam0_to_velo[:,[1,2]] = self.cam0_to_velo[:,[2,1]]
        self.cam0_to_pose = self.cameras[0].camToPose
        # pt_cam0 = pose_to_cam0 @ cami_to_pose @ pt_cami
        self.cam_to_cam0 = [np.linalg.inv(self.cam0_to_pose) @ i.camToPose
                                for i in self.cameras]
        # cam_to_velo = cam0_to_velo @cam_to_cam0
        self.cam2lidar = [
            self.cam0_to_velo @ cami_to_cam0
                for cami_to_cam0 in self.cam_to_cam0
        ]
        # velo2pose = cam0_to_pose @ velo_to_cam0
        self.lidar2ego =self.cam0_to_pose @ np.linalg.inv(self.cam0_to_velo)
        
        self.lidar_dir = osp.join(self.kitti360Path,'data_3d_raw',
                                  self.sequence,'velodyne_points')
        # May miss out some files, but no timestamps are missed
        self.lidar_timestamps = self.loadTimestamps(self.lidar_dir)
        self.lidar_dir = osp.join(self.lidar_dir,'data')
        lidar_files = sorted(os.listdir(self.lidar_dir))
        self.lidar_files = {}
        for file in lidar_files:
            self.lidar_files[int(file.split('.')[0])] = file
        print('there are {} pointclouds'.format(len(self.lidar_files)))

    def load_images(self,cam_ids):
        self.image_dirs = [osp.join(self.kitti360Path,'data_2d_raw',self.sequence,
                                    "image_0{}".format(id)) for id in cam_ids]
        self.image_timestamps = [self.loadTimestamps(i) for i in self.image_dirs]
        self.image_dirs = [osp.join(i,'data_rect') for i in self.image_dirs]
        image_files = [os.listdir(i) for i in self.image_dirs]
        image_files = [sorted(i) for i in image_files]
        self.image_files = []
        for files in image_files:
            files_dict = {}
            for file in files:
                files_dict[int(file.split('.')[0])] = file
            self.image_files.append(files_dict)
            print('there are {} images'.format(len(files_dict)))
        
    def parse_anno_to_frame(self):
        object_per_frame = { i:[] for i in self.cameras[0].frames }
        # I don't know what globalID stand for, but roughly the same shit
        # TODO: check if duplicate gt for -1
        for globalID in self.annotation3D.objects:
            objs = self.annotation3D.objects[globalID]
            if -1 in objs:
                if len(objs) !=1:
                    breakpoint()
                obj = objs[-1]
                self.class_names[obj.name] = 1
                L,R = obj.start_frame, obj.end_frame
                for frameID in range(L,R+1):
                    if frameID in object_per_frame:
                        object_per_frame[frameID].append(
                            {'object': obj,
                            'globalID': globalID})
            else:
                for frameID in objs:
                    if frameID in object_per_frame:
                        obj = objs[frameID]
                        self.class_names[obj.name] = 1
                        object_per_frame[frameID].append(
                            {'object': obj,
                            'globalID': globalID})
        return object_per_frame

    def project_gt_to_cam0(self):
        print('\nprojecting 3D gt boxes to each frame of scene', self.sequence)
        prog_bar = mmengine.ProgressBar(len(self.object_per_frame))
        cam0 = self.cameras[0]
        for frameID in self.object_per_frame:
            objs = self.object_per_frame[frameID]
            for obj_dict in objs:
                obj = obj_dict['object']
                vertices_world = obj.vertices
                # center_world = obj.T.reshape(1,3)
                # obj2world_R = obj.R # this is not SE3 matrix
                scale_xyz = np.linalg.norm(obj.R,axis=0)
                scale_matrix = np.diag(1/scale_xyz)
                obj2world_R = obj.R @ scale_matrix
                curr_pose = cam0.cam2world[frameID]
                T = curr_pose[:3,  3]
                R = curr_pose[:3, :3]

                # convert points from world coordinate to local coordinate 
                # TODO: filter by 3d range first to speed up
                vertices_cam0 = cam0.world2cam(vertices_world, R, T, inverse=True)
                obj_dict['in_image'] = False
                for cam in self.cameras:
                    cam.projectCenter(obj, frameID)
                    u,v = obj.vertices_proj
                    if (0<u and u<cam.width) and (0<v and v<cam.height):
                        obj_dict['in_image'] = True
                        break
                # centers_cam0 = cam0.world2cam(center_world, R, T, inverse=True)
                obj2cam0_R = R.T @ obj2world_R
                obj_dict['obj2cam0_R'] = obj2cam0_R
                obj_dict['vertices_cam0'] = vertices_cam0
                obj_dict['scale_xyz'] = scale_xyz
                # obj2cam0_R.T @ points_cam0 = cam02obj_R @ points_cam0
            prog_bar.update()

    def make_data_list(self):
        print('\nmaking MMDet3D data_list of scene', self.sequence)
        self.data_list = []
        prog_bar = mmengine.ProgressBar(len(self.object_per_frame))
        i=0
        for frameID in self.object_per_frame:
            ID = int(frameID)
            if ID not in self.image_files[0]:
                # print('skipping frame',ID,'since no image data')
                prog_bar.update()
                continue

            sample_idx = i
            i+=1
            log_id = self.sequence
            timestamp = frameID
            # ego2global = 
            cam0_to_global = self.cameras[0].cam2world[frameID]
            images = {}
            for idx,(cam_id,cam) in enumerate(zip(self.cam_ids,self.cameras)):
                image = {}
                img_path = osp.join(self.image_dirs[idx], self.image_files[idx][ID])
                image['img_path'] = osp.relpath(img_path, self.data_root)
                image['height'] = cam.height
                image['width'] = cam.width
                image['cam2img'] = cam.K[:3,:3]
                image['lidar2cam'] = np.linalg.inv(self.cam2lidar[idx])
                image['timestamp'] = self.image_timestamps[idx][ID]
                images['CAM{}'.format(cam_id)] = image

            lidar_points = {}
            lidar_points['num_pts_feats'] = 4
            lidar_path = osp.join(self.lidar_dir, self.lidar_files[ID])
            lidar_points['lidar_path'] = osp.relpath(lidar_path, self.data_root)
            lidar_points['lidar2ego'] = self.lidar2ego
            lidar_points['timestamp'] = self.lidar_timestamps[ID]

            if self.calc_num_lidar_pts:
                lidar_pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,4)
            
            instances = []
            objs = self.object_per_frame[frameID]
            cam0 = self.cameras[0]
            objid=0
            for obj_dict in objs:
                obj = obj_dict['object']
                if obj.name not in SELECTED_CLASS:
                    continue
                if self.gt_in_cam_only and obj_dict['in_image'] == False:
                    continue
                vertices_cam0 = obj_dict['vertices_cam0']
                obj2cam0_R = obj_dict['obj2cam0_R']
                cam0_to_velo_T = self.cam0_to_velo[:3,3:4]
                cam0_to_velo_R = self.cam0_to_velo[:3,:3]
                vertices_lidar = cam0_to_velo_R @ vertices_cam0 + cam0_to_velo_T
                obj2velo_R = cam0_to_velo_R @ obj2cam0_R # @ pt_obj
                vertices_box = obj2velo_R.T @ vertices_lidar

                x,y,z = vertices_lidar.mean(1)
                L,W,H = obj_dict['scale_xyz']
                if np.linalg.det(obj2velo_R) > 0:
                    rot = Quaternion(matrix=obj2velo_R,atol=1e-6) 
                    yaw, pitch, roll = rot.yaw_pitch_roll
                else:
                    R = obj2velo_R
                    yaw = np.arctan2(R[1,0], R[0,0])
                    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
                    roll = np.arctan2(R[2,1], R[2,2])
                    print('\n invalid Rot matrix:',obj.name, yaw, pitch, roll)

                bbox_label_3d = CLASSES.index(obj.name)
                # if ID == 115:
                #     breakpoint()
                if self.calc_num_lidar_pts:
                    from av2.geometry.geometry import compute_interior_points_mask
                    Perm = [0,2,3,1,5,7,6,4]    # kitti-360 format to av2 format
                    vertices_lidar_av2_format = vertices_lidar.T[Perm]
                    is_interior = compute_interior_points_mask(lidar_pts[:,:3], vertices_lidar_av2_format)
                    box_pts = lidar_pts[is_interior]
                    num_lidar_pts = box_pts.shape[0]
                else:
                    num_lidar_pts = -1  # temporarily not available
                instances.append({
                    'bbox_3d': [x, y, z, L, W, H, yaw],
                    'global_id': obj_dict['globalID'],
                    'pitch_roll': [pitch,roll],
                    'bbox_label_3d': bbox_label_3d,
                    'num_lidar_pts': num_lidar_pts
                })
                objid+=1

            if len(instances) == 0:
                # print('skip frame {} since no gt'.format(frameID))
                prog_bar.update()
                continue
            info = {
                'sample_idx': sample_idx,
                'log_id': log_id,
                # 'city_name': city_name,
                'timestamp': timestamp,
                'cam0_to_global': cam0_to_global,
                'images': images,
                'lidar_points': lidar_points,
                'instances': instances
            }
            self.data_list.append(info)
            prog_bar.update()

    def save(self):
        infos_all = {
            'data_list': self.data_list,
            'metainfo': METAINFO,
            }
        out_name = osp.join(self.out_dir, 'kitti360_infos_{}.pkl'.format(self.sequence))
        mmengine.dump(infos_all, out_name)
        print('saved to', out_name)

if __name__ == '__main__':
    loader = kitti360Dataloader('data/kitti-360')
    loader.make_data_list()
    loader.save()
# breakpoint()
# scene 0003 has invalid Rot matrix: car -1.621960167967955 -0.019969754133386607 -0.027501106275214935 at frame 261

# info = {
#         'sample_idx': #sample_idx,
#         'log_id': #log_id,
#         'city_name': #loader.get_city_name(log_id),
#         'timestamp': #ts,
#         'ego2global': #ego2global,
#         'images': {
#             'img_path':
#             # str(img_path.relative_to(self.root_path)
#             #     ),  # add {split}/{log_id}/sensors/cameras/{cam_name}
#             'height': #intrinsics.height_px,
#             'width': #intrinsics.width_px,
#             'cam2img': #cam2img,
#             'lidar2cam': #lidar2cam,
#             'timestamp': #img_path.stem
#         },
#         'lidar_points': {
#         'num_pts_feats': #data.shape[1],
#         'lidar_path': #str(osp.relpath(save_path, self.root_path)),
#         'lidar2ego_down': # lidar2ego_down,
#         'lidar2ego_up': #lidar2ego_up,
#         'timestamp': #sweep.timestamp_ns
#     },
#         'instances': {
#                 # we leave bbox_3d coordinate unchange, and proceed it in class argo2_dataset
#                 'bbox_3d': #[x, y, z, l, w, h, yaw],
#                 'bbox_label_3d': # int
#                 'num_lidar_pts': # int
#             }
#     }