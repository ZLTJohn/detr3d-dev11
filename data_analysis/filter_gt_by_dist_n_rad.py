import pandas
import os.path as osp
import os
import shutil
import mmengine
import torch
import numpy as np
np.linalg.norm()
def filter_gt(pkl, range_dist, range_theta, range_x_size,range_z):
    info = pandas.read_pickle(pkl)['data_list']
    for i,item in enumerate(info):
        for inst in item['instances']:
            if 'waymo' in pkl:
                z = inst['bbox_3d'][1]-0.5*inst['bbox_3d'][4]
                x = inst['bbox_3d'][0]
                y = inst['bbox_3d'][2]
            else:
                x,y,z = inst['bbox_3d'][:3]
            x_size = inst['bbox_3d'][3]
            dist = (x**2+y**2+z**2)**0.5
            if 'nus' in pkl:
                x,y = y,x
            theta = np.arctan2(y,x)
            if z<range_z[0] or z>range_z[1]:
                continue
            if inst['num_lidar_pts']<0:
                continue
            if x_size<range_x_size[0] or x_size>range_x_size[1]:
                continue
            if dist<range_dist[0] or dist>range_dist[1]:
                continue
            if theta<range_theta[0] or theta>range_theta[1]:
                continue
            print(dist, inst['bbox_3d'], item['images']['CAM_FRONT']['img_path'], i)
            # return item

            
pkl = 'data/nus_v2/nuscenes_infos_val_part.pkl'
print(filter_gt(pkl,[29,31], [-0.5,0.5], [3.0,5.0],[-3,3]))
print('---------------------------')
pkl = 'data/waymo_dev1x/kitti_format/waymo_infos_val_2Hz_part.pkl'
print(filter_gt(pkl,[29.5,31.5], [-0.5,0.5], [3.0,5.0],[-3,3]))
# waymo更前一点，x更大一点
# [2.8936e+01,  3.9331e+00, -5.4543e-03,  5.0000e+00,  2.2000e+00, 1.8000e+00, -7.9632e-04]
# 1056194.jpg 751


# 29.267973311487864 [5.110840951315745, 28.81608330903527, 0.35624325826429987, 4.828, 1.952, 1.774, 1.5283746918192567] 
# n008-2018-09-18-15-12-01-0400__CAM_FRONT__1537298101162404.jpg 1438
# [ 5.1108, 28.8161, -0.5308,  4.8280,  1.9520,  1.7740,  1.5284]