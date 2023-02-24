import pandas
import os.path as osp
import os
import shutil
import mmengine
def copy_or_exist(src,dst):
    suffix = osp.basename(src)
    if not osp.exists(osp.dirname(dst)):
        os.makedirs(osp.dirname(dst))
    if not osp.exists(dst):
        shutil.copy(src,dst)
    return 1

def move_one_split(new_path, data_root, pkl, data_prefix):
    print('moving {} to {}'.format(data_root+pkl, new_path))
    # new_path = '/localdata_ssd/debug/'
    info = pandas.read_pickle(osp.join(data_root,pkl))
    copy_or_exist(osp.join(data_root,pkl), osp.join(new_path,pkl))
    bar = mmengine.ProgressBar(len(info['data_list']))
    for i,item in enumerate(info['data_list']):
        for img in item['images']:
            suffix = item['images'][img]['img_path']
            src = osp.join(data_root,data_prefix.get(img,''),suffix)
            dst = osp.join(new_path,data_prefix.get(img,''),suffix)
            copy_or_exist(src,dst)
        lidar = item['lidar_points']
        suffix = lidar['lidar_path']
        src = osp.join(data_root,data_prefix.get('pts',''),suffix)
        dst = osp.join(new_path,data_prefix.get('pts',''),suffix)
        copy_or_exist(src,dst)
        bar.update()

argo2_data_prefix = dict()
nusc_data_prefix = dict(pts='samples/LIDAR_TOP',
                   sweeps='sweeps/LIDAR_TOP',
                   CAM_FRONT='samples/CAM_FRONT',
                   CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                   CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                   CAM_BACK='samples/CAM_BACK',
                   CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
                   CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',)

waymo_data_prefix = dict(
    pts='training/velodyne',
    sweeps='training/velodyne',
    CAM_FRONT='training/image_0',
    CAM_FRONT_LEFT='training/image_1',
    CAM_FRONT_RIGHT='training/image_2',
    CAM_SIDE_LEFT='training/image_3',
    CAM_SIDE_RIGHT='training/image_4',)

argo2_type = 'Argo2Dataset'
argo2_data_root = 'data/argo2/'
argo2_train_pkl = 'argo2_infos_train_2Hz.pkl'  
argo2_train_interval = 1    # 2Hz_part means interval = 5x3
argo2_val_pkl = 'argo2_infos_val_2Hz.pkl'
argo2_val_interval = 1

nusc_type = 'CustomNusc'
nusc_data_root = 'data/nus_v2/'
nusc_train_pkl = 'nuscenes_infos_train.pkl' 
nusc_train_interval = 1
nusc_val_pkl = 'nuscenes_infos_val.pkl'
nusc_val_interval = 1

waymo_type = 'WaymoDataset'
waymo_data_root = 'data/waymo_dev1x/kitti_format/'
waymo_train_pkl = 'waymo_infos_train_2Hz.pkl'
waymo_train_interval = 1    # 2Hz_part means interval = 5x3
waymo_val_pkl = 'waymo_infos_val_2Hz.pkl'
waymo_val_interval = 1

argo2_ssd_path = '/localdata_ssd/argo2_dev1x/'
nusc_ssd_path = '/localdata_ssd/nusc_dev1x/'
waymo_ssd_path = '/localdata_ssd/waymo_dev1x/'

move_one_split(argo2_ssd_path, argo2_data_root, argo2_train_pkl, argo2_data_prefix)
move_one_split(argo2_ssd_path, argo2_data_root, argo2_val_pkl, argo2_data_prefix)

move_one_split(nusc_ssd_path, nusc_data_root, nusc_train_pkl, nusc_data_prefix)
move_one_split(nusc_ssd_path, nusc_data_root, nusc_val_pkl, nusc_data_prefix)

move_one_split(waymo_ssd_path, waymo_data_root, waymo_train_pkl, waymo_data_prefix)
move_one_split(waymo_ssd_path, waymo_data_root, waymo_val_pkl, waymo_data_prefix)

# load images:  0.09715604782104492
# finish loss 0.1034231185913086
# whole iter 0.29329466819763184
# load images:  0.1042027473449707
# finish loss 0.10422778129577637
# whole iter 0.29163479804992676
# load images:  0.09953713417053223
# finish loss 0.1049656867980957
# whole iter 0.2934000492095947
# load images:  0.10286903381347656
# finish loss 0.1018226146697998
# whole iter 0.2909352779388428
# finish loss 0.10185575485229492
# whole iter 0.2820737361907959
# load images:  0.11583065986633301
# load images:  0.10680437088012695
# finish loss 0.10410261154174805
# whole iter 0.29427194595336914
# load images:  0.10814452171325684
# finish loss 0.10738515853881836
# whole iter 0.2980465888977051
# finish loss 0.10192108154296875
# whole iter 0.29662442207336426
# load images:  0.12159967422485352
# load images:  0.09928703308105469
# finish loss 0.10420012474060059
# whole iter 0.2938556671142578
# load images:  0.09821724891662598
# finish loss 0.10322737693786621
# whole iter 0.29383373260498047
# load images:  0.10012340545654297
# finish loss 0.10582160949707031
# whole iter 0.29642200469970703
# finish loss 0.1026158332824707
# whole iter 0.2915918827056885
# load images:  0.1121988296508789
# finish loss 0.10629963874816895
# whole iter 0.2953028678894043
# load images:  0.11156582832336426
# [1][ 150/2007]  lr: 1.0648e-04  eta: 4:08:21  time: 0.2922  data_time: 0.0060  memory: 3937  grad_norm: 75.7118  loss: 16.6331  loss_cls: 1.1791  loss_bbox: 1.5984  d0.loss_cls: 1.1676  d0.loss_bbox: 1.6001  d1.loss_cls: 1.1480  d1.loss_bbox: 1.5722  d2.loss_cls: 1.1989  d2.loss_bbox: 1.5606  d3.loss_cls: 1.2354  d3.loss_bbox: 1.5940  d4.loss_cls: 1.2249  d4.loss_bbox: 1.5539
# 02/16 16:12:07 - mmengine - INFO - Epoch(train)  [1][100/502]  lr: 9.3120e-05  eta: 1:05:51  time: 0.3009  data_time: 0.0062  memory: 3934  grad_norm: 32.4890  loss: 14.4074  loss_cls: 0.9644  loss_bbox: 1.4462  d0.loss_cls: 0.9508  d0.loss_bbox: 1.4579  d1.loss_cls: 0.9655  d1.loss_bbox: 1.4637  d2.loss_cls: 0.9529  d2.loss_bbox: 1.4467  d3.loss_cls: 0.9498  d3.loss_bbox: 1.4337  d4.loss_cls: 0.9474  d4.loss_bbox: 1.4283

# finish loss 0.10716557502746582
# whole iter 0.29162096977233887
# load images:  0.29627060890197754
# finish loss 0.10903787612915039
# whole iter 0.30219149589538574
# load images:  0.23152422904968262
# finish loss 0.10386085510253906
# whole iter 0.29495716094970703
# load images:  0.24914884567260742
# finish loss 0.10822081565856934
# whole iter 0.2994537353515625
# load images:  0.26613402366638184
# finish loss 0.10783863067626953
# whole iter 0.2992818355560303
# load images:  0.3121073246002197
# finish loss 0.10354828834533691
# whole iter 0.29583311080932617
# load images:  0.28586244583129883
# finish loss 0.10687947273254395
# whole iter 0.29989123344421387
# load images:  0.31377434730529785
# finish loss 0.11006498336791992
# whole iter 0.30021190643310547
# load images:  0.27286839485168457
# finish loss 0.11147809028625488
# whole iter 0.30165767669677734
# load images:  0.26531553268432617
# finish loss 0.10762739181518555
# whole iter 0.2904167175292969
# load images:  0.254058837890625
# finish loss 0.10593366622924805
# whole iter 0.2973930835723877
# load images:  0.23450803756713867
# finish loss 0.10194611549377441
# whole iter 0.2925877571105957
# load images:  0.24721908569335938
# finish loss 0.10258841514587402
# whole iter 0.29495835304260254
# load images:  0.30864453315734863
# finish loss 0.10826683044433594
# whole iter 0.29880261421203613
# load images:  0.26010894775390625
# finish loss 0.10658836364746094
# whole iter 0.2949695587158203
# 02/16 16:07:07 - mmengine - INFO - Epoch(train)  [1][ 100/9377]  lr: 9.3120e-05  eta: 20:14:56  time: 0.2946  data_time: 0.0059  memory: 3937  grad_norm: 72.0072  loss: 17.1613  loss_cls: 1.2365  loss_bbox: 1.6171  d0.loss_cls: 1.2305  d0.loss_bbox: 1.6571  d1.loss_cls: 1.2415  d1.loss_bbox: 1.5901  d2.loss_cls: 1.2337  d2.loss_bbox: 1.6702  d3.loss_cls: 1.2353  d3.loss_bbox: 1.6304  d4.loss_cls: 1.2376  d4.loss_bbox: 1.5812
# mmengine - INFO - Epoch(train)  [1][ 100/2345]  lr: 9.3120e-05  eta: 7:17:02  time: 0.4328  data_time: 0.0066  memory: 3940  grad_norm: 31.1313  loss: 17.4731  loss_cls: 1.1049  loss_bbox: 1.8268  d0.loss_cls: 1.0948  d0.loss_bbox: 1.8195  d1.loss_cls: 1.1235  d1.loss_bbox: 1.7639  d2.loss_cls: 1.1374  d2.loss_bbox: 1.7759  d3.loss_cls: 1.1050  d3.loss_bbox: 1.8091  d4.loss_cls: 1.1201  d4.loss_bbox: 1.7923