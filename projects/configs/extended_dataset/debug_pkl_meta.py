argo2_type = 'Argo2Dataset'
argo2_data_root = 'data/argo2/'
argo2_train_pkl = 'debug_val_mono_front.pkl'  
argo2_train_interval = 4    # 2Hz_part means interval = 5x3
argo2_val_pkl = 'debug_val_mono_front.pkl'
argo2_val_interval = 4

nusc_type = 'CustomNusc'
nusc_data_root = 'data/nus_v2/'
nusc_train_pkl = 'debug_val_mono_front.pkl' 
nusc_train_interval = 1
nusc_val_pkl = 'debug_val_mono_front.pkl'
nusc_val_interval = 1

waymo_type = 'CustomWaymo'
waymo_data_root = 'data/waymo_dev1x/kitti_format'
waymo_train_pkl = 'debug_val_mono_front.pkl'
waymo_train_interval = 9    # 2Hz_part means interval = 5x3
waymo_val_pkl = 'debug_val_mono_front.pkl'
waymo_val_interval = 9

lyft_type = 'CustomLyft'
lyft_data_root = 'data/lyft/'
lyft_train_pkl = 'debug_val_mono_front.pkl' 
lyft_train_interval = 4
lyft_val_pkl = 'debug_val_mono_front.pkl'
lyft_val_interval = 4

kitti_type = 'CustomKitti'
kitti_data_root = 'data/kitti/'
kitti_train_pkl = 'kitti_infos_train.pkl'
kitti_train_interval = 100
kitti_val_pkl = 'kitti_infos_val.pkl'
kitti_val_interval = 1

K360_type = 'Kitti360Dataset'
K360_data_root = 'data/kitti-360/'
K360_train_pkl = 'kitti360_infos_train.pkl' # 40000 frame
K360_train_interval = 1000
K360_val_pkl = 'kitti360_infos_val.pkl' # 10000 frame
K360_val_interval = 200