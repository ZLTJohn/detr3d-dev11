from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas,torch
joint_pkl = {
'nus': {
    'train': 'data/nus_v2/nuscenes_infos_train_part.pkl',
    'val': 'data/nus_v2/nuscenes_infos_val_part.pkl'
},
'way': {
    'train': 'data/waymo_dev1x/kitti_format/waymo_infos_train_2Hz_part.pkl',
    'val': 'data/waymo_dev1x/kitti_format/waymo_infos_val_2Hz_part.pkl',
},
'arg':{
    'train': 'data/argo2/argo2_infos_train_2Hz_part.pkl',
    'val': 'data/argo2/argo2_infos_val_2Hz_part.pkl'
}}
joint_class = {
'nus' : {
    'car': [0,1,2,3,4,6],
    'ped': [7],
    'cyclist':[5]
},
'way' : {
    'car': [0],
    'ped': [1],
    'cyclist':[2]
},
'arg':{
    'car': [15,9,5,4,20,12,22,21,16,0,10],
    'ped': [13,25],
    'cyclist': [1,2]
}}
def draw_stats(ds, class_name):
    interval=10
    max_dist = 50
    stats = [0]*(max_dist//interval)
    dists = []
    tot =0
    for split in ['train','val']:
        infos = pandas.read_pickle(joint_pkl[ds][split])
    
        for info in infos['data_list']:
            for i in info['instances']:
                if i['bbox_label_3d'] not in joint_class[ds][class_name]:
                    continue
                bbox3d = torch.tensor(i['bbox_3d'][:3])
                dist =bbox3d.norm()
                if dist>max_dist: continue
                stats[int(dist//interval)]+=1
                dists.append(dist)
                tot+=1

    name = '{}_{}'.format(class_name,ds)
    plt.hist(dists,bins = 10,)
    plt.savefig('debug/gt_stat_pictures/'+name)
    print('save to '+'gt_stat_pictures/'+name)

for class_name in ['car','ped','cyclist']:
    for ds in ['way','nus','arg']:
            draw_stats(ds,class_name)
            plt.gcf().clear()