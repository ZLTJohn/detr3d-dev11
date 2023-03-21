import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import namedtuple
import numpy as np
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
    'ped': [14,25],
    'cyclist': [1,2]
}}
def draw_stats(ds, class_name):
    interval=10
    max_dist = 50
    dist_unit = max_dist//interval
    angle_interval = 50
    angle_unit = 2*np.pi / angle_interval
    a = np.linspace(-np.pi,np.pi,angle_interval)
    b = np.linspace(0,max_dist,interval)
    A, B = np.meshgrid(a, b)
    C = np.zeros(A.shape, dtype=np.int32)
    stats = [0]*(max_dist//interval)
    dists = []
    tot =0
    for split in ['train','val']:
        infos = pandas.read_pickle(joint_pkl[ds][split])
    
        for info in infos['data_list']:
            breakpoint()
            key = 'instances'
            if ds == 'way':
                key = 'cam_sync_instances'
            for i in info[key]:
                if i['bbox_label_3d'] not in joint_class[ds][class_name]:
                    continue
                bbox3d = torch.tensor(i['bbox_3d'][:3])
                dist =bbox3d.norm()
                if dist>=max_dist: continue
                if ds =='way':
                    Trans = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
                    bbox3d = np.array(bbox3d) @ Trans.T
                x,y,z = bbox3d     
                if ds =='nus':
                    x,y = y,x
                    y= -y
                theta = np.arctan2(y,x)+np.pi
                theta = int(theta/angle_unit)
                if theta>=angle_interval: 
                    theta -= angle_interval
                C[int(dist/dist_unit)][theta]+=1
                dists.append(dist)
                tot+=1
    #fake data:
    C = np.log10(C+1)
    #actual plotting
    ax = plt.subplot(111, polar=True)
    # ax.set_yticklabels([])
    ctf = ax.contourf(A, B, C, cmap=cm.jet)
    plt.colorbar(ctf)
    name = 'polar_{}_{}'.format(class_name,ds)
    plt.savefig('debug/gt_stat_pictures/'+name)
    print('save to '+'gt_stat_pictures/'+name)


for class_name in ['car']:
    for ds in ['nus']:
            draw_stats(ds,class_name)
            plt.gcf().clear()