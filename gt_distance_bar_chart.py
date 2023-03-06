from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas,torch
def draw_stats(ds, split, class_name):
    joint_pkl = {
    'nus': {
        'train': 'data/nus_v2/nuscenes_infos_train_part.pkl',
        'val': 'data/nus_v2/nuscenes_infos_val_part.pkl'
    },
    'way': {
        'train': 'data/waymo_dev1x/kitti_format/waymo_infos_train_2Hz_part.pkl',
        'val': 'data/waymo_dev1x/kitti_format/waymo_infos_val_2Hz_part.pkl',
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
    }}
    interval=10
    max_dist = 50
    infos = pandas.read_pickle(joint_pkl[ds][split])
    stats = [0]*(max_dist//interval)
    dists = []
    tot =0
    for info in infos['data_list']:
        for i in info['instances']:
            if i['bbox_label'] not in joint_class[ds][class_name]:
                continue
            bbox3d = torch.tensor(i['bbox_3d'][:3])
            dist =bbox3d.norm()
            if dist>max_dist: continue
            stats[int(dist//interval)]+=1
            dists.append(dist)
            tot+=1

    name = '{}_{}_{}'.format(class_name,split,ds)
    # # def plot_student_results(student, scores_by_test, cohort_size):
    # fig, ax1 = plt.subplots(figsize=(9, 7), constrained_layout=True)
    # # fig.canvas.manager.set_window_title('{}_{}_{}'.format)

    # ax1.set_title(name)
    # ax1.set_xlabel('percentage')

    # percentiles = [i/tot*100 for i in stats]

    # rects = ax1.barh(['[{},{})'.format(i,i+10) for i in range(0,max_dist,interval)], percentiles, align='center', height=0.5)
    # # Partition the percentile values to be able to draw large numbers in
    # # white within the bar, and small numbers in black outside the bar.
    # # large_percentiles = [to_ordinal(p) if p > 40 else '' for p in percentiles]
    # str_percentiles = ['{}%'.format(round(p,2)) for p in percentiles]
    # ax1.bar_label(rects, str_percentiles,
    #                 padding=5, color='black', fontweight='bold')
    # # ax1.bar_label(rects, large_percentiles,
    # #               padding=-32, color='white', fontweight='bold')

    # ax1.set_xlim([0, 100])
    # ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # ax1.xaxis.grid(True, linestyle='--', which='major',
    #                 color='grey', alpha=.25)
    # ax1.axvline(50, color='grey', alpha=0.25)  # median position

    # # Set the right-hand Y-axis ticks and labels
    # ax2 = ax1.twinx()
    # # Set equal limits on both yaxis so that the ticks line up
    # ax2.set_ylim(ax1.get_ylim())
    # # Set the tick locations and labels

    # ax2.set_yticks(
    #     np.arange(len(stats)),
    #     labels=['[{},{})'.format(i,i+10) for i in range(0,max_dist,interval)])

    # ax2.set_ylabel('gt dist to ego center')
    plt.hist(dists,bins = 10)
    plt.savefig('gt_stat_pictures/'+name)
    print('save to '+'gt_stat_pictures/'+name)

for split in ['train','val']:
    for class_name in ['car','ped','cyclist']:
        for ds in ['way','nus']:
            draw_stats(ds,split,class_name)
        plt.gcf().clear()