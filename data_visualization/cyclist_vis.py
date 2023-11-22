import pandas
from projects.detr3d.vis_zlt import visualizer_zlt
from mmengine.structures import InstanceData
import torch
from mmdet3d.structures import LiDARInstance3DBoxes,CameraInstance3DBoxes

info = pandas.read_pickle('data/nus_v2/nuscenes_infos_val_part.pkl')
boxes = []
for item in info['data_list']:
    for i in item['instances']:
        if i['bbox_label_3d'] == 5:
            x,y = i['bbox_3d'][0], i['bbox_3d'][1]
            i['bbox_3d'][0], i['bbox_3d'][1] = y, x
            boxes.append(i['bbox_3d'])
boxes = torch.tensor(boxes)
bboxes =LiDARInstance3DBoxes(boxes)

info = pandas.read_pickle('data/waymo_dev1x/kitti_format/waymo_infos_val_2Hz_part.pkl')
boxes1 = []
for item in info['data_list']:
    for i in item['instances']:
        if i['bbox_label_3d'] == 2:
            boxes1.append(i['bbox_3d'])
boxes1 = torch.tensor(boxes1)
bboxes1 =CameraInstance3DBoxes(boxes1).convert_to(0)

bboxes = bboxes.cat((bboxes,bboxes1))
gt = InstanceData()
gt.bboxes_3d = bboxes
vis = visualizer_zlt(debug_dir='./',debug_name='test',draw_box=False,draw_score_type=False)
pts = torch.tensor([0,0,0]).reshape(1,3)
gt.labels_3d = torch.ones_like(gt.bboxes_3d.tensor[:,0])
gt =  vis.add_score(gt)
vis.save_bev(pts,gt,'./','cyclist_nuway_rotate')