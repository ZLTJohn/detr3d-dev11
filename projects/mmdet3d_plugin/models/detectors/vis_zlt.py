import cv2
import numpy as np
import time
import torchvision.utils as vutils
import torch
import copy
import os
import torchvision.transforms as Trans
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import random
import os.path as osp
from  projects.mmdet3d_plugin.models.utils.detr3d_transformer import feature_sampling
##lidar_path, img_path
'''
forget to transform the coordinate of previous frame 3d box centers
'''

class visualizer_zlt():
    def __init__(self,
                 debug_dir='debug/visualization',
                 gt_range=[0, 105],
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 vis_count=300,
                 debug_name=None):
        self.debug_dir = debug_dir
        self.gt_range = gt_range
        self.vis_count = vis_count
        self.pc_range = pc_range
        if debug_name == None:
            self.debug_name = str(time.time())
        else:
            self.debug_name = debug_name

    def infer_dataset_name(self, img_paths):
        if 'way' in img_paths:
            self.ds_name = 'waymo'
            self.pts_dim = 6
        elif 'nus' in img_paths:
            self.ds_name = 'nuscenes'
            self.pts_dim = 5
        else: 
            self.ds_name = 'argo2'
            self.pts_dim = -1# not implemented

    def load_pts(self, path):
        points = np.fromfile(path, dtype=np.float32)
        return points.reshape(-1, self.pts_dim)

    def get_dir(self, sample_idx):
        dirs = [self.debug_dir, self.ds_name, sample_idx]
        dir = ''
        for name in dirs:
            dir = osp.join(dir,name)
            if not osp.exists(dir):
                os.mkdir(dir)
        return dir
    def filter_range(self, instances_3d):
        centers = instances_3d.bboxes_3d.gravity_center
        dist = torch.sqrt(torch.square(centers[...,0])+torch.square(centers[...,1]))
        mask = ((self.gt_range[0] < dist) & (dist < self.gt_range[1]))
        instances_3d = instances_3d[mask==1]
        return instances_3d

    def topk(self, instances_3d):
        return instances_3d[:self.vis_count]

    def filter_score(self, instances_3d):
        pass#0.3
    def get_name(self):
        return self.debug_name

    def visualize(self, instances_3d, img_meta, img = None):
        # support only one sample once
        if type(instances_3d) == list:
            instances_3d = instances_3d[0]
        if type(img_meta) == list:
            img_meta = img_meta[0]
        img_paths = img_meta['img_path']
        if type(img_paths) != list:
            img_paths = [img_paths]
        self.infer_dataset_name(img_paths[0])
        instances_3d = self.filter_range(instances_3d)
        instances_3d = self.topk(instances_3d)
        dirname = self.get_dir(str(img_meta['sample_idx']))
        filename = self.get_name()
        if img_meta.get('lidar_path'):
            pc = self.load_pts(img_meta['lidar_path'])
            self.save_bev(pc, instances_3d, dirname, filename, test_output = None)
        self.save_bbox2img(img, instances_3d, img_meta, dirname, filename)

    def save_bbox2img(self, img, instances_3d, img_meta, dirname='debug_coord', name = None):

        gt_bboxes_3d = instances_3d.bboxes_3d
        ref_pt = gt_bboxes_3d.gravity_center.view(1, -1, 3) # 1 num_gt, 3
        print(ref_pt.size())
        breakpoint()
        pt_cam, mask = feature_sampling(None, ref_pt, self.pc_range, [img_meta], no_sampling=True) 
        pt_cam = pt_cam.squeeze()
        mask = mask.squeeze()
        img_paths = img_meta['img_path']
        if type(img_paths) != list:
            img_paths = [img_paths]

        num_cam = len(img_paths)
        num_query = pt_cam.shape[-1]

        if (name == None):
            name = str(time.time())
        for i in range(num_cam):
            img_out = cv2.imread(img_paths[i])
            h,w = img_meta['pad_shape']
            print(h,w)
            for j in range(num_query):
                pt = pt_cam[i,j]

                if mask[i,j] == True:
                    color = np.random.randint(256, size=3)
                    color = [int(x) for x in color]
                    cv2.circle(img_out, (int(pt[0]*w),int(pt[1]*h)), radius=6,color=color, thickness = 4)
                    #  cv2.circle(imgs[j], (int(pt[0]),int(pt[1])), radius=5 , color = color, thickness = 4)
                    if instances_3d.get('scores_3d') !=None:
                        score_type = str(round(float(instances_3d.scores_3d[j]),3))+'Cls'+str(int(instances_3d.labels_3d[j]))
                        cv2.putText(img_out, score_type, (int(pt[0]*w),int(pt[1]*h)),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=color)
            cv2.imwrite(dirname+'/{}_{}.png'.format(name, i), img_out)


    def save_bev(self, pts, instances_3d, dirname, out_name = None, test_output = None):
        ref = instances_3d.bboxes_3d.gravity_center

        if isinstance(pts, list):
            pts = pts[0]
        if isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts)
        pc_range= self.pc_range
        mask = ((pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) & 
            (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4]) &
            (pts[:, 2] > pc_range[2]) & (pts[:, 2] < pc_range[5]))
        pts = pts[mask]
        res = 0.05
        x_max = 1 + int((pc_range[3] - pc_range[0]) / res)
        y_max = 1 + int((pc_range[4] - pc_range[1]) / res)
        im = torch.zeros(x_max+1, y_max+1, 3)
        x_img = (pts[:, 0] - pc_range[0]) / res
        x_img = x_img.round().long()
        y_img = (pts[:, 1] - pc_range[1]) / res
        y_img = y_img.round().long()
        im[x_img, y_img, :] = 1

        for i in [ 0]:
            for j in [0]:
                im[(x_img.long()+i).clamp(min=0, max=x_max), 
                    (y_img.long()+j).clamp(min=0, max=y_max), :] = 1
        print('reference', ref.size())
        ref_pts_x = ((ref[..., 0] - pc_range[0]) / res).round().long()
        ref_pts_y = ((ref[..., 1] - pc_range[1]) / res).round().long()
        for i in range(-2,3):
            for j in range(-2,3):
                im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                    (ref_pts_y.long()+j).clamp(min=0, max=y_max), 0] = 1
                im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                    (ref_pts_y.long()+j).clamp(min=0, max=y_max), 1:2] = 0
        im = im.permute(2, 0, 1)
        out_name = dirname + '/' + out_name + '_bev.jpg'
        T = Trans.ToPILImage()
        img = T(im)
        draw = ImageDraw.Draw(img)
        if osp.exists('/usr/share/fonts/gnu-free/FreeMonoBold.ttf'):
            font = ImageFont.truetype('/usr/share/fonts/gnu-free/FreeMonoBold.ttf',24)
        else:
            font = ImageFont.truetype('/usr/share/fonts/truetype/ttf-bitstream-vera/VeraMono.ttf',24)

        num_q = ref_pts_x.shape[-1]
        for i in range(num_q):
            x,y = int(ref_pts_y[i]), int(ref_pts_x[i])
            r=10
            dy = 0#random.randint(0,20)*30
            if i % 10 ==0:
                color = tuple([int(x) for x in np.random.randint(256, size=3)])
            if instances_3d.get('scores_3d') !=None:
                score_type = str(int(instances_3d.labels_3d[i])) + str(round(float(instances_3d.scores_3d[i]),2))[1:]
                draw.text((x+r-3,y+r-3) ,score_type , color, font=font)
            draw.arc([(x-r,y-r),(x+r,y+r)],0,360, width=100, fill=color)
            # draw.line([x,y,x,y+dy], fill=color, width=3)
        img.save(out_name)
        # vutils.save_image(im, out_name)



#             img_ret.append(img_out)
#         return img_ret

# def show_camera_image(camera_image, layout):
#   """Display the given camera image."""
#   ax = plt.subplot(*layout)
#   plt.imshow(camera_image)
#   plt.grid(False)
#   plt.axis('off')
#   return ax

# def save_temporal_frame(union, dirname='debug/debug_temporal1', ds_name='waymo'):
#     # union = np.load('queue_union.npy',allow_pickle=True).reshape(1)[0]
#     gt = union['gt_bboxes_3d'].data
#     imgs = union['img'].data
#     name = str(union['img_metas'].data[1]['sample_idx']) 
#     # dirname='/home/zhengliangtao/pure-detr3d/debug/debug_temporal1'
#     colors = np.random.randint(256, size=(300,3))
#     out1 = save_bbox2img(imgs[1:2], [gt], [union['img_metas'].data[1]], union['img_metas'].data[1]['filename'], 
#                 dirname= dirname, name = name, colors = colors)
#     out0 = save_bbox2img(imgs[1:2], [gt], [union['img_metas'].data[0]], union['img_metas'].data[0]['filename'], 
#                 dirname= dirname, name = name+'_prev', colors = colors)
#     plt.figure(figsize=(40, 60))    
#     for i,img_out in enumerate(out0):
#         show_camera_image(img_out[...,::-1], [6, 2, i*2+1])
#     for i,img_out in enumerate(out1):
#         show_camera_image(img_out[...,::-1], [6, 2, i*2+2])
#     plt.savefig(dirname+'/{}_{}_all.png'.format(ds_name,name))