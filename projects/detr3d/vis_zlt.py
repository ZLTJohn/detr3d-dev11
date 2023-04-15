import math
import os
import os.path as osp
import time
from copy import copy
from pathlib import Path

# import cv2
# import copy
# import random
import numpy as np
import torch
import torchvision.transforms as Trans
import torchvision.utils as vutils
from av2.structures.sweep import Sweep
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.structures import InstanceData
from PIL import Image, ImageDraw, ImageFont
from mmdet3d.structures.bbox_3d.utils import get_lidar2img

from .detr3d_featsampler import DefaultFeatSampler, GeoAwareFeatSampler

# lidar_path, img_path
class visualizer_zlt():

    def __init__(self,
                 debug_dir='debug/visualization',
                 gt_range=[0, 105],
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 vis_count=300,
                 debug_name=None,
                 draw_box=True,
                 vis_tensor=True,
                 draw_score_type=True,
                 ROIsampling=None):
        # TODO: add label details to BEV objects
        self.debug_dir = debug_dir
        self.gt_range = gt_range
        self.vis_count = vis_count
        self.pc_range = pc_range
        self.identity_range = np.array([0, 0, 0, 1, 1, 1])
        self.draw_box = draw_box
        self.PIL_transform = Trans.ToPILImage()
        self.vis_tensor = vis_tensor
        self.ROIsampling = ROIsampling
        if debug_name is None:
            self.debug_name = str(time.time())
        else:
            self.debug_name = debug_name
        self.draw_score_type = draw_score_type
    def infer_dataset_name(self,path):
        if type(path) == list:
            path = path[0]
        if 'nus' in path:
            return 'nuscenes'
        elif 'way' in path:
            return 'waymo'
        elif 'argo' in path:
            return 'argoverse2'
        elif 'lyft' in path:
            return 'lyft'
        elif 'kitti' in path:
            if '360' in path:
                return 'kitti-360'
            else:
                return 'kitti'
        else:
            return 'unknown'
        
    def get_dataset_name(self, img_meta):
        self.ds_name = img_meta.get('dataset_name',None)
        if self.ds_name is None:
            self.ds_name = self.infer_dataset_name(img_meta['img_path'])

    def load_pts(self, img_meta, pts= None):
        if type(pts) == type(None):
            path = img_meta['lidar_path']
            self.pts_dim = img_meta['num_pts_feats']
            file_format = os.path.splitext(path)[1]
            if file_format == '.feather':# TODO: use condition of file format
                sweep = Sweep.from_feather(Path(path))
                pts = sweep.xyz
            else:
                points = np.fromfile(path, dtype=np.float32)
                pts = points.reshape(-1, self.pts_dim)
        pts_homo = np.ones((pts.shape[0],4))
        pts_homo[:,:3] = pts[:,:3]
        trans = img_meta.get('trans_mat', np.identity(4))
        pts_homo = (trans @ pts_homo.T).T
        return pts_homo
    def load_imgs(self, img_paths):
        # imgs = [cv2.imread(path) for path in img_paths]
        imgs = [Image.open(path) for path in img_paths]
        return imgs

    def get_dir(self, sample_idx):
        dirs = [self.debug_dir, self.ds_name, sample_idx]
        dir = ''
        for name in dirs:
            dir = osp.join(dir, name)
            if not osp.exists(dir):
                os.mkdir(dir)
        return dir

    def filter_range(self, instances_3d):
        centers = instances_3d.bboxes_3d.gravity_center
        dist = torch.sqrt(
            torch.square(centers[..., 0]) + torch.square(centers[..., 1]) + torch.square(centers[..., 2]))
        mask = ((self.gt_range[0] < dist) & (dist < self.gt_range[1]))
        instances_3d = instances_3d[mask == 1]
        return instances_3d

    def topk(self, instances_3d):
        return instances_3d[:self.vis_count]

    def filter_score(self, instances_3d):
        pass  # 0.3

    def get_name(self):
        return self.debug_name

    def add_score(self, instances_3d):
        if instances_3d.get('scores_3d') is None:
            instances_3d.scores_3d = torch.ones_like(instances_3d.labels_3d)
        return instances_3d

    def get_color(self):
        return tuple([int(x) for x in np.random.randint(256, size=3)])

    def get_nice_font(self):
        if osp.exists('/usr/share/fonts/gnu-free/FreeMonoBold.ttf'):
            font = ImageFont.truetype(
                '/usr/share/fonts/gnu-free/FreeMonoBold.ttf', 24)
        else:
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/ttf-bitstream-vera/VeraMono.ttf',
                24)
        return font

    def tensor_to_PIL(self, tensor):
        grid = vutils.make_grid(tensor)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(
            'cpu', torch.uint8).numpy()
        return Image.fromarray(ndarr)

    def toInstance(self, instances_3d, device=torch.device('cuda', 0)):
        gt_instances_3d = InstanceData()
        bboxes = instances_3d['gt_bboxes_3d'].to(device)
        labels = torch.tensor(instances_3d['gt_labels_3d']).to(device)
        gt_instances_3d['bboxes_3d'] = bboxes
        gt_instances_3d['labels_3d'] = labels

        return self.add_score(gt_instances_3d)

    def visualize_dataset_item(self, frame, batch_gt_instances_3d=None, pts=None, name_suffix=None, dirname=None):
        batch_data_samples = frame['data_samples']
        batch_inputs_dict = frame['inputs']
        batch_input_metas = [batch_data_samples.metainfo]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        if batch_gt_instances_3d == None:
            batch_gt_instances_3d = [
                batch_data_samples.gt_instances_3d
            ]
        self.visualize(batch_gt_instances_3d, batch_input_metas,
                        batch_inputs_dict.get('imgs', None), pts,
                        name_suffix, dirname)

    def visualize(self, instances_3d=None, img_meta=None, img=None, pts= None, name_suffix='', dirname = None, pause = True):
        # support only one sample once
        if pause:
            breakpoint()
        if type(instances_3d) == list:
            instances_3d = instances_3d[0]
        if type(instances_3d) == dict:
            instances_3d = self.toInstance(instances_3d)
        if type(img_meta) == list:
            img_meta = img_meta[0]
        img_paths = img_meta['img_path']
        if type(img_paths) != list:
            img_paths = [img_paths]
        self.get_dataset_name(img_meta)
        if dirname is None:
            dirname = self.get_dir(str(img_meta['sample_idx']))
        filename = self.get_name()+name_suffix

        if instances_3d is not None:
            instances_3d = self.filter_range(instances_3d)
            instances_3d = self.topk(instances_3d)
            instances_3d = self.add_score(instances_3d)

            if img_meta.get('lidar_path'):
                pc = self.load_pts(img_meta, pts)
                self.save_bev(pc, instances_3d, dirname, filename)
            img_from_file = self.load_imgs(img_paths)
            metacopy = copy(img_meta)
            metacopy['lidar2img'] = metacopy['ori_lidar2img']
            metacopy['cam2img'] = metacopy['ori_cam2img']
            # (height, width) , not verified on waymo and nus
            w = max([i.size[0] for i in img_from_file])
            h = max([i.size[1] for i in img_from_file])
            metacopy['pad_shape'] = h, w
            self.save_bbox2img(img_from_file, instances_3d, metacopy, dirname,
                               filename)

        if img is not None and self.vis_tensor:
            if img.dim() == 5:
                img = img[0]
            N, C, H, W = img.size()
            # self.save_input_images(img, img_meta)
            img_from_tensor = [self.tensor_to_PIL(i) for i in img]
            self.save_bbox2img(img_from_tensor, instances_3d, img_meta,
                               dirname, filename + '_tsr')

    def save_input_images(self, img, img_meta):
        if type(img_meta) == list:
            img_meta = img_meta[0]
        # img.permute(0, 2, 3, 1)
        dirname = self.get_dir(str(img_meta['sample_idx']))
        for i in range(img.size(0)):
            out_name = dirname + '/' + 'input_img_{}.jpg'.format(i)
            vutils.save_image(img[i], out_name)
    # TODO: sync this function with the one in lyc_map
    def get_bbox2img(self, ref_pt, img_meta, IDs):
        '''
        imgs: n images
        refpt: tensor [num_pt,3]
        IDs: [num_pt], group id of each ref point
        '''
        if type(img_meta) == list:
            img_meta = img_meta[0]
        ref_pt = ref_pt.reshape(-1,3) # x,y,z, type
        sampler = DefaultFeatSampler()
        pt_cam, mask = sampler.project_ego2cam(ref_pt,self.identity_range, [img_meta])
        pt_cam = pt_cam.squeeze(0)
        mask = mask.squeeze(0).squeeze(-1)  # [cam, gt]

        imgs = self.load_imgs(img_meta['img_path'])
        num_cam = len(imgs)
        num_pt = ref_pt.shape[0]
        h, w = img_meta['pad_shape']
        pt_cam[..., 0] *= w
        pt_cam[..., 1] *= h
        draws = [ImageDraw.Draw(i) for i in imgs]
        self.colors = []
        for i in range(num_pt):
            self.colors.append(self.get_color())
        pt_colors = []
        for ID in IDs:
            pt_colors.append(self.colors[ID])
        for i in range(num_cam):
            self.draw_img_center(draws[i], pt_cam[i], mask[i], None, colors = pt_colors)

        imgs_np = [np.array(img) for img in imgs]
        return imgs_np

    def print_img_to_canvas(self, ax, imgs):
        sub_regions = np.split(np.split(ax, 3, axis=1), 2, axis=0)
        for i,img in enumerate(imgs):
            x = i/3
            y = i%3
            sub_regions[x,y].imshow(img)

    def save_bbox2img(self, imgs, instances_3d, img_meta, dirname, name=None):

        gt_bboxes_3d = instances_3d.bboxes_3d
        num_cam = len(imgs)
        num_query = len(instances_3d)
        h, w = img_meta['pad_shape']
        print(h, w)

        ref_pt = gt_bboxes_3d.gravity_center.view(1, -1, 3)  # 1 num_gt, 3
        # the ref_pt is not normalized, give identity pc_range to feat_sample
        if self.ROIsampling is not None:
            scale = 8
            tsr=[]
            for i in range(4):
                tsr.append(torch.ones((1,num_cam, 256, h//scale, w//scale)))
                scale *= 2
                base_dist = self.ROIsampling.get('base_dist',51.2)
                base_fxfy = self.ROIsampling.get('base_fxfy',-1)
            sampler = GeoAwareFeatSampler(base_dist=base_dist,base_fxfy=base_fxfy,debug=True)
            pt_cam, mask = sampler.forward(tsr,ref_pt,self.identity_range, [img_meta])
        else:
            sampler = DefaultFeatSampler()
            pt_cam, mask = sampler.project_ego2cam(ref_pt,self.identity_range, [img_meta])
        pt_cam = pt_cam.squeeze(0)
        mask = mask.squeeze(0).squeeze(-1)  # [cam, gt]
        pt_cam[..., 0] *= w
        pt_cam[..., 1] *= h
        print(ref_pt.size())
        print('frame:', img_meta['sample_idx'])
        # print('ego2global:', img_meta['ego2global'])
        draws = [ImageDraw.Draw(i) for i in imgs]

        if self.ROIsampling is None and self.draw_box:
            sampler = DefaultFeatSampler()
            corners_pt = gt_bboxes_3d.corners.view(1, -1, 3)  # 1 num_gt*8 3
            corners_cam, mask_corner = sampler.project_ego2cam(corners_pt,
                                                               self.identity_range,
                                                               [img_meta])
            corners_cam = corners_cam.squeeze(0)  # num_cam num_gt*8 2
            corners_cam[..., 0] *= w
            corners_cam[..., 1] *= h
            # TODO: a box should be printed out as long as one of 9 is in the image
            mask_corner = mask_corner.squeeze().reshape(num_cam, -1, 8)
            mask = (mask | (mask_corner.any(-1)))
            for i in range(num_cam):
                self.draw_img_bbox(draws[i], corners_cam[i], mask[i])

        for i in range(num_cam):
            self.draw_img_center(draws[i], pt_cam[i], mask[i], instances_3d)

        for i in range(num_cam):
            out_name = dirname + '/{}_{}.png'.format(name, i)
            imgs[i].save(out_name)

    def draw_img_center(self, draw, pt_cam, mask, instances_3d, colors = None):
        num_query = pt_cam.shape[0]
        font = self.get_nice_font()
        for j in range(num_query):
            if mask[j] == True:
                pt = pt_cam[j]
                x, y = (int(pt[0]), int(pt[1]))
                if colors is None:
                    color = self.get_color()
                else:
                    color = colors[j]
                r = 6
                draw.arc([(x - r, y - r), (x + r, y + r)],
                         0,
                         360,
                         width=4,
                         fill=color)
                if self.ROIsampling is None and self.draw_score_type:
                    score_type = str(round(float(instances_3d.scores_3d[j]),3))+\
                                'Cls'+str(int(instances_3d.labels_3d[j]))
                    draw.text((x + r, y + r), score_type, color, font=font)

    def draw_img_bbox(self, draw, corners_cam, mask):
        num_query = corners_cam.shape[0] // 8
        for j in range(num_query):
            if mask[j] == True:
                color = self.get_color()
                pts = corners_cam[j * 8:j * 8 + 8]
                for i in [0, 1, 2, 3]:

                    self.draw_img_line(draw, pts[i], pts[(i + 1) % 4], color)
                    self.draw_img_line(draw, pts[i + 4], pts[(i + 1) % 4 + 4],
                                       color)
                    self.draw_img_line(draw, pts[i], pts[i + 4], color)

    def draw_img_line(self, draw, pt0, pt1, color):
        pt0 = (int(pt0[0]), int(pt0[1]))
        pt1 = (int(pt1[0]), int(pt1[1]))
        draw.line([pt0, pt1], fill=color, width=1)
        # cv2.line(img_out, pt0, pt1, color, thickness=1)

    def add_egocar(self, instances_3d):
        # width: 1.8403966418483375
        # length: 4.42378805432057
        # height: 1.490000000000009
        box_tsr = instances_3d.bboxes_3d.tensor
        box = [0] * box_tsr.shape[-1]
        box[:7] = [0, 0, 0, 4.4, 1.8, 1.5, 0]
        ego_tensor = torch.tensor(box).reshape(1, -1).to(box_tsr)
        zero = torch.tensor(0).reshape(1).to(box_tsr)
        ego = InstanceData()
        ego.bboxes_3d = LiDARInstance3DBoxes(ego_tensor,
                                             box_dim=ego_tensor.shape[1])
        ego.labels_3d = zero
        ego.scores_3d = zero
        return InstanceData.cat([instances_3d, ego])

    def save_bev(self, pts, instances_3d, dirname, out_name=None):
        if isinstance(pts, list):
            pts = pts[0]
        if isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts)
        pc_range = self.pc_range
        mask = ((pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) &
                (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4]) &
                (pts[:, 2] > pc_range[2]) & (pts[:, 2] < pc_range[5]))
        pts = pts[mask,:3]
        res = 0.05
        x_max = 1 + int((pc_range[3] - pc_range[0]) / res)
        y_max = 1 + int((pc_range[4] - pc_range[1]) / res)
        im = torch.zeros(x_max + 1, y_max + 1, 3)
        x_img = (pts[:, 0] - pc_range[0]) / res
        x_img = x_img.round().long()
        y_img = (pts[:, 1] - pc_range[1]) / res
        y_img = y_img.round().long()
        im[x_img, y_img, :] = 1

        for i in [0]:
            for j in [0]:
                im[(x_img.long() + i).clamp(min=0, max=x_max),
                   (y_img.long() + j).clamp(min=0, max=y_max), :] = 1

        instances_3d = self.add_egocar(instances_3d.clone())
        ref = instances_3d.bboxes_3d.gravity_center
        print('reference', ref.size())
        ref_pts_x = ((ref[..., 0] - pc_range[0]) / res).round().long()
        ref_pts_y = ((ref[..., 1] - pc_range[1]) / res).round().long()

        im = im.permute(2, 0, 1)
        out_name = dirname + '/' + out_name + '_bev.jpg'
        img = self.PIL_transform(im)
        draw = ImageDraw.Draw(img)
        font = self.get_nice_font()

        num_q = ref_pts_x.shape[-1]
        for i in range(num_q):
            x, y = int(ref_pts_y[i]), int(ref_pts_x[i])
            r = 5
            dy = 0  # random.randint(0,20)*30
            if i % 10 == 0:
                color = self.get_color()
                color_front = self.get_color()
            if self.draw_score_type:
                score_type = str(int(instances_3d.labels_3d[i]))
                # + str(round(float(instances_3d.scores_3d[i]),2))[1:]
                draw.text((x + r - 3, y + r - 3), score_type, color, font=font)
            draw.arc([(x - r, y - r), (x + r, y + r)],
                     0,
                     360,
                     width=100,
                     fill=color)
            # draw.line([x,y,x,y+dy], fill=color, width=3)
        if self.draw_box:
            # 1 num_gt*4 3
            corners = instances_3d.bboxes_3d.corners[:, ::2, :].view(-1, 3)
            corners_x = ((corners[..., 0] - pc_range[0]) / res).round().long()
            corners_y = ((corners[..., 1] - pc_range[1]) / res).round().long()
            for j in range(num_q):
                drawx = corners_y[j * 4:j * 4 + 4]
                drawy = corners_x[j * 4:j * 4 + 4]
                pts = [(int(drawx[i]), int(drawy[i])) for i in [0, 1, 3, 2]]
                draw.polygon(pts, outline=color, width=2)
                #(x0y0z0, x0y1z1, x1y1z1, x1y0z0)
                (fx, fy), r = self.get_circle_from_diam(pts[-2], pts[-1])
                cx, cy = int(ref_pts_y[j]), int(ref_pts_x[j])
                self.draw_arrowedLine(draw, (cx, cy), (fx, fy),
                                      width=3,
                                      color=color_front)

        img.save(out_name)
        # vutils.save_image(im, out_name)

    def get_circle_from_diam(self, pt1, pt2):
        d = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
        o = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        return o, d // 2

    def draw_arrowedLine(self, draw, ptA, ptB, width=1, color=(0, 255, 0)):
        """Draw line from ptA to ptB with arrowhead at ptB."""
        # Draw the line without arrows
        draw.line((ptA, ptB), width=width, fill=color)

        # Now work out the arrowhead
        # = it will be a triangle with one vertex at ptB
        # - it will start at 95% of the length of the line
        # - it will extend 8 pixels either side of the line
        x0, y0 = ptA
        x1, y1 = ptB
        # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
        xb = 0.7 * (x1 - x0) + x0
        yb = 0.7 * (y1 - y0) + y0

        # Work out the other two vertices of the triangle
        # Check if line is vertical
        if x0 == x1:
            vtx0 = (xb - 5, yb)
            vtx1 = (xb + 5, yb)
        # Check if line is horizontal
        elif y0 == y1:
            vtx0 = (xb, yb + 5)
            vtx1 = (xb, yb - 5)
        else:
            alpha = math.atan2(y1 - y0, x1 - x0) - 90 * math.pi / 180
            a = 8 * math.cos(alpha)
            b = 8 * math.sin(alpha)
            vtx0 = (xb + a, yb + b)
            vtx1 = (xb - a, yb - b)

        # Now draw the arrowhead triangle
        draw.polygon([vtx0, vtx1, ptB], fill=color)
    def add_lidar2img(self, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            l2i = list()
            ori_l2i = list()
            for i in range(len(meta['cam2img'])):
                c2i = torch.tensor(meta['cam2img'][i]).double()
                l2c = torch.tensor(meta['lidar2cam'][i]).double()
                # l2c_t = l2c[0:3,3:4]
                # noise = torch.randn_like(l2c_t) *0.1 # -0.2 ~ 0.2
                # l2c[0:3,3:4] = l2c[0:3,3:4] + l2c_t * noise
                l2i.append(get_lidar2img(c2i, l2c).float().numpy())

                ori_c2i = torch.tensor(meta['ori_cam2img'][i]).double()
                ori_l2i.append(get_lidar2img(ori_c2i, l2c).float().numpy())
            meta['lidar2img'] = l2i
            meta['ori_lidar2img'] = ori_l2i
        return batch_input_metas

#             img_ret.append(img_out)
#         return img_ret

# def show_camera_image(camera_image, layout):
#   """Display the given camera image."""
#   ax = plt.subplot(*layout)
#   plt.imshow(camera_image)
#   plt.grid(False)
#   plt.axis('off')
#   return ax

# def save_temporal_frame(union,
#                         dirname='debug/debug_temporal1',
#                         ds_name='waymo'):
#     # union = np.load('queue_union.npy',allow_pickle=True).reshape(1)[0]
#     gt = union['gt_bboxes_3d'].data
#     imgs = union['img'].data
#     name = str(union['img_metas'].data[1]['sample_idx'])
#     # dirname='/home/zhengliangtao/pure-detr3d/debug/debug_temporal1'
#     colors = np.random.randint(256, size=(300,3))
#     out1 = save_bbox2img(imgs[1:2], [gt], [union['img_metas'].data[1]],
#                           union['img_metas'].data[1]['filename'],
#                           dirname= dirname, name = name, colors = colors)
#     out0 = save_bbox2img(imgs[1:2], [gt], [union['img_metas'].data[0]],
#                           union['img_metas'].data[0]['filename'],
#                           dirname= dirname, name = name+'_prev',
#                           colors = colors)
#     plt.figure(figsize=(40, 60))
#     for i,img_out in enumerate(out0):
#         show_camera_image(img_out[...,::-1], [6, 2, i*2+1])
#     for i,img_out in enumerate(out1):
#         show_camera_image(img_out[...,::-1], [6, 2, i*2+2])
#     plt.savefig(dirname+'/{}_{}_all.png'.format(ds_name,name))
