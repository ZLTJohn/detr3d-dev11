import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


class QuickCumsum(torch.autograd.Function):
    """Sum up the features of all points within the same voxel through
    cumulative sum operator."""

    @staticmethod
    def forward(ctx, x, coor, ranks):
        """Forward function.
        All inputs should be sorted by the rank of voxels.
        The function implementation process is as follows:
            - step 1: Cumulatively sum the point-wise feature alone the point
                queue.
            - step 2: Remove the duplicated points with the same voxel rank and
                only retain the last one in the point queue.
            - step 3: Subtract each point feature with the previous one to
                obtain the cumulative sum of the points in the same voxel.
        Args:
            x (torch.tensor): Point-wise features in shape (N_Points, C).
            coor (torch.tensor): The coordinate of points in the feature
                coordinate system in shape (N_Points, D).
            ranks (torch.tensor): The rank of voxel that a point is belong to.
                The shape should be (N_Points).
        Returns:
            tuple[torch.tensor]: Voxel-wise features in shape (N_Voxels, C);
                The coordinate of voxels in the feature coordinate system in
                shape (N_Voxels,3).
        """
        x = x.cumsum(0) #前缀和
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, coor = x[kept], coor[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for coordinates
        ctx.mark_non_differentiable(coor)

        return x, coor

    @staticmethod
    def backward(ctx, gradx, gradcoor):
        """Backward propagation function.
        Args:
            gradx (torch.tensor): Gradient of the output parameter 'x' in the
                forword function.
            gradcoor (torch.tensor): Gradient of the output parameter 'coor' in
                the forword function.
        Returns:
            torch.tensor: Gradient of the input parameter 'x' in the forword
                function.
        """
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


@MODELS.register_module()
class LSSViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer.
    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_
    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        max_voxel_points (int): Specify the maximum point number in a single
            voxel during the acceleration.
    """

    def __init__(self,
                 grid_config,
                 input_size,
                 downsample,
                 in_channels,
                 out_channels,
                 accelerate=False,
                 max_voxel_points=300,
                 dataset_names = None,
                 uniform_frustum = False,
                 focal_length = -1,
                 debug = False):
        super(LSSViewTransformer, self).__init__()
        self.create_grid_infos(**grid_config)
        self.uniform_frustum = uniform_frustum
        if dataset_names is None:
            self.uniform_frustum = True
        self.create_frustum(grid_config['depth'], input_size, downsample, dataset_names)
        self.out_channels = out_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.max_voxel_points = max_voxel_points
        self.initial_flag = True
        self.focal_length = focal_length
        self.debug = debug

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.
        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def create_frustum(self, depth_cfg, input_sizes, downsample, dataset_names):
        """Generate the frustum template for each image.
        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        self.frustum = {}
        if self.uniform_frustum:
            assert type(input_sizes) == tuple
            input_sizes = [input_sizes]
            dataset_names = ['uniform']
        for input_size, dataset_name in zip(input_sizes,dataset_names):
            H_in, W_in = input_size
            H_feat, W_feat = H_in // downsample, W_in // downsample # fpn他不管了么？看起来只有一层？
            d = torch.arange(*depth_cfg, dtype=torch.float)\
                .view(-1, 1, 1).expand(-1, H_feat, W_feat)
            self.D = d.shape[0]
            x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
                .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
            y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
                .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

            # D x H x W x 3
            self.frustum[dataset_name] = torch.stack((x, y, d), -1)

    def get_lidar_coor(self, rots, trans, cam2imgs, post_rots, post_trans, scale_factors_2d):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.
        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).
        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _ = trans.shape
        # post-transformation
        # B x N x D x H x W x 3
        # frustum里面存的是每个img feat grid point中心在resize过的图中对应的像素坐标，frustum的大小就是图片最终的输入大小。
        # 注意到 img feat过了fpn之后只剩一层了。
        # post_trans：crop了之后，新图的(0,0)对应着原来的(u,v)， post trans就是做这个的
        # breakpoint()  
        points = self.frustum[self.curr_dataset].to(rots) - post_trans.view(B, N, 1, 1, 1, 3) 
        # post_rots：resize、rotate、warping对frustum的影响。注意到我们是要把现在的tsr还原成原图像分辨率所在的位置
        # 我们这里只考虑(u,v)，对深度是没有任何影响的。
        # bevdet里面只考虑x上的scale factor，不考虑y的，这肯定有问题
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))
        # scale the depth!!
        if self.focal_length != -1:
            fx = cam2imgs[:,:,0,0] * scale_factors_2d[...,0]
            fy = cam2imgs[:,:,1,1] * scale_factors_2d[...,1]
            px_size = torch.sqrt((1/fx)**2+(1/fy)**2)
            fix_size = ((1/self.focal_length**2)+(1/self.focal_length**2))**0.5
            # assuming we have got focal-invariant depth
            depth_scalor = fix_size / px_size
            if self.debug is True:
                depth_scalor = 1/depth_scalor

            points[..., 2:3, :] *= depth_scalor.view(B,N,1,1,1,1,1)
            # WRONG code:
            # fx = cam2imgs[:,:,0,0] * torch.abs(post_rots[:,:,0,0])# 直接用0,0不对的
            # scale_factor = self.focal_length / fx
            # points[..., 2:3, :] *= scale_factor.view(B,N,1,1,1,1,1)
        # cam_to_ego
        # 将不同深度的点的三维图像坐标恢复出来 # here we can add relative depth
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        # 内参矩阵乘外参rotation，combine就是img2lidar
        combine = rots.matmul(torch.inverse(cam2imgs[...,:3,:3]))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # 外参translation
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def voxel_pooling_prepare(self, coor, x):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        Returns:
            tuple[torch.tensor]: Feature of points in shape (N_Points, C);
                Coordinate of points in the voxel space in shape (N_Points, 3);
                Rank of the voxel that a point is belong to in shape
                (N_Points); Reserved index of points in the input point queue
                in shape (N_Points).
        """
        B, N, D, H, W, C = x.shape
        num_points = B * N * D * H * W
        # flatten x
        x = x.reshape(num_points, C)
        # record the index of selected points for acceleration purpose
        point_idx = torch.range(0, num_points - 1, dtype=torch.long)
        # convert coordinate into the voxel space
        # voxel坐标
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        # 取整
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B-1).reshape(B, 1).\
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2]) # another dataset-specific mask
        x, coor, point_idx = x[kept], coor[kept], point_idx[kept]

        # get tensors from the same voxel next to each other
        ranks = coor[:, 0] * (self.grid_size[1] * self.grid_size[2] * B)
        ranks += coor[:, 1] * (self.grid_size[2] * B)
        ranks += coor[:, 2] * B + coor[:, 3]
        order = ranks.argsort()
        # point_idx原本的点编号，主要是有的点被筛掉了
        return x[order], coor[order], ranks[order], point_idx[order]

    def voxel_pooling(self, coor, x):
        """Generate bird-eye-view features with the pseudo point cloud.
        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        Returns:
            torch.tensor: Bird-eye-view features in shape (B, C, H_BEV, W_BEV).
        """
        B, _, _, _, _, C = x.shape
        x, coor, ranks, _ = self.voxel_pooling_prepare(coor, x)

        # cumsum trick
        x, coor = QuickCumsum.apply(x, coor, ranks)

        # griddify (B x C x Z x X x Y)
        grid_size = self.grid_size.to(torch.long)
        final = torch.zeros((B, C, grid_size[2], grid_size[1], grid_size[0]),
                            device=x.device)
        final[coor[:, 3], :, coor[:, 2], coor[:, 1], coor[:, 0]] = x
        # collapse Z，有时候grid里设置的z不是1,这时候需要将同一个pillar内的voxel feature concat起来
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def init_acceleration(self, coor, x):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.
        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """
        x, coor, ranks, point_idx = self.voxel_pooling_prepare(coor, x)
        # count for the repeat times of the same voxel rank in the point queue.
        repeat_times = torch.ones(
            coor.shape[0], device=coor.device, dtype=coor.dtype)
        times = 0
        repeat_times[0] = 0
        cur_rank = ranks[0]

        for i in range(1, ranks.shape[0]):
            if cur_rank == ranks[i]:
                times += 1
                repeat_times[i] = times
            else:
                cur_rank = ranks[i]
                times = 0
                repeat_times[i] = times
        # remove the point whose repeat time is exceed the threshold.
        kept = repeat_times < self.max_voxel_points
        repeat_times, coor = repeat_times[kept], coor[kept]
        x, point_idx = x[kept], point_idx[kept]

        coor = torch.cat([coor, repeat_times.unsqueeze(-1)], dim=-1)
        self.coor = coor
        self.point_idx = point_idx
        self.initial_flag = False

    def voxel_pooling_accelerated(self, x):
        """Conducting voxel pooling in accelerated mode.
        Args:
            x (torch.tensor): The feature of the volumes in shape
                (B, N_cams, D, H, W, C).
        Returns:
            torch.tensor: Bird-eye-view features in shape (B, C, H_BEV, W_BEV).
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)[self.point_idx]

        # griddify (B x C x Z x X x Y)
        gs = self.grid_size.to(torch.long)
        final = torch.zeros((B, C, gs[2], gs[1], gs[0], self.max_voxel_points),
                            device=x.device)
        c = self.coor
        final[c[:, 3], :, c[:, 2], c[:, 1], c[:, 0], c[:, 4]] = x
        final = final.sum(-1)

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        return final

    def forward(self, input, img_metas, visualizer=None):
        """Transform image-view feature into bird-eye-view feature.
        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)
        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x)
        depth = x[:, :self.D].softmax(dim=1)
        if visualizer is not None:
            visualizer.store_depth(depth.view(B,N,-1,H,W))
        tran_feat = x[:, self.D:(self.D + self.out_channels)]
        bevdet_input = self.get_bevdet_input(img_metas)
        self.curr_dataset = img_metas[0].get('dataset_name','nuscenes') # TODO: infer ds name if get None
        if self.uniform_frustum:
            self.curr_dataset = 'uniform'
        # Lift
        volume = depth.unsqueeze(1) * tran_feat.unsqueeze(2)
        volume = volume.view(B, N, self.out_channels, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        if self.accelerate:
            if self.initial_flag:
                coor = self.get_lidar_coor(*bevdet_input)
                self.init_acceleration(coor, volume)
            bev_feat = self.voxel_pooling_accelerated(volume)
        else:
            coor = self.get_lidar_coor(*bevdet_input)
            bev_feat = self.voxel_pooling(coor, volume)
        return bev_feat

    def get_bevdet_input(self,img_metas):
        bevdet_input = [[],[],[],[],[],[]]
        for meta in img_metas:
            for idx,item in enumerate(meta['bevdet_input']):
                bevdet_input[idx].append(item)
            bevdet_input[5].append(torch.tensor(meta['scale_factor']))
        for i in range(6): 
            bevdet_input[i] = torch.stack(bevdet_input[i])
        return bevdet_input
        # results['bevdet_input'] = (rots, trans, intrins, post_rots, post_trans) with scale factor
        # self, rots, trans, cam2imgs, post_rots, post_trans):
    
    # def get_lidar_coor(self, rots, trans, cam2imgs, post_rots, post_trans, scale_factors_2d):
    #     B, N, _ = trans.shape
    #     # image crop && rotate
    #     points = self.frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3) 
    #     points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
    #         .matmul(points.unsqueeze(-1))
        
    #     # scale the depth!!
    #     if self.focal_length != -1:
    #         fx = cam2imgs[:,:,0,0] * scale_factors_2d[...,0]  # 原始内参*resize_scale
    #         fy = cam2imgs[:,:,1,1] * scale_factors_2d[...,1]
    #         px_size = torch.sqrt((1/fx)**2+(1/fy)**2)
    #         fix_size = ((1/self.focal_length**2)+(1/self.focal_length**2))**0.5   # 计算参考焦距的像素大小
    #         # assuming we have got focal-invariant depth
    #         depth_scalor = fix_size / px_size
    #         points[..., 2:3, :] *= depth_scalor.view(B,N,1,1,1,1,1)

    #     points = torch.cat(
    #         (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
    #     combine = rots.matmul(torch.inverse(cam2imgs[...,:3,:3]))
    #     points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    #     points += trans.view(B, N, 1, 1, 1, 3)

    #     return points
    #  谢谢你的回复！我明白了，原来的depth estimation，估出的深度是基于源数据集的焦距的，所以切换焦距的时候要把深度缩放。
    #  根据我的理解，我在生成image frustum坐标的时候，添加了这一模块：
    #  这样做，就能把深度这一维缩放了。我基于nuscenes训练了一个模型，在nuscenes和waymo上测，结果nuscenes不变，
    #  waymo还是很糟糕，当然比之前好了一些。具体来说，是29.5%/0.1% -> 28.9%/1.6%
    #  做一下可视化