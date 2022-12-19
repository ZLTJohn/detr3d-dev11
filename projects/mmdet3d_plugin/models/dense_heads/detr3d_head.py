import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmdet3d.models.task_modules.builder import \
    build_bbox_coder  # need to change
from mmdet3d.registry import MODELS
from mmdet.models.dense_heads import DETRHead
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig, reduce_mean)
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from projects.mmdet3d_plugin.models.task_modules.util import normalize_bbox
from projects.mmdet3d_plugin.models.utils.old_env import force_fp32

# from mmcv.runner import force_fp32#failed


@MODELS.register_module()
class Detr3DHead(DETRHead):
    """Head of Detr3D.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
            self,
            *args,
            with_box_refine=False,
            as_two_stage=False,
            transformer=None,
            bbox_coder=None,
            num_cls_fcs=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2,
                          0.2],  ## origin for nus
            code_size=10,
            **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.code_size = code_size
        self.code_weights = code_weights

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1  #？？？这不就没用了么...
        super(Detr3DHead, self).__init__(*args,
                                         transformer=transformer,
                                         **kwargs)
        # DETR sampling=False, so use PseudoSampler, format the result
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = TASK_UTILS.build(sampler_cfg)

        self.code_weights = nn.Parameter(torch.tensor(self.code_weights,
                                                      requires_grad=False),
                                         requires_grad=False)

    # forward_train -> loss
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])  #so shared weights？
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, img_metas, **kwargs):  #forward

        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            reg_branches=self.reg_branches
            if self.with_box_refine else None,  # noqa:E501
            img_metas=img_metas,
            **kwargs)
        hs = hs.permute(0, 2, 1, 3)  #what is this
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](
                hs[lvl])  #multiple forward？ 这里主要还是把format同步成target的样子
            tmp = self.reg_branches[lvl](hs[lvl])
            # print('tmp shape: {}'.format(tmp.shape))  # tmp shape: torch.Size([1, 900, 8])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            tmp[...,
                0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) +
                        self.pc_range[0])
            tmp[...,
                1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) +
                        self.pc_range[1])
            tmp[...,
                4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) +
                        self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # print('  '*3+'head: restrore outputs:',time.time()-__,'ms')
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def _get_target_single(
            self,
            cls_score,  #[query, 1]
            bbox_pred,  #[query, 8]
            gt_instances):  #[num_gt, 1+7]
        gt_bboxes = gt_instances.bboxes_3d  ##!!!!!
        gt_bboxes = torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        gt_labels = gt_instances.labels_3d
        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),        # turn bottm center into gravity center, key step
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler,PseudoSampler
        assign_result = self.assigner.assign(bbox_pred,
                                             cls_score,
                                             gt_bboxes,
                                             gt_labels,
                                             gt_bboxes_ignore=None)
        sampling_result = self.sampler.sample(
            assign_result, InstanceData(priors=bbox_pred),
            InstanceData(bboxes_3d=gt_bboxes))
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(
            num_bboxes)  #all query should learn its classification

        # bbox targets
        bbox_targets = torch.zeros_like(
            bbox_pred)[..., :self.code_size -
                       1]  #theta in gt_bbox here is still a single scalar
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[
            pos_inds] = 1.0  #only matched query will learn from bbox coord
        # if (gt_labels.shape[0]==0):
        #     breakpoint()
        # DETR
        if sampling_result.pos_gt_bboxes.shape[
                0] == 0:  #fix empty gt bug in multi gpu training
            sampling_result.pos_gt_bboxes = sampling_result.pos_gt_bboxes.reshape(
                0, self.code_size - 1)

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(
            self,  #get_targets
            batch_cls_scores,
            batch_bbox_preds,
            batch_gt_instances):

        num_imgs = len(batch_cls_scores)  #bs

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list
         ) = multi_apply(  #get targets each frame in batch, here bs = 1
             self._get_target_single, batch_cls_scores, batch_bbox_preds,
             batch_gt_instances)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_by_feat_single(
            self,
            batch_cls_scores,
            batch_bbox_preds,  ## variable names need to corrected!!!???
            batch_gt_instances):

        num_imgs = batch_cls_scores.size(0)  #batch size
        cls_scores_list = [batch_cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [batch_bbox_preds[i]
                           for i in range(num_imgs)]  #tensor to list of tensor
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg
         ) = cls_reg_targets  #here we get [bs, query, code_size-1]
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        batch_cls_scores = batch_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                batch_cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(batch_cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)
        # weights is query-wise 用于为每个query的loss加权，加权后统计loss和，然后用avg_factor除一下。
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        batch_bbox_preds = batch_bbox_preds.reshape(-1,
                                                    batch_bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(
            dim=-1)  #neg_query is all 0, log(0) is NaN
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            batch_bbox_preds[isnotnan, :self.code_size],
            normalized_bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss_by_feat(
            self,  #<-->loss_old
            batch_gt_instances,
            preds_dicts,
            batch_gt_instances_ignore=None):
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for batch_gt_instances_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = all_cls_scores[0].device
        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),        # turn bottm center into gravity center, key step
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        # batch_gt_instances_ignore_list = [batch_gt_instances_ignore
        #                                     for _ in range(num_dec_layers)]

        # all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        # all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        # all_gt_bboxes_ignore_list = [
        #     gt_bboxes_ignore for _ in range(num_dec_layers)
        # ]

        losses_cls, losses_bbox = multi_apply(  #calculate loss for each decoder layer
            self.loss_by_feat_single, all_cls_scores, all_bbox_preds,
            batch_gt_instances_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_by_feat_single(enc_cls_scores, enc_bbox_preds, batch_gt_instances_list)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def predict_by_feat(self,
                        preds_dicts,
                        img_metas,
                        rescale=False) -> InstanceList:
        #-->
        preds_dicts = self.bbox_coder.decode(
            preds_dicts)  #sin theta & cosine theta ---> theta
        num_samples = len(preds_dicts)  #batch size
        ret_list = []
        for i in range(num_samples):
            results = InstanceData()
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, self.code_size - 1)

            results.bboxes_3d = bboxes
            results.scores_3d = preds['scores']
            results.labels_3d = preds['labels']
            ret_list.append(results)
        return ret_list
