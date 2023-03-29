#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import math
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Tuple, List, Dict, Optional, Union, Any

from .. import LOGGER, check_version, load_cfg

# from .layers import *
from .utils_general import make_divisible
from .utils_torch import fuse_conv_and_bn, initialize_weights, model_info, scale_img, freeze_params, freeze_bn
# from .utils.plots import feature_visualization
from .yolov5 import *

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Model(nn.Module):
    """ Yolo model with backbone, neck and head."""
    def __init__(self, cfg='yolov5s.yaml', hyp='./hyp.scratch.yaml', 
                 ch=3, anchors=None, is_scripting=False,
                ):
        super().__init__()
        self.cfg = load_cfg(cfg)
        self.hyp = deepcopy(load_cfg(hyp))
        self.inplace = self.cfg.get('inplace', True)

        # Define model
        self.cfg['ch'] = self.cfg.get('ch', ch)  # input channels
        self.amp = self.cfg.get('amplification', None)
        if anchors:
            LOGGER.info(f'Overriding model.cfg anchors with anchors={anchors}')
            self.cfg['anchors'] = round(anchors)  # override yaml value

        # Parse modules from config, specify fpn-backbone and headers
        self.backbone, self.neck, self.headers = build_network(
            self.cfg, self.hyp, is_scripting=is_scripting)  # deepcopy(self.cfg)

        # Init weights, biases
        initialize_weights(self)
        # self.freeze(freeze)  # freeze if needed

        self.info()
        LOGGER.info('')

    def forward(self, x: torch.Tensor, targets=None, visualize=False, compute_masks=False):
        bs, ch, h, w = x.shape
        # image_shapes = [(h, w) for i in range(bs)]

        x = self.backbone(x)  #, visualize=visualize)
        task_features = self.neck(x)

        losses, outputs = {}, {}
        for task_id, header in self.headers.items():
            if targets is not None:
                # remove image without annotation under task_id
                task_gts, keep_idx = [], []
                for idx, _ in enumerate(targets):
                    if task_id in _['anns']:
                        task_gts.extend(_['anns'][task_id])
                        keep_idx.extend([idx] * len(_['anns'][task_id]))
                task_features = {k: fmap[keep_idx] for k, fmap in task_features.items()}
            else:
                task_gts = None
            task_losses, task_outputs = header(task_features, task_gts, compute_masks=compute_masks)
            losses[task_id] = task_losses
            # losses = {**losses, **{f'{task_id}/{k}': v for k, v in task_losses.items()}}
            outputs[task_id] = task_outputs

        outputs = [dict(zip(outputs.keys(), _)) for _ in zip(*outputs.values())]
        outputs = self.post_processing(outputs)

        return losses, outputs

    def post_processing(self, outputs: Any):
        return outputs

    def fuse(self):
        """ Fuse model Conv2d() + BatchNorm2d() layers.
            If model is half precision (float16) and want to run on cpu, 
            Don't fuse model, "sqrt_vml_cpu" not implemented for 'Half'.
        """
        LOGGER.info('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def freeze(self, layers=[]):
        freeze_params(self, layers)
        freeze_bn(self, layers)

        return self


class Deploy(nn.Module):
    def __init__(self, model, fuse=False):
        super().__init__()
        if fuse:
            model = deepcopy(model).fuse()   # fused model cannot be used on cpu
        self.backbone = torch.jit.script(model.backbone)
        self.neck = torch.jit.script(model.neck)
        self.headers = torch.jit.script(model.headers)  # example_inputs not working on 1.10.1

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor], compute_masks: bool = True):
        if torch.jit.isinstance(x, List[torch.Tensor]):
            x = torch.stack(x)
        bs = x.shape[0]  # don't use len(x), len == 1...
        x = self.backbone(x)
        task_features = self.neck(x)

        task_o: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        for task_id, header in self.headers.items():
            task_o[task_id] = header(task_features, compute_masks=compute_masks)[1]

        outputs: List[Dict[str, Dict[str, torch.Tensor]]] = []
        for idx in range(bs):
            img_o: Dict[str, Dict[str, torch.Tensor]] = {}
            for k, v in task_o.items():
                img_o[k] = v[idx]
            outputs.append(img_o)

        outputs = self.post_processing(outputs)

        return None, outputs
    
    def post_processing(self, x: Any):
        return x


class Ensemble(nn.ModuleList):
    # Ensemble of models, not finished, add masks support, add multi-label support
    def __init__(self, models, nms_params: Dict[str, float] = {}):
        super().__init__(models)
        self.nms_params: Dict[str, float] = self.get_nms_params(nms_params)
    
    def get_nms_params(self, args={}):
        default_args = {'conf_thres': 0.15, 'iou_thres': 0.45, 'max_det': 300}
        return {k: float(args.get(k, v)) for k, v in default_args.items()}
    
    def forward(self, x: Union[List[torch.Tensor], torch.Tensor], compute_masks: bool = True):
        if torch.jit.isinstance(x, List[torch.Tensor]):
            x = torch.stack(x)
        bs = x.shape[0]  # don't use len(x), len == 1...

        y = [module(x, compute_masks=compute_masks)[1] for module in self]
        outputs = [self.merge([_[idx] for _ in y]) for idx in range(bs)]

        return None, outputs

    def merge(self, x: List[Dict[str, Any]]):
        task_ids: Set = set().union(*x)  # jit don't recognize set...

        res: Dict[str, Dict[str, torch.Tensor]] = {}
        for task_id in task_ids:
            boxes, scores, labels, masks = [], [], [], []
            for r in x:
                if task_id in r:
                    boxes.append(r[task_id]['boxes'])
                    scores.append(r[task_id]['scores'])
                    labels.append(r[task_id]['labels'])
                    if 'masks' in r[task_id]:
                        masks.append(r[task_id]['masks'])
                    else:
                        masks.append(None)
            boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
            if all(m is None for m in masks):
                masks = None  # no masks
            else:
                m_shape, m_dtype, m_device = [(m.shape[1:], m.dtype, m.device) for m in masks if m is not None][0]
                masks = [torch.zeros(m_shape).to(m_device, m_dtype) if m is None else m for m in masks]
                masks = torch.cat(masks)

            # filter by score
            keep = scores > self.nms_params['conf_thres']
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if masks is not None:
                masks = masks[keep]

            if len(boxes):
                keep = torchvision.ops.nms(boxes, scores, self.nms_params['iou_thres'])[:int(self.nms_params['max_det'])]
                # print(self.nms_params['iou_thres'], len(boxes), len(keep))
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if masks is not None:
                    masks = masks[keep]
            res[task_id] = {'boxes': boxes, 'scores': scores, 'labels': labels,}
            if masks is not None:
                res[task_id]['masks'] = masks

        return res
