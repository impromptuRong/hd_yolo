# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
from torch import Tensor
from typing import Tuple, List, Dict, Optional, Union

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# if platform.system() != 'Windows':
#     ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from .. import LOGGER, check_version, load_cfg

from .layers import *
from .utils_general import make_divisible, check_anchor_order, nms_per_image, non_max_suppression, xyxy2xywhn, xyxy2xywh
from .utils_torch import fuse_conv_and_bn, initialize_weights, model_info, scale_img, torch_meshgrid
from .loss import DetLoss, SegLoss
# from .utils.plots import feature_visualization

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

ROI_ALIGN = False


class Detect(nn.Module):  # detection layer
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, ch, anchors, stride, nc, nc_masks=None, 
                 dim_reduced=256, mask_output_size=28, dilation=1, 
                 nms_params={}, loss_hyp={}):
        super().__init__()
        self.inplace = False  # use in-place ops (e.g. slice assignment)
        assert len(ch) == len(anchors) == len(stride), "ch, anchors, stride should have same length."

        ## anchors, strides, channels
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.ch = ch  # number of channels
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor

        stride = torch.tensor(stride).float()  # [8.0, 16.0, 32.0, 64.0]
        anchors = torch.tensor(anchors).float().view(len(anchors), -1, 2) / stride.view(-1, 1, 1)  # shape(nl,na,2)
        self.register_buffer('stride', stride)
        self.register_buffer('anchors', anchors)

        for idx in range(self.nl):
            # ny = nx = int(640/2**(3+i))
            # grid, anchor_grid = self._make_grid(self.anchors[i], self.stride[i], nx, ny)
            grid, anchor_grid = None, None
            self.register_buffer(f'grid_{idx}', grid)
            self.register_buffer(f'anchor_grid_{idx}', anchor_grid)
        check_anchor_order(self)

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self._initialize_biases()  # initialize bias for yolo detect layer

        # get losses
        autobalance = True
        ssi = list(self.stride).index(16)+1 if autobalance else 0  # stride 16 index layer
        self.det_loss = DetLoss(self.nc, self.nl, loss_hyp, ssi=ssi)
        self.seg_loss = SegLoss()
        self.nms_params = self.get_nms_params(nms_params)

        ## Add instance segmentation header
        if nc_masks:
            # assert nc_masks in [True, 1, self.nc], "If specified, nc_masks need to be True, 1, or same as detection classes! "
            self.nc_masks = self.nc if nc_masks is True else nc_masks  # number of classes for masks, same as nc, 1, or None
            self.mask_output_size = mask_output_size
            # self.m = nn.ModuleList(Conv(x, dim_reduced, k=3, act=False) for x in in_channels)  # mask fuse
            # self.seg = nn.ModuleList(Conv(self.ch[i], (self.ch[i-1] if i > 0 else dim_reduced), k=3, act=False) 
            #                          for i in range(self.nl))
            self.seg = nn.ModuleList(Conv(self.ch[i], (self.ch[i-1] if i > 0 else dim_reduced), k=3, act=False) 
                                     for i in range(self.nl-1, -1, -1))  # top-down for easy scripting

            # self.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
            # mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            # mask_features = self.mask_head(mask_features)
            # mask_logits = self.mask_predictor(mask_features)

            self.seg_h = nn.Sequential(OrderedDict([
                ("act", nn.SiLU(inplace=True)),
                ("seg_conv", Conv(dim_reduced, dim_reduced, k=3, act=True)),
                # ("seg_conv2", Conv(dim_reduced, dim_reduced, k=3, act=True)),
                # ("seg_conv3", Conv(dim_reduced, dim_reduced, k=3, act=True)),
                # ("seg_conv4", Conv(dim_reduced, dim_reduced, k=3, act=True)),
                # ("conv_mask", nn.Conv2d(dim_reduced, dim_reduced, kernel_size=3, stride=1, padding=dilation, dilation=dilation)),
                # ("conv_mask", nn.ConvTranspose2d(dim_reduced, dim_reduced, kernel_size=2, stride=2, padding=0)),
                # ("act", nn.SiLU(inplace=True)),
                ("mask_logits", nn.Conv2d(dim_reduced, nc_masks, 1, 1, 0, bias=True)),
            ]))
        else:
            self.nc_masks = 0
            self.seg = None
            self.seg_h = None
            self.mask_output_size = None
        
    def forward(self, x: List[torch.Tensor], targets=None, compute_masks=True):
        # 1. training: targets rois are required, return x, masks
        # 2. validation: targets rois are provided, return (output, x), masks
        # 3. inference: rois not provided, nms on output, return (output_filtered, x), masks_filtered

        # Run det header on feature map
        dets, preds = [], []
        # for i in range(self.nl):
        #     f = self.m[i](x[i])  # conv
        for i, layer in enumerate(self.m):
            f = layer(x[i])  # conv
            bs, _, ny, nx = f.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            det = f.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            dets.append(det)

            if targets is None:  # not self.training only apply for inference, we want eval model in training
                grid_i = getattr(self, f'grid_{i}')
                if grid_i is None or grid_i.shape[2:4] != det.shape[2:4] or self.onnx_dynamic:
                    grid_i, anchor_grid_i = self._make_grid(self.anchors[i], self.stride[i], nx, ny)
                    setattr(self, f'grid_{i}', grid_i)
                    setattr(self, f'anchor_grid_{i}', anchor_grid_i)
                grid, anchor_grid = getattr(self, f'grid_{i}'), getattr(self, f'anchor_grid_{i}')
                # if self.grid[i] is None or self.grid[i].shape[2:4] != det.shape[2:4] or self.onnx_dynamic:
                #     # grid, anchor_grid = self._make_grid(nx, ny, i)
                #     self.grid[i], self.anchor_grid[i] = self._make_grid(self.anchors[i], self.stride[i], nx, ny)
                # grid, anchor_grid = self.grid[i], self.anchor_grid[i]

                y = det.sigmoid()
                if self.inplace:  # inplace cause gradient problem, don't use
                    y[..., 0:2] = (y[..., 0:2] * 2. + grid) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.tensor_split((2, 4), -1)  # torch 1.8.0
                    xy = (xy * 2 + grid) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * anchor_grid  # wh
                    y = torch.cat((xy, wh, conf), -1)
                preds.append(y.view(bs, -1, self.no))
            # else:
                # preds.append(f.new_empty(size=(bs, 0, self.no)))  # avoid torch.cat error        

        compute_masks *= (self.seg is not None)
        if compute_masks > 0.:  # Run seg header on feature map
            for i, layer in enumerate(self.seg, 1):
                if i == 1:
                    f = layer(x[-i])
                else:
                    h, w = x[-i].shape[-2], x[-i].shape[-1]
                    f = F.interpolate(f, size=(h, w), mode='bilinear', align_corners=True)
                    f = layer(f + x[-i])

#             for i in range(self.nl-1, -1, -1):
#                 if i == self.nl-1:
#                     f = self.seg[i](x[i])
#                 else:
#                     h, w = x[i].shape[-2], x[i].shape[-1]
#                     f = F.interpolate(f, size=(h, w), mode='bilinear', align_corners=True)
#                     f = self.seg[i](f + x[i])

        ## compute losses and outputs
        if self.training:
            assert targets is not None
            compute_loss, compute_outputs = True, False
        else:
            compute_loss, compute_outputs = targets is not None, True

        if compute_loss:
            losses = self.compute_loss(dets, f, targets, compute_masks=compute_masks)
        else:
            losses = {}

        if compute_outputs:
            outputs = self.compute_outputs(preds, f, compute_masks=compute_masks)
        else:
            outputs = []

        return losses, outputs
    
    def compute_loss(self, dets, f, targets, compute_masks=True):
        # det loss
        gts = [torch.cat([torch.full_like(_['labels'][:, None], idx), _['labels'][:, None], 
                          xyxy2xywh(_['boxes'], clip=True, eps=0.0)], -1)
               for idx, _ in enumerate(targets)]
        tcls, tbox, indices, anchors = self.build_targets(dets, torch.cat(gts))  # targets, tcls start from 1
        det_loss, det_loss_items = self.det_loss(dets, tcls, tbox, indices, anchors)

        # seg loss
        if compute_masks > 0.:
            bs, nc, h, w = f.shape
            rois = [_['boxes'] * _['boxes'].new([w, h, w, h]) for _ in targets]
            f = torchvision.ops.roi_align(f, rois, self.mask_output_size, 1.0, aligned=ROI_ALIGN)  # (n_roi, 256, 28, 28)
            mask_logits = self.seg_h(f)

            gt_masks = torch.cat([_['masks'] for _ in targets])[:, None] * 1.0
            bs, nc, h, w = gt_masks.shape
            gt_boxes = torch.cat([_['boxes'] * _['boxes'].new([w, h, w, h]) for _ in targets])
            # gt_labels = torch.cat([_['labels'] - (_['labels']>0).type(_['labels'].dtype) for _ in targets])
            gt_labels = torch.cat([_['labels'] for _ in targets])
            mask_loss = self.seg_loss(mask_logits, gt_boxes, gt_masks, gt_labels, mask_patches=None)
        else:
            mask_loss = torch.zeros_like(det_loss)

        return {'det_loss': det_loss, 'mask_loss': mask_loss, 'loss_items': {**det_loss_items, 'mask': mask_loss.detach()}}

    def compute_outputs(self, preds, f, compute_masks=True):
        # det_output has boxes + scores, anchor_features in inference
        outputs = nms_per_image(torch.cat(preds, 1), nc=self.nc, **self.nms_params)
        
        if compute_masks > 0.:
            rois = [_['boxes'].detach() / self.stride[0] for _ in outputs]
            n_obj_per_image = [_['boxes'].shape[0] for _ in outputs]
            f = torchvision.ops.roi_align(f, rois, self.mask_output_size, 1.0, aligned=ROI_ALIGN)  # (n_roi, 256, 28, 28)
            mask_logits = self.seg_h(f)
            mask_probs = mask_logits.sigmoid().split(n_obj_per_image, dim=0)
            
            for r, m in zip(outputs, mask_probs):
                if self.nc == self.nc_masks:
                    labels = r['labels']
                    index = torch.arange(len(m), device=labels.device)
                    m = m[index, labels - 1][:, None]  # r['labels'] starts from 1
                r['masks'] = m
                # r['masks'] = paste_masks_in_image(m, boxes, img_shape, padding=1).squeeze(1)

        return outputs

    def build_targets(self, p, targets):
        device = targets.device
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                           ], device=device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i].to(device)
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.det_loss.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#                 print(f"For layer {i}/{self.nl}: matched {t.shape} targets")
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


    def _make_grid(self, anchors, stride, nx=20, ny=20):
        d, t = anchors.device, anchors.dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch_meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (anchors * stride).view((1, self.na, 1, 1, 2)).expand(shape)

        return grid, anchor_grid

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        for mi in self.m:  # from
            b = mi.bias.detach().view(self.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def get_nms_params(self, args={}):
        default_args = {'conf_thres': 0.15, 'iou_thres': 0.45, 'multi_label': False, 'max_det': 300}
        return {k: args.get(k, v) for k, v in default_args.items()}

    def merge_outputs(self, r):
        """ Merge outputs from different rois under same amp. 
            rois and outputs should under same amplification.
        """
        boxes = torch.cat([_['boxes'] + _['boxes'].new([_['roi'][0], _['roi'][1], _['roi'][0], _['roi'][1]]) for _ in r])
        labels = torch.cat([_['labels'] for _ in r])
        scores = torch.cat([_['scores'] for _ in r])
        res = {'boxes': boxes, 'labels': labels, 'scores': scores}
        
        if 'masks' in r[0]:
            res['masks'] = torch.cat([_['masks'] for _ in r])

        return res
    
    def rescale_outputs(self, r, scale=1.0):
        """ Rescale outputs to another amplification. """
        if scale != 1.0:
            r['boxes'] *= scale
        
        return r


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', hyp='./hyp.scratch.yaml', ch=3, anchors=None):
        # model/cfg, loss_hyp, input channels, number of classes
        super().__init__()
        self.cfg = load_cfg(cfg)
        hyp = deepcopy(load_cfg(hyp))
        self.inplace = self.cfg.get('inplace', True)

        # Define model
        ch = self.cfg['ch'] = self.cfg.get('ch', ch)  # input channels
        if anchors:
            LOGGER.info(f'Overriding model.cfg anchors with anchors={anchors}')
            self.cfg['anchors'] = round(anchors)  # override yaml value

        # Parse modules from config, specify fpn-backbone and headers
        modules, self.save = parse_model(deepcopy(self.cfg), ch=[ch], hyp=hyp)
        n1, n2, n3 = len(self.cfg['backbone']), len(self.cfg['fpn']), len(self.cfg['headers'])
        self.amp = self.cfg.get('amplification', None)
        self.backbone = nn.ModuleList(modules[:n1])
        self.fpn = nn.ModuleList(modules[n1:n1+n2])
        self.headers = nn.ModuleDict({m.tag: m for m in modules[n1+n2:]})

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, targets=None, augment=False, profile=False, visualize=False, compute_masks=False):
        if augment:
            LOGGER.warning("***** augment inference is not ready yet.")
            return self._forward_once(x)
            # return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, targets, profile, visualize, compute_masks)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x: torch.Tensor, targets=None, profile=False, visualize=False, compute_masks=False):
        y: Dict[int, torch.Tensor] = {}  # intermediate layers
        bs, ch, h, w = x.shape

        # forward backbones
        for m_b in self.backbone:
            if m_b.f != -1:  # if not from previous layer
                if isinstance(m_b.f, int):
                    x = m_b(y[m_b.f])
                else:
                    x = m_b([x if j == -1 else y[j] for j in m_b.f])
            else:
                x = m_b(x)
            if m_b.i in self.save:
                y[-1] = y[m_b.i] = x
#             if visualize:
#                 feature_visualization(x, m_p.type, m_p.i, save_dir=visualize)

        # forward fpn and rescale feature map sizes
        for m_p in self.fpn:
            if m_p.f != -1:  # if not from previous layer
                if isinstance(m_p.f, int):
                    x = m_p(y[m_p.f])
                else:
                    x = m_p([x if j == -1 else y[j] for j in m_p.f])
            else:
                x = m_p(x)
            if m_p.i in self.save:
                y[-1] = y[m_p.i] = x
#             if visualize:
#                 feature_visualization(x, m_p.type, m_p.i, save_dir=visualize)
        
        losses, outputs = {}, {}
        for task_id, header in self.headers.items():
            task_features = y[header.f] if isinstance(header.f, int) else [y[j] for j in header.f]
            if targets is not None:
                # remove image without annotation under task_id
                task_gts, keep_idx = [], []
                for idx, _ in enumerate(targets):
                    if task_id in _['anns']:
                        task_gts.extend(_['anns'][task_id])
                        keep_idx.extend([idx] * len(_['anns'][task_id]))
                task_features = [fmap[keep_idx] for fmap in task_features]
            else:
                task_gts = None
            task_losses, task_outputs = header(task_features, task_gts, compute_masks)
            losses[task_id] = task_losses
            # losses = {**losses, **{f'{task_id}/{k}': v for k, v in task_losses.items()}}
            outputs[task_id] = task_outputs
        
        outputs = [dict(zip(outputs.keys(), _)) for _ in zip(*outputs.values())]
        outputs = self.post_processing(outputs)
        
        return losses, outputs
    
    def post_processing(self, outputs):
        return outputs
    
    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

#     def _profile_one_layer(self, m, x, dt):
#         c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
#         o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
#         t = time_sync()
#         for _ in range(10):
#             m(x.copy() if c else x)
#         dt.append((time_sync() - t) * 100)
#         if m == self.model[0]:
#             LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
#         LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
#         if c:
#             LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")


#     def _print_weights(self):
#         for m in self.model.modules():
#             if type(m) is Bottleneck:
#                 LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
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

#     def _apply(self, fn):
#         # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
#         self = super()._apply(fn)
#         m = self.model[-1]  # Detect()
#         if isinstance(m, Detect):
#             m.stride = fn(m.stride)
#             m.grid = list(map(fn, m.grid))
#             if isinstance(m.anchor_grid, list):
#                 m.anchor_grid = list(map(fn, m.anchor_grid))
#         return self


def parse_model(d, ch, hyp):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, gd, gw = d['anchors'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # nc = d['nc']
    # no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, _ in enumerate(d['backbone'] + d['fpn'] + d['headers']):
        f, n, m, args = _[0], _[1], _[2], _[3]  # from, number, module, args, (tag)
        tag = _[4] if len(_) > 4 else None
        header_args = _[5] if len(_) > 5 else None  # header only
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m is Detect:
            args = [[ch[x] for x in f]] + args
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            # get task header hyperparameters
            tag = tag or 'det'
            loss_keys = ['box', 'cls', 'cls_pw', 'cls_cw', 'obj', 'obj_pw', 'iou_t', 
                         'anchor_t', 'fl_gamma', 'label_smoothing',]
            nms_keys = ['conf_thres', 'iou_thres', 'multi_label', 'max_det']
            loss_hyp = {k: hyp[tag][k] for k in loss_keys if k in hyp[tag]}
            nms_params = {k: hyp[tag][k] for k in nms_keys if k in hyp[tag]}
            m_ = m(*args, nms_params=nms_params, loss_hyp=loss_hyp)  # module
        else:
            if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                     BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
                c1, c2 = ch[f], args[0]
                # if c2 != no:  # if not output, this won't happen as I switch nc in Detect to 3rd place
                c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2
            elif m is Expand:
                c2 = ch[f] // args[0] ** 2
            else:
                c2 = ch[f]
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np, m_.tag = i, f, t, np, tag  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return layers, sorted(save)


class Ensemble(nn.ModuleList):
    # Ensemble of models, not finished
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            _, outputs = module(x, augment, profile, visualize)
            y.append(outputs)
        return None, y


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
#     parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--profile', action='store_true', help='profile model speed')
#     parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
#     parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
#     opt = parser.parse_args()
#     opt.cfg = check_yaml(opt.cfg)  # check YAML
#     print_args(vars(opt))
#     device = select_device(opt.device)

#     # Create model
#     im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
#     model = Model(opt.cfg).to(device)

#     # Options
#     if opt.line_profile:  # profile layer by layer
#         _ = model(im, profile=True)

#     elif opt.profile:  # profile forward-backward
#         results = profile(input=im, ops=[model], n=3)

#     elif opt.test:  # test all models
#         for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
#             try:
#                 _ = Model(cfg)
#             except Exception as e:
#                 print(f'Error in {cfg}: {e}')
