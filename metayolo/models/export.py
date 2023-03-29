import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

from .. import LOGGER, check_version, load_cfg

from .layers import *
from .utils_general import make_divisible, check_anchor_order, nms_per_image, non_max_suppression, xyxy2xywhn, xyxy2xywh
from .utils_torch import fuse_conv_and_bn, initialize_weights, model_info, scale_img, torch_meshgrid
# from .utils.plots import feature_visualization

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

ROI_ALIGN = False


def _convert_to_roi_format(boxes: List[torch.Tensor]) -> torch.Tensor:
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device) for i, b in enumerate(boxes)],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


class Detect(nn.Module):  # detection layer
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, ch, anchors, stride, nc, nc_masks=None, 
                 dim_reduced=256, mask_output_size=28, dilation=1, 
                 input_size=None, nms_params={}):
        super().__init__()
        self.inplace = True  # use in-place ops (e.g. slice assignment)
        assert len(ch) == len(anchors) == len(stride), "ch, anchors, stride should have same length."
        
        ## anchors, strides, channels
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.ch = ch  # number of channels
        self.input_size = input_size
        
        stride = torch.tensor(stride).float()  # [8.0, 16.0, 32.0, 64.0]
        anchors = torch.tensor(anchors).float().view(len(anchors), -1, 2) / stride.view(-1, 1, 1)  # shape(nl,na,2)
        self.register_buffer('stride', stride)
        self.register_buffer('anchors', anchors)
        
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        # self.grid = [torch.zeros(1)] * self.nl  # init grid, buffer to avoid calculating in every forward
        # self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid

        self.grid, self.anchor_grid = [], []
        for i in range(self.nl):
            if self.input_size:
                ny = nx = int(input_size/2**(3+i))
                grid, anchor_grid = self._make_grid(self.anchors[i], self.stride[i], nx, ny)
            else:
                grid, anchor_grid = None, None
            self.register_buffer('grid_{}'.format(i), grid)
            self.grid.append(getattr(self, 'grid_{}'.format(i)))
            self.register_buffer('anchor_grid_{}'.format(i), anchor_grid)
            self.anchor_grid.append(getattr(self, 'anchor_grid_{}'.format(i)))
        check_anchor_order(self)

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self._initialize_biases()  # initialize bias for yolo detect layer
        self.nms_params = self.get_nms_params(nms_params)
        for k, v in self.nms_params.items():
            setattr(self, k, v)

        ## Add instance segmentation header
        if nc_masks:
            # assert nc_masks in [True, 1, self.nc], "If specified, nc_masks need to be True, 1, or same as detection classes! "
            nc_masks = self.nc if nc_masks is True else nc_masks
            self.nc_masks = nc_masks  # number of classes for masks, same as nc, 1, or None

            # self.m = nn.ModuleList(Conv(x, dim_reduced, k=3, act=False) for x in in_channels)  # mask fuse
            # self.seg = nn.ModuleList(Conv(self.ch[i], (self.ch[i-1] if i > 0 else dim_reduced), k=3, act=False) 
            #                          for i in range(self.nl))
            self.seg = nn.ModuleList(Conv(self.ch[i], (self.ch[i-1] if i > 0 else dim_reduced), k=3, act=False) 
                                     for i in range(self.nl-1, -1, -1))  # top-down for easy scripting
            self.mask_output_size = mask_output_size
            
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
        
    def forward(self, x: List[torch.Tensor]):
        # 1. training: targets rois are required, return x, masks
        # 2. validation: targets rois are provided, return (output, x), masks
        # 3. inference: rois not provided, nms on output, return (output_filtered, x), masks_filtered

        # Run det header on feature map
        preds = []
        for i, layer_d in enumerate(self.m):
            f = layer_d(x[i])  # conv
            bs, _, ny, nx = f.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            det = f.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.input_size is not None:  # fixed size inputs
                grid, anchor_grid = self.grid[i], self.anchor_grid[i]
            else:  # dynamic size inputs
                grid, anchor_grid = self._make_grid(self.anchors[i], self.stride[i], nx, ny)
            y = det.sigmoid()
            # For exporting just use inplace=True, 
            # Not compatible with: for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
            y[..., 0:2] = (y[..., 0:2] * 2. + grid) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
            preds.append(y.view(bs, -1, self.no))

        outputs = nms_per_image(
            torch.cat(preds, 1), nc=self.nc, conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres, multi_label=self.multi_label, max_det=self.max_det,
        )
        if self.seg is not None:
            outputs = self.compute_masks(x, outputs)

        return outputs
    
    def compute_masks(self, x: List[torch.Tensor], outputs: List[Dict[str, torch.Tensor]]):
        # det_output has boxes + scores, anchor_features in inference
        f = x[-1]
        for i, layer_s in enumerate(self.seg, 1):
            if i == 1:
                f = layer_s(f)  # x[-i]
            else:
                h, w = x[-i].shape[-2], x[-i].shape[-1]
                f = F.interpolate(f, size=(h, w), mode='bilinear', align_corners=True)
                f = layer_s(f + x[-i])

        rois = [_['boxes'].detach() / self.stride[0] for _ in outputs]
        rois = _convert_to_roi_format(rois)  # no need for 1.10
        n_obj_per_image = [_['boxes'].shape[0] for _ in outputs]
        f = torchvision.ops.roi_align(f, rois, self.mask_output_size, 1.0, aligned=False)  # (n_roi, 256, 28, 28)
        mask_logits = self.seg_h(f)
        mask_probs = mask_logits.sigmoid().split(n_obj_per_image, dim=0)

        for r, m in zip(outputs, mask_probs):
            if self.nc_masks > 1:
                labels = r['labels']
                index = torch.arange(len(m), device=labels.device)
                m = m[index, labels - 1][:, None]
            r['masks'] = m
            # r['masks'] = paste_masks_in_image(m, boxes, img_shape, padding=1).squeeze(1)

        return outputs

    def _make_grid(self, anchors: torch.Tensor, stride: torch.Tensor, nx: int=20, ny: int=20):
        d, t = anchors.device, anchors.dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch_meshgrid(y, x)
        grid = torch.stack([xv, yv], 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
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


class YoloMask(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        """ Note that nms_params is not assigned. Need manually assign the best one for each header. """
        # model/cfg, loss_hyp, input channels, number of classes
        super().__init__()
        self.cfg = load_cfg(cfg)
        self.inplace = self.cfg.get('inplace', True)
        # self.nc = self.cfg['nc']
        # self.names = [str(i) for i in range(self.cfg['nc'])]  # default names

        # Define model
        ch = self.cfg['ch'] = self.cfg.get('ch', ch)  # input channels
        if anchors:
            LOGGER.info(f'Overriding model.cfg anchors with anchors={anchors}')
            self.cfg['anchors'] = round(anchors)  # override yaml value
        
        # Parse modules from config, specify fpn-backbone and headers
        modules, self.save = parse_model(deepcopy(self.cfg), ch=[ch])  # module_lists, savelist
        n1, n2, n3 = len(self.cfg['backbone']), len(self.cfg['fpn']), len(self.cfg['headers'])
        self.amp = self.cfg.get('amplification', None)
        self.backbone = nn.ModuleList(modules[:n1])
        self.fpn = nn.ModuleList(modules[n1:n1+n2])
        self.headers = nn.ModuleDict({m.tag: m for m in modules[n1+n2:]})

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x: List[torch.Tensor]):
        x = torch.stack(x)
        bs = x.shape[0]
        
        y: Dict[int, torch.Tensor] = {}  # intermediate layers
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

        task_o: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        for task_id, header in self.headers.items():
            task_features = y[header.f] if isinstance(header.f, int) else [y[j] for j in header.f]
            task_o[task_id] = header(task_features)

        outputs: List[Dict[str, Dict[str, torch.Tensor]]] = []
        for i in range(bs):
            img_o: Dict[str, Dict[str, torch.Tensor]] = {}
            for k, v in task_o.items():
                img_o[k] = v[i]
            outputs.append(img_o)

        # outputs = [dict(zip(task_o.keys(), _)) for _ in zip(*task_o.values())]
        # outputs = self.post_processing(outputs)

        return outputs

    
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



def parse_model(d, ch):  # model_dict, input_channels(3)
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
            m_ = m(*args, input_size=header_args, nms_params={})  # module
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


def load_export_cfg(cfg, input_size=None, compute_masks=True):
    # input_size is None will use dynamic input size
    cfg = load_cfg(cfg)
    n_headers = len(cfg['headers'])
    if not isinstance(compute_masks, list):
        compute_masks = [compute_masks] * n_headers
    
    for _, compute_mask in zip(cfg['headers'], compute_masks):
        if len(_) > 5:
            _[5] = input_size
        else:
            _.append(input_size)
        if not compute_mask:
            _[3][-1] = None
    
    return cfg