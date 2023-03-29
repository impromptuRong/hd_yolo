import torch
import torch.nn as nn
import math

from .layers import *
from .utils_general import nms_per_image, xyxy2xywhn, xyxy2xywh, xywh2xyxy, paired_box_iou
from .utils_torch import torch_meshgrid, one_hot_labels
from .loss_yolov6 import ComputeLoss


class Detect(nn.Module):
    """ Yolov6 Efficient Decoupled Head. """
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, ch, anchors, stride, nc, masks={}, 
                 dim_reduced=256, mask_output_size=28, dilation=1, 
                 nms_params={}, loss_hyp={}):
    # def __init__(self, num_classes=80, anchors=1, stride=[8, 16, 32], num_layers=3, head_layers=None):
        super().__init__()
        self.inplace = False
        assert len(ch) == len(stride), f"ch, anchors, stride should have same length."

        ## anchors, strides, channels
        self.ch = ch  # number of channels
        self.nl = len(ch)  # number of detection layers
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        if isinstance(anchors, (list, tuple)):  # number of anchors
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.na = 1
        
        stride = torch.tensor(stride).float()  # [8.0, 16.0, 32.0, 64.0]
        self.register_buffer('stride', stride)
        # self.anchors = anchors  # not used
        
        # self.grid = [torch.zeros(1)] * num_layers
        for idx in range(self.nl):
            self.register_buffer(f'grid_{idx}', None)

        # Efficient decoupled head layers
        self.stems = nn.ModuleList(Conv(x, x, kernel_size=1, stride=1, act=True) for x in self.ch)
        self.cls_convs = nn.ModuleList(Conv(x, x, kernel_size=3, stride=1, act=True) for x in self.ch)
        self.reg_convs = nn.ModuleList(Conv(x, x, kernel_size=3, stride=1, act=True) for x in self.ch)
        self.cls_preds = nn.ModuleList(nn.Conv2d(x, self.nc * self.na, kernel_size=1) for x in self.ch)
        self.reg_preds = nn.ModuleList(nn.Conv2d(x, 4 * self.na, kernel_size=1) for x in self.ch)
        self.obj_preds = nn.ModuleList(nn.Conv2d(x, 1 * self.na, kernel_size=1) for x in self.ch)
        self.initialize_biases()

        # get losses
        self.det_loss = ComputeLoss(
            reg_weight=5.0,
            iou_weight=3.0,
            cls_weight=1.0,
            center_radius=2.5,
            eps=1e-7,
            in_channels=self.ch,
            strides=self.stride,
            n_anchors=self.na,
            iou_type='ciou',
        )
        self.nms_params = self.get_nms_params(nms_params)

        # no masks for now
        self.nc_masks = 0
        self.mask_output_size = None
        self.seg = None
        self.seg_h = None
        # self.seg_loss = SegLoss()

    def get_nms_params(self, args={}):
        default_args = {'conf_thres': 0.15, 'iou_thres': 0.45, 'multi_label': False, 'max_det': 300}
        return {k: args.get(k, v) for k, v in default_args.items()}        

    def initialize_biases(self, prior_prob=1e-2):
        for conv in self.cls_preds:
            b = conv.bias.view(self.na, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.na, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: Dict[int, torch.Tensor], targets=None, compute_masks=True):
        # 1. training: targets rois are required, return x, masks
        # 2. validation: targets rois are provided, return (output, x), masks
        # 3. inference: rois not provided, nms on output, return (output_filtered, x), masks_filtered

        x = x[self.f] if isinstance(self.f, int) else [x[j] for j in self.f]

        if self.training:
            assert targets is not None
            compute_losses, compute_outputs = True, False
        else:
            compute_losses, compute_outputs = targets is not None, True
        compute_masks = (self.nc_masks > 0) * compute_masks

        # Run det header on feature map
        dets = []
        for i in range(self.nl):
            f = self.stems[i](x[i])
            cls_feat = self.cls_convs[i](f)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](f)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)
            
            det = torch.cat([reg_output, obj_output, cls_output], 1)
            bs, _, ny, nx = det.shape
            det = det.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            dets.append(det)

        # calculate predictions
        if self.training and not compute_masks:  # only in training and ignore masks, we don't need to compute proposals
            preds = []
        else:
            preds = self.compute_proposals(dets)  # [(bs, 20x20, xywhcoefs), (bs, 40x40, xywhcoefs), ...]

        if compute_masks > 0.:  # Run seg header on feature map
            panoptic_feature_maps = []
            for i, layer in enumerate(self.seg, 1):
                panoptic_feature_maps.append(layer(x[-i]))
            panoptic_feature_maps = panoptic_feature_maps[::-1]
        else:
            panoptic_feature_maps = None

        ## compute losses and outputs
        if compute_losses:
            losses = self.compute_losses(dets, preds, panoptic_feature_maps, targets, compute_masks=compute_masks)
        else:
            losses = {}

        if compute_outputs:
            outputs = self.compute_outputs(preds, panoptic_feature_maps, compute_masks=compute_masks)
        else:
            outputs = []

        return losses, outputs

    def compute_proposals(self, dets):
        preds = []
        for i in range(self.nl):
            y = dets[i]

            # update grid, actually not upgrade, need a better way
            grid_i = getattr(self, f'grid_{i}')
            if grid_i is None or grid_i.shape[2:4] != y.shape[2:4] or self.onnx_dynamic:
                bs, na, ny, nx, no = y.shape
                grid = self._make_grid(nx, ny)
                # setattr(self, f'grid_{i}', grid)
            else:
                grid = getattr(self, f'grid_{i}')  # not used

            if self.inplace:
                y[..., 0:2] = (y[..., 0:2] + grid) * self.stride[i]  # xy
                y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i] # wh
                y[..., 4:] = y[..., 4:].sigmoid()
            else:
                xy, wh, conf = y.tensor_split((2, 4), -1)  # torch 1.8.0
                xy = (xy + grid) * self.stride[i]  # xy
                wh = torch.exp(wh) * self.stride[i]  # wh
                y = torch.cat((xy, wh, conf.sigmoid()), -1)
            preds.append(y)  # y.view(bs, -1, self.no)
        # print(f"compute_proposals", [y.shape for y in preds])

        return preds

    def compute_losses(self, dets, preds, features, targets, compute_masks=True):
        # gts: [img_id, label, x, y, w, h]
        gts = [torch.cat([torch.full_like(_['labels'][:, None], idx), _['labels'][:, None], 
                          xyxy2xywh(_['boxes'], clip=True, eps=0.0)], -1)
               for idx, _ in enumerate(targets)]

        det_loss, det_loss_items = self.det_loss(dets, torch.cat(gts))

        return {'det_loss': det_loss, 'mask_loss': 0., 'loss_items': {**det_loss_items, 'mask': {}}}

    def compute_outputs(self, preds, features, compute_masks=True):
        # det_output has boxes + scores, anchor_features in inference
        preds = torch.cat([F.pad(y.view(y.shape[0], -1, self.no), [0,1], value=idx) 
                           for idx, y in enumerate(preds)], 1)
        outputs = nms_per_image(preds, nc=self.nc, **self.nms_params) # [[x0, y0, x1, y1, label, score, extra:],]
        results = [{'boxes': _[:, :4], 'labels': _[:, 4] + 1, 'scores': _[:, 5],} for _ in outputs]  # 'extra': _[:,6:]

        return results

    def _make_grid(self, nx=20, ny=20):
        d, t = self.stride.device, self.stride.dtype
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch_meshgrid(y, x)
        # grid = torch.stack((xv, yv), 2).view(1, self.na, ny, nx, 2).float()
        grid = torch.stack((xv, yv), 2).expand(1, self.na, ny, nx, 2).float()
        
        return grid

