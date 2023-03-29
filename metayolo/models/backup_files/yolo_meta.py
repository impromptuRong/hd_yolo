# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Rewrite Yolo for multi-task
Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
import numbers
import time

from copy import deepcopy
from pathlib import Path
from collections import OrderedDict

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# # ROOT = ROOT.relative_to(Path.cwd())  # relative

from .. import LOGGER, check_version, load_cfg

from .layers import *
from .utils_general import make_divisible, check_anchor_order, nms_per_image, xyxy2xywhn, xyxy2xywh
from .utils_torch import fuse_conv_and_bn, initialize_weights, model_info, scale_img
from .loss import DetLoss, SegLoss
from .utils_o import mosaic_roi_feature_maps, sliding_window_scanner, split_by_sizes
# from .utils.plots import feature_visualization


try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

ROI_ALIGN = False

# from .utils_image import *
# COLORS = {
#     "tumor": [0, 255, 0],
#     "stroma": [255, 0, 0],
#     "lymphocyte": [0, 0, 255],
#     "blood": [255, 0, 255],
#     "macrophage": [255, 255, 0],
#     "dead": [0, 148, 225],
#     "ductal": [100, 0, 255],
#     "unlabeled": [148, 148, 148],
# }

# CLASSES = ["tumor", "stroma", "lymphocyte", "macrophage", "dead", "ductal", "blood", ]  # "unlabeled"
# LABELS_COLOR = {class_id: np.array(COLORS[name])/255 for class_id, name in enumerate(CLASSES)}
# LABELS_COLOR[-100] = np.array(COLORS['unlabeled'])/255
# LABELS_TEXT = {**{i: _ for i, _ in enumerate(CLASSES)}, -100: "unlabeled"}


# class DynamicFeaturePyramidNetwork(FeaturePyramidNetwork):
#     """ FeaturePyramidNetwork on given ROI.
#         rescale fpn feature map will cause problem for anchor-based detections.
#         So rescale raw feature map before FPN.
#     """
#     def get_result_from_inner_blocks(self, x, idx, boxes, roi_size):
#         out = torchvision.ops.roi_align(x, boxes, roi_size, aligned=True)
# #         out = [F.interpolate(_[None, :, int(x0):int(x1), int(y0):int(y1)], size=roi_size, mode="nearest")[0]
# #                for _, rois in zip(x, boxes) for y0, x0, y1, x1 in rois]
# #         out = torch.stack(out)
        
#         return super().get_result_from_inner_blocks(out, idx)
    
#     def forward(self, x, image_shape, roi_size, feature_maps=None, targets=None):
#         """
#             Computes the FPN for a set of feature maps.
#             Args:
#                 x (Tuple, List): feature maps for each feature level.
#                 image_shape (Tuple):  the input image shape (img_h_after_transform, img_w_after_transform).
#                 roi_size (Tuple): specific to task_id, (roi_h_after_transform, roi_w_after_transform).
#                 feature_maps: (OrderedDict[str, int]): feature_map names -> indices mapping.
#                 targets (Optional, List[List[Dict]]): list of annotations.
#             Returns:
#                 results (OrderedDict[Tensor]): feature maps after FPN layers.
#                     They are ordered from highest resolution first.
#         """
#         ## calculate scale_factors for each feature map:
#         roi_sizes, roi_boxes = [], []
#         for f in x:
#             scale_h, scale_w = f.shape[2]/image_shape[0], f.shape[3]/image_shape[1]
#             roi_sizes.append((int(roi_size[0] * scale_h), int(roi_size[1] * scale_w)))
#             ## calculate roi on each feature map
#             if targets is not None:
#                 boxes = [torch.stack([ann['roi'] for ann in t]) * f.new([scale_w, scale_h, scale_w, scale_h]) 
#                          for t in targets]
#             else:
#                 boxes = None
#             roi_boxes.append(boxes)
        
#         ## Last feature layer
#         results = []
#         last_inner = self.get_result_from_inner_blocks(x[-1], -1, roi_boxes[-1], roi_sizes[-1])
#         results.append(self.get_result_from_layer_blocks(last_inner, -1))
        
#         for idx in range(len(x) - 2, -1, -1):
#             inner_lateral = self.get_result_from_inner_blocks(x[idx], idx, roi_boxes[idx], roi_sizes[idx])
#             feat_shape = inner_lateral.shape[-2:]
#             inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
#             last_inner = inner_lateral + inner_top_down
#             results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))            
        
#         task_features = OrderedDict({k: results[v] for k, v in feature_maps.items()})
#         task_targets = [ann for t in targets for ann in t]
        
#         return task_features, task_targets


class Segment(nn.Module):
    def __init__(self, in_channels, nc, feature_layers, 
                 dim_reduced=256, output_size=28, dilation=1, 
                 stride=None, inplace=False):
        super().__init__()
        self.nc = nc  # number of classes
        self.in_channels = in_channels
        self.f = feature_layers
        self.nl = len(in_channels)  # number of detection layers
        # self.m = nn.ModuleList(Conv(x, dim_reduced, k=3, act=False) for x in in_channels)  # mask fuse
        self.m = nn.ModuleList(Conv(in_channels[i], (in_channels[i-1] if i > 0 else dim_reduced), k=3, act=False) 
                               for i in range(self.nl))
        self.output_size = output_size
        self.register_buffer('stride', stride)  # shape(nl,na,2)
        self.inplace = inplace
        
        self.header = nn.Sequential(OrderedDict([
            ("act", nn.SiLU(inplace=True)),
            ("seg_conv", Conv(dim_reduced, dim_reduced, k=3, act=True)),
            # ("seg_conv2", Conv(dim_reduced, dim_reduced, k=3, act=True)),
            # ("seg_conv3", Conv(dim_reduced, dim_reduced, k=3, act=True)),
            # ("seg_conv4", Conv(dim_reduced, dim_reduced, k=3, act=True)),
            # ("conv_mask", nn.Conv2d(dim_reduced, dim_reduced, kernel_size=3, stride=1, padding=dilation, dilation=dilation)),
            # ("conv_mask", nn.ConvTranspose2d(dim_reduced, dim_reduced, kernel_size=2, stride=2, padding=0)),
            # ("act", nn.SiLU(inplace=True)),
            ("mask_logits", nn.Conv2d(dim_reduced, nc, 1, 1, 0, bias=True)),
        ]))

    def forward(self, x, anchors):
        for i in range(self.nl-1, -1, -1):
            if i == self.nl-1:
                f = self.m[i](x[i])
            else:
                h, w = x[i].shape[-2], x[i].shape[-1]
                f = torch.nn.functional.interpolate(f, size=(h, w), mode='bilinear', align_corners=True)
                f = self.m[i](f + x[i])
        
        rois = [_ / self.stride[0] for _ in anchors]  # rois: (bs, [n_bbox, 4])
        f = torchvision.ops.roi_align(f, rois, self.output_size, 1.0, aligned=ROI_ALIGN)  # (n_roi, 256, 28, 28)
        
        return self.header(f)

#     def forward(self, x, anchors):
#         rois = [_ / self.stride[0] for _ in anchors]  # rois: (bs, [n_bbox, 4])
#         f = torchvision.ops.roi_align(x[0], rois, self.output_size, 1.0, aligned=ROI_ALIGN)  # (n_roi, 256, 28, 28)
        
#         return self.header(f)

#     def forward_1(self, x, anchors):
#         # for train, directly use anchors and masks from gt. 
#         # for inference, use predicted anchors after nms.
#         for i in range(self.nl):
#             f = self.m[i](x[i])  # f: (bs, channel, h, w)
#             rois = [_ / self.stride[i] for _ in anchors]  # rois: (bs, [n_bbox, 4])
#             # print(f.shape, self.stride[i])
#             x[i] = torchvision.ops.roi_align(f, rois, self.output_size, 1.0, aligned=ROI_ALIGN)  # (n_roi, 256, 28, 28)
        
#         return self.header(sum(x))


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
            self.register_buffer('grid_{}'.format(i), None)
            self.grid.append(getattr(self, 'grid_{}'.format(i)))
            self.register_buffer('anchor_grid_{}'.format(i), None)
            self.anchor_grid.append(getattr(self, 'anchor_grid_{}'.format(i)))
        check_anchor_order(self)
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self._initialize_biases()  # initialize bias for yolo detect layer
        self.det_loss = DetLoss(self, loss_hyp, autobalance=False)
        self.seg_loss = SegLoss()
        self.nms_params = self.get_nms_params(nms_params)
        
        ## Add instance segmentation header
        if nc_masks:
            assert nc_masks in [True, 1, self.nc], "If specified, nc_masks need to be True, 1, or same as detection classes! "
            nc_masks = self.nc if nc_masks is True else nc_masks
            self.nc_masks = nc_masks  # number of classes for masks, same as nc, 1, or None
            
            # self.m = nn.ModuleList(Conv(x, dim_reduced, k=3, act=False) for x in in_channels)  # mask fuse
            self.seg = nn.ModuleList(Conv(self.ch[i], (self.ch[i-1] if i > 0 else dim_reduced), k=3, act=False) 
                                     for i in range(self.nl))
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
            self.nc_masks = None
            self.seg = None
            self.seg_h = None
            self.mask_output_size = None
        
    def forward(self, x, targets=None, calculate_masks=True):
        # 1. training: targets rois are required, return x, masks
        # 2. validation: targets rois are provided, return (output, x), masks
        # 3. inference: rois not provided, nms on output, return (output_filtered, x), masks_filtered

        # Run det header on feature map
        dets, preds = [], []
        for i in range(self.nl):
            f = self.m[i](x[i])  # conv
            bs, _, ny, nx = f.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            dets.append(f.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous())

            if targets is None:  # not self.training only apply for inference, we want eval model in training
                if self.grid[i] is None or self.grid[i].shape[2:4] != dets[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = dets[i].sigmoid()
                if self.inplace:  # inplace cause gradient problem, don't use
                    y[..., 0:2] = (y[..., 0:2] * 2. + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.tensor_split((2, 4, 5), -1)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), -1)
                preds.append(y.view(bs, -1, self.no))
            # else:
                # preds.append(f.new_empty(size=(bs, 0, self.no)))  # avoid torch.cat error
        
        if targets is not None:
            gts = [torch.cat([torch.full_like(_['labels'][:, None], idx), _['labels'][:, None], xyxy2xywh(_['boxes'])], -1)
                   for idx, _ in enumerate(targets)]
            # calculate loss
            det_loss, det_loss_items = self.det_loss(dets, torch.cat(gts))
        
        compute_masks = self.seg is not None and calculate_masks
        if compute_masks:  # Run seg header on feature map
            for i in range(self.nl-1, -1, -1):
                if i == self.nl-1:
                    f = self.seg[i](x[i])
                else:
                    h, w = x[i].shape[-2], x[i].shape[-1]
                    f = torch.nn.functional.interpolate(f, size=(h, w), mode='bilinear', align_corners=True)
                    f = self.seg[i](f + x[i])

        ## compute losses and outputs
        if self.training:
            assert targets is not None
            compute_loss, compute_outputs = True, False
        else:
            compute_loss, compute_outputs = targets is not None, True

        if targets is None:  # inference only, use nms on results
            # det_output has boxes + scores, anchor_features in inference
            preds = nms_per_image(torch.cat(preds, 1), **self.nms_params)

            if calculate_masks:
                rois = [_['boxes'].detach() / self.stride[0] for _ in preds]
                n_obj_per_image = [len(_['boxes']) for _ in preds]
                f = torchvision.ops.roi_align(f, rois, self.mask_output_size, 1.0, aligned=ROI_ALIGN)  # (n_roi, 256, 28, 28)
                mask_logits = self.seg_h(f)
                mask_probs = mask_logits.sigmoid().split(n_obj_per_image, dim=0)
            else:
                mask_probs = [None] * len(preds)

            outputs = []
            for r, m in zip(preds, mask_probs):
                if m is not None:
                    if self.nc == self.nc_masks:
                        index = torch.arange(len(m), device=labels.device)
                        m = m[index, labels][:, None]
                    r['masks'] = m
                    # r['masks'] = paste_masks_in_image(m, boxes, img_shape, padding=1).squeeze(1)
                outputs.append(r)

            return None, outputs
        else:
            if calculate_masks:
                # rois = targets['rois']
                # assert rois is not None, "rois can't be empty in training mode!"
                # rois = [_ / self.stride[0] for _ in rois]  # rois: (bs, [n_bbox, 4])
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
                mask_loss = 0.0

            return {'det_loss': det_loss, 'det_loss_items': det_loss_items, 'mask_loss': mask_loss}, None

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
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
        default = {'conf_thres': 0.25, 'iou_thres': 0.45, 'multi_label': False, 'max_det': 300,}
        return {**default, **args}

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


class ConstrainModulev2(nn.Module):
    def __init__(self, cst, gamma=5.0):
        super(ConstrainModulev2, self).__init__()
        self.cst = cst
        self.graph = self.build_graph(cst)
        self.gamma = gamma
    
    def build_graph(self, x):
        nodes, edges, values = [], [], []
        if isinstance(x, dict):  # {(in_node, out_node): weight}
            x = [[node1, node2, v] for (node1, node2), v in x.items()]  # [[in_node, out_node, weight]]
        
        for entry in x:
            node1, node2 = entry[0], entry[1]
            if node1 not in nodes:
                nodes.append(node1)
            if node2 not in nodes:
                nodes.append(node2)
            values.append(entry[2] if len(entry) > 2 else 1.0)
            edges.append((nodes.index(node1), nodes.index(node2)))
        
        comp_edges, comp_values = [], []
        for (node1, node2), val in zip(edges, values):
            if (node2, node1) not in edges:
                comp_edges.append((node2, node1))
                comp_values.append(val)

        return {'nodes': nodes, 'edges': edges + comp_edges, 'values': values + comp_values}
    
    def get_output_rois_and_masks(self, r, c_label):
        assert 'labels' in r and 'size' in r, "We add `labels`, `size` slots to all header outputs." 
        
        roi_h, roi_w = r['size'][0], r['size'][1]
        indices = r['labels'] == c_label

        if 'boxes' in r:  # instances related 
            boxes = r['boxes'][indices]
            scores = r['scores'][indices] if 'scores' in r else torch.ones_like(r['labels'][indices])
            masks = r['masks'][indices] if 'masks' in r else torch.ones((len(scores), 1, 28, 28)).to(scores.device)
            masks = masks * scores[:, None, None, None] 
            im_rois = boxes
            
            # im_mask = 0. # boxes.new_zeros((roi_h, roi_w))
            im_mask = boxes.new_zeros((roi_h, roi_w))
            for box, mask in zip(boxes, masks):  # use forloop instead of paste+sum to avoid memory issue
                x_0_b, y_0_b, x_1_b, y_1_b = box.detach().int()
                x_0_b, y_0_b, x_1_b, y_1_b = x_0_b.item(), y_0_b.item(), x_1_b.item(), y_1_b.item()
                w, h = int(x_1_b - x_0_b + 1), int(y_1_b - y_0_b + 1)
                x_0, x_1 = max(x_0_b, 0), min(x_1_b + 1, roi_w)
                y_0, y_1 = max(y_0_b, 0), min(y_1_b + 1, roi_h)
                mask = torch.nn.functional.interpolate(mask[None], size=(h, w), mode="bilinear", align_corners=False)[0][0]
                # im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - y_0_b) : (y_1 - y_0_b), (x_0 - x_0_b) : (x_1 - x_0_b)]
                im_mask = torch.maximum(im_mask, torch.nn.functional.pad(
                    mask[(y_0 - y_0_b) : (y_1 - y_0_b), (x_0 - x_0_b) : (x_1 - x_0_b)], 
                    (x_0, roi_w-x_1, y_0, roi_h-y_1),
                ))
        elif 'masks' in r:  # semantics related
            im_mask = r['masks'][indices].sum(0)
            im_rois = im_mask.new([[0, 0, roi_w, roi_w]])
        elif 'scores' in r:  # other tasks, cl, reg, survival, etc. must have a 
            im_mask = r['scores'][indices][:, None, None].repeat(1, roi_h, roi_w).sum(0)
            im_rois = im_mask.new([[0, 0, roi_w, roi_w]])
        
        return im_rois, im_mask

    def forward(self, outputs):
        losses = {_: 0. for _ in self.graph['edges']}
        display_maps = []
        for idx, output in enumerate(outputs):
            task_rois, task_maps = {}, {}
            for node_id, (task_id, c_label) in enumerate(self.graph['nodes']):
                task_rois[node_id], task_maps[node_id] = self.get_output_rois_and_masks(output[task_id], c_label)
            display_maps.append(task_maps)
            
            for (node1, node2), w in zip(self.graph['edges'], self.graph['values']):
                loss = 0.
                for x0, y0, x1, y1 in task_rois[node1]:
                    # thinking about roi_align for speed
                    m_pred = task_maps[node1][int(y0):int(y1), int(x0):int(x1)]
                    m_true = task_maps[node2][int(y0):int(y1), int(x0):int(x1)]
                    p = m_pred * m_true
                    loss += -(p ** self.gamma * torch.log(1-p)).sum()
                losses[(node1, node2)] += loss * w
                
#                 loss = 0.
#                 for x0, y0, x1, y1 in task_rois[node2]:
#                     m_pred = task_maps[node2][int(y0):int(y1), int(x0):int(x1)]
#                     m_true = task_maps[node1][int(y0):int(y1), int(x0):int(x1)]
#                     p = m_pred * m_true
#                     loss += -(p ** self.gamma * torch.log(1-p)).sum()
#                 losses[(node2, node1)] += loss * w
        
        return {k: v/len(outputs) for k, v in losses.items()}, display_maps


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', hyp='./hyp.scratch.yaml', ch=3, nc=None, anchors=None, cst=None, nms_params={}):
        # model, input channels, number of classes
        super().__init__()
        self.cfg = load_cfg(cfg)
        self.hyp = load_cfg(hyp)
        self.inplace = self.cfg.get('inplace', True)
        # self.nc = self.cfg['nc']
        # self.names = [str(i) for i in range(self.cfg['nc'])]  # default names
        
        # Define model
        ch = self.cfg['ch'] = self.cfg.get('ch', ch)  # input channels
#         if nc and nc != self.cfg['nc']:
#             LOGGER.info(f"Overriding model.yaml nc={self.cfg['nc']} with nc={nc}")
#             self.cfg['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.cfg['anchors'] = round(anchors)  # override yaml value
        
        # Parse modules from config, specify fpn-backbone and headers
        modules, self.save = parse_model(deepcopy(self.cfg), deepcopy(self.hyp), ch=[ch])  # module_lists, savelist
        n1, n2, n3 = len(self.cfg['backbone']), len(self.cfg['fpn']), len(self.cfg['headers'])
        self.amp = self.cfg['amplification']
        self.backbone = nn.ModuleList(modules[:n1])
        self.fpn = nn.ModuleList(modules[n1:n1+n2])
        self.headers = nn.ModuleDict({m.tag: m for m in modules[n1+n2:]})
        
        # conflicting solver
        if cst is not None:
            self.constrains = ConstrainModulev2(cst)
            self.constrains.input_size = 640
            self.constrains.amp = 10
            self.constrains.K = 2  # select 2 random patches
        else:
            self.constrains = None
        
        # Initialize weights and bias
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, targets=None, augment=False, profile=False, visualize=False):
        if augment:
            LOGGER.warning("***** augment inference is not ready yet.")
            return self._forward_once(x)
            # return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, targets, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        ## NOT FINISHED, don't use ##
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
    
    def _forward_once(self, x, targets=None, profile=False, visualize=False):
        # input_imgs, input_amps = torch.stack([_.img for _ in x]), [_.amp for _ in x]
        # assert len(set(input_amps)) == 1, f"input images has different amplification in a batch."
        input_imgs = torch.stack(x)
        input_amp = 20
        f_scale = self.amp/input_amp
        
        y = []  # intermediate layers
        image_size = input_imgs.shape[-2:]
        bs, ch, h, w = input_imgs.shape
        device = input_imgs.device
        
        # forward backbones and fpn
        x = input_imgs
        for m in self.backbone:  # backbone
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
#             if visualize:
#                 feature_visualization(x, m.type, m.i, save_dir=visualize)
        
        # forward fpn and rescale feature map sizes
        # x = torch.nn.functional.interpolate(x, scale_factor=f_scale, mode='bilinear', align_corners=False)
        for m in self.fpn:
            if m.f != -1:  # if not from previous layer
                if isinstance(m.f, int):
                    if m.f < len(self.backbone):
                        x = torch.nn.functional.interpolate(y[m.f], scale_factor=f_scale, 
                                                            mode='bilinear', align_corners=False)
                    else:
                        x = y[m.f]
                else:
                    f_list = []
                    for j in m.f:
                        if j == -1:
                            f_i = x
                        elif j < len(self.backbone):
                            f_i = torch.nn.functional.interpolate(y[j], scale_factor=f_scale, 
                                                                  mode='bilinear', align_corners=False)
                        else:
                            f_i = y[j]
                        f_list.append(f_i)
                    x = f_list
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if targets is not None:
            losses, outputs = {}, {}
            if self.constrains is not None:
                # randomly select K rois under (h, w, input_amp)
                num_rois_per_image = self.constrains.K
                random_roi_size = self.constrains.input_size / self.constrains.amp * input_amp
                # random_rois: bs*[K, 4]: each image select K random_roi_size patch
#                 random_rois = [
#                     torch.cat([_, _+random_roi_size], -1) 
#                     for _ in torch.rand((bs, num_rois_per_image, 2)) * (torch.tensor([w, h])-random_roi_size)
#                 ]
                random_rois = (torch.rand((bs, num_rois_per_image, 2)) * 
                               (torch.tensor([w, h])-random_roi_size))[..., [0,1,0,1]].float().to(device)
                random_rois += random_rois.new([0., 0., random_roi_size, random_roi_size])
                # print(f"Line 669, random_rois={random_rois} (bs={bs}, K={num_rois_per_image})")
                # print(f"Line 670, random_roi_size is constrain.input_size/constrain_amp * input_amp, random_rois should have length bs, each one is a (K, 4) tensor")
                assert len(random_rois) == bs
                assert random_rois[0].shape[0] == num_rois_per_image

            for task_id, header in self.headers.items():
                header.train()
                x = y[header.f] if isinstance(header.f, int) else [y[j] for j in header.f]
                header_scale = header.amp/input_amp
                task_roi_size = header.input_size/header_scale
                # calculate loss
                task_targets = [_['anns'][task_id] if task_id in _['anns'] else [] for _ in targets]
                task_features, task_gts, num_anns_per_target = mosaic_roi_feature_maps(
                    x, image_size, task_targets, roi_size=task_roi_size,
                    output_size=header.input_size, fpn_scale=f_scale, 
                )
                # print(f"^^^^^^^^^^This is training^^^^^^^^^^^^^")
                # print(task_targets)
                # print(f"^^^^^^^^^^This is training^^^^^^^^^^^^^")
                # print(task_gts)
                # print(f"^^^^^^^^^^This is training^^^^^^^^^^^^^")
                task_losses, _ = header(task_features, task_gts)
                losses = {**losses, **{f'{task_id}/{k}': v for k, v in task_losses.items()}}
                
                # print(f"+++++++++++++++++++ Line 706 extra plot +++++++++++++++++++")
#                 _, task_outputs = header(task_features, None)
#                 r = task_outputs[0]
#                 o_boxes = r['boxes'].detach().cpu()
#                 o_labels = r['labels'].detach().cpu()
#                 o_masks = r['masks'].detach().cpu()
#                 rx0, ry0, rx1, ry1 = task_gts[0]['roi']
#                 patch = input_imgs[0][:, int(ry0):int(ry1), int(rx0):int(rx1)].detach().cpu()
#                 fig, axes = plt.subplots(1, 2, figsize=(24, 12))
#                 axes[0].imshow(patch.permute(1, 2, 0).numpy())
#                 axes[1].imshow(patch.permute(1, 2, 0).numpy())
#                 overlay_detections(axes[1], bboxes=o_boxes, labels=o_labels, masks=None, scores=None,
#                                    labels_color=LABELS_COLOR, labels_text=LABELS_TEXT,
#                                    show_bboxes=True, show_texts=False, show_masks=True, show_scores=False,
#                                   )
#                 plt.show()
                # print(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                
                
                if self.constrains is not None:
                    header.eval()
                    # print("********************  Extract random roi feature maps **********************")
                    # random_gts: bs*[K*N, 4]: each random_roi split into N task_roi_size windows
                    windows = sliding_window_scanner(random_roi_size, task_roi_size, overlap=0).to(device)
                    # num_windows_per_roi = [len(windows)] * bs * num_rois_per_image
                    # print(f"Line 694, task_roi_size={task_roi_size}, windows={windows}")
                    # print(f"Line 695, task_roi_size should be header.input_size/header.amp * input_amp, windows is tensor[N, 4]")
                    random_gts = [torch.cat([windows+windows.new([x0, y0, x0, y0]) for x0, y0, x1, y1 in rois])  
                                  for rois in random_rois]
                    # print(f"Line 698, random_gts={random_gts}, len(random_gts)={len(random_gts)}, len(random_gts[0])={len(random_gts[0])}, K={num_rois_per_image}, N={len(windows)}")
                    # print(f"Line 699, random_gts should have length bs, each is a tensor[K*N, 4]")
                    task_features, task_gts, _ = mosaic_roi_feature_maps(
                        x, image_size, random_gts, roi_size=task_roi_size, 
                        output_size=header.input_size, fpn_scale=f_scale,
                    )
                    # print(f"Line 704, fmaps={[_.shape for _ in x]}, task_roi_size={task_roi_size}, tmaps={[_.shape for _ in task_features]}, output_size={header.input_size}")
                    # run header predictions on all random window (bs*K*N)
                    _, task_outputs = header(task_features, None, calculate_masks=False)
                    # print(f"Line 707, random_outputs_size={len(task_outputs)}, should be bs({bs}) * K({num_rois_per_image}) * N({len(windows)})")
#                     r = task_outputs[0]
#                     o_boxes = r['boxes'].detach().cpu()
#                     o_labels = r['labels'].detach().cpu()
#                     o_masks = r['masks'].detach().cpu()
#                     rx0, ry0, rx1, ry1 = task_gts[0]['roi']
#                     patch = input_imgs[0][:, int(ry0):int(ry1), int(rx0):int(rx1)].detach().cpu()
#                     fig, axes = plt.subplots(1, 2, figsize=(24, 12))
#                     axes[0].imshow(patch.permute(1, 2, 0).numpy())
#                     axes[1].imshow(patch.permute(1, 2, 0).numpy())
#                     overlay_detections(axes[1], bboxes=o_boxes, labels=o_labels, masks=None, scores=None,
#                                        labels_color=LABELS_COLOR, labels_text=LABELS_TEXT,
#                                        show_bboxes=True, show_texts=False, show_masks=True, show_scores=False,
#                                       )
#                     plt.show()
                    
                    # split by N windows, then group results to bs*K, roi is coords relative to input img under fpn scale
                    task_outputs = split_by_sizes(
                        [{**o, 'roi': ann['roi']*header_scale} for o, ann in zip(task_outputs, task_gts)], 
                        [len(windows)] * bs * num_rois_per_image,
                    )
                    # print(f"Line 713, random_outputs={len(random_outputs)}, each outputs has={[len(_) for _ in random_outputs]}")
                    # print(f"Line 714, should have length bs({bs})*K({num_rois_per_image}), each should be N({len(windows)})")
                    # outputs are under header.amp, random_rois are under img.amp, 
                    # switch to relative coordinate inside roi
                    outputs[task_id] = []
                    for o, roi in zip(task_outputs, random_rois.view(-1, 4)):
                        o = header.merge_outputs(o)  # merge window results
                        o['boxes'] -= o['boxes'].new([roi[0], roi[1], roi[0], roi[1]])*header_scale  # relative boxes
                        o['boxes'] *= self.constrains.amp/header.amp  # from header_scale to constrain_scale
                        o['size'] = [self.constrains.input_size, self.constrains.input_size]  # define input size
                        o['roi'] = roi * self.constrains.amp / input_amp  # not needed, roi in raw image to in constrain mode
                        outputs[task_id].append(o)
                    # print(f"Line 720, random_outputs={len(random_outputs)}, {type(random_outputs[0])}, each one has dict item.")

            # contradiction losses
            if self.constrains is not None:
                outputs = [dict(zip(outputs.keys(), _)) for _ in zip(*outputs.values())]
                losses['constrains'] = self.constrains(outputs)
                
            return losses, outputs
        else:
            # task outputs, testing/solving memory issues if image is too large
            outputs = {}
            for task_id, header in self.headers.items():
                x = y[header.f] if isinstance(header.f, int) else [y[j] for j in header.f]
                task_roi_size = header.input_size/header.amp*input_amp
                windows = sliding_window_scanner((h, w), task_roi_size, overlap=0).to(device)
                # num_windows_per_roi = len(windows)
                task_features, task_gts, num_anns_per_target = mosaic_roi_feature_maps(
                    x, image_size, [windows]*bs, roi_size=task_roi_size, 
                    output_size=header.input_size, fpn_scale=f_scale,
                )
#                 print(f"^^^^^^^^^^This is inference^^^^^^^^^^^^^")
#                 print([windows]*bs)
#                 print(f"^^^^^^^^^^This is inference^^^^^^^^^^^^^")
#                 print(task_gts)
#                 print(f"^^^^^^^^^^This is inference^^^^^^^^^^^^^")
                _, task_outputs = header(task_features, None)
                # task_outputs is under header.amp, task_gts is under img.amp
                task_outputs = split_by_sizes(
                    [{**o, 'roi': ann['roi']/input_amp*header.amp} for o, ann in zip(task_outputs, task_gts)], 
                    num_anns_per_target
                )
                outputs[task_id] = [header.merge_outputs(_) for _ in task_outputs]
            
            # contradiction post processing
            outputs = [dict(zip(outputs.keys(), _)) for _ in zip(*outputs.values())]
            outputs = self.post_processing(outputs)

            return None, outputs
    
    def post_processing(self, outputs):
        return outputs

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
        for m in self.backbone.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

#     def autoshape(self):  # add AutoShape module
#         LOGGER.info('Adding AutoShape... ')
#         m = AutoShape(self)  # wrap model
#         copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
#         return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

#     def _apply(self, fn):
#         # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
#         self = super()._apply(fn)
#         for _, m in self.headers.items():
#             if isinstance(m, Detect):
#                 m.grid = list(map(fn, m.grid))
#                 # if isinstance(m.anchor_grid, list):
#                 m.anchor_grid = list(map(fn, m.anchor_grid))
#         return self


def parse_model(d, hyp, ch):  # model_dict, input_channels(3)
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
            m_ = m(*args, nms_params={}, loss_hyp=hyp)  # module
            input_size, amp = header_args
            m_.input_size = input_size
            m_.amp = amp
        elif m is Segment:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            m_ = m(*args, loss_hyp=hyp)  # module
            input_size, amp = header_args
            m_.input_size = (input_size, input_size) if isinstance(input_size, numbers.Number) else input_size
            m_.amp = amp
        else:
            if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                     BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
                c1, c2 = ch[f], args[0]
                # if c2 != no:  # if not output, this won't happen as I switch nc to 3rd place
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


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--profile', action='store_true', help='profile model speed')
#     opt = parser.parse_args()
#     opt.cfg = check_yaml(opt.cfg)  # check YAML
#     print_args(FILE.stem, opt)
#     device = select_device(opt.device)

#     # Create model
#     model = Model(opt.cfg).to(device)
#     model.train()

#     # Profile
#     if opt.profile:
#         img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
#         y = model(img, profile=True)

#     # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
#     # from torch.utils.tensorboard import SummaryWriter
#     # tb_writer = SummaryWriter('.')
#     # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
#     # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph

