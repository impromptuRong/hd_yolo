# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
import torchvision
from collections import OrderedDict


# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# # ROOT = ROOT.relative_to(Path.cwd())  # relative

from .. import LOGGER, check_version

from .layers import *
from .utils_general import make_divisible, check_anchor_order, non_max_suppression
# from .utils.plots import feature_visualization
from .utils_torch import fuse_conv_and_bn, initialize_weights, model_info, scale_img

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


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

#     def forward(self, x, anchors):
#         rois = [_ / self.stride[0] for _ in anchors]  # rois: (bs, [n_bbox, 4])
#         f = torchvision.ops.roi_align(x[0], rois, self.output_size, 1.0, aligned=False)  # (n_roi, 256, 28, 28)
        
#         return self.header(f)
    
    def forward(self, x, anchors):
        for i in range(self.nl-1, -1, -1):
            if i == self.nl-1:
                f = self.m[i](x[i])
            else:
                h, w = x[i].shape[-2], x[i].shape[-1]
                f = torch.nn.functional.interpolate(f, size=(h, w), mode='bilinear', align_corners=True)
                f = self.m[i](f + x[i])
        
        rois = [_ / self.stride[0] for _ in anchors]  # rois: (bs, [n_bbox, 4])
        f = torchvision.ops.roi_align(f, rois, self.output_size, 1.0, aligned=False)  # (n_roi, 256, 28, 28)
        
        return self.header(f)
    
#     def forward_1(self, x, anchors):
#         # for train, directly use anchors and masks from gt. 
#         # for inference, use predicted anchors after nms.
#         for i in range(self.nl):
#             f = self.m[i](x[i])  # f: (bs, channel, h, w)
#             rois = [_ / self.stride[i] for _ in anchors]  # rois: (bs, [n_bbox, 4])
#             # print(f.shape, self.stride[i])
#             x[i] = torchvision.ops.roi_align(f, rois, self.output_size, 1.0, aligned=False)  # (n_roi, 256, 28, 28)
        
#         return self.header(sum(x))


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.ch = ch  # number of channels
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, nc_masks=False, nms_params={}):
        # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.nc = self.yaml['nc']
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        self.det_l = len(self.model)-1
        self.seg_l = None
        
        # Build strides, anchors
        m = self.model[self.det_l]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0]])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

            if nc_masks:
                assert nc_masks in [True, 1, m.nc], "nc_masks need to be True, 1, or same as detection classes! "
                nc_masks = m.nc if nc_masks is True else nc_masks
                seg = Segment(m.ch, nc_masks, feature_layers = m.f,
                              dim_reduced=256, output_size=28, 
                              stride=m.stride, inplace=m.inplace)
                self.model.append(seg)
                self.seg_l = len(self.model)-1
                self.nms_params = self.get_nms_params(nms_params)
            else:
                self.seg_l = None
                self.nms_params = None
        else:
            self.seg_l = None

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')
    
    def get_nms_params(self, args={}):
        default = {'conf_thres': 0.25, 'iou_thres': 0.45, 'classes': None, 'agnostic': False, 
                   'multi_label': False, 'labels': (), 'max_det': 300,}
        return {**default, **args}
        
    def forward(self, x, rois=None, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, rois, profile, visualize)  # single-scale inference, train

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
    
    def _forward_det(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        
        # backbones
        for m in self.model[:self.det_l]:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        # detections
        det = self.model[self.det_l]
        if det.f != -1:  # if not from previous layer
            x = y[dt.f] if isinstance(det.f, int) else [x if j == -1 else y[j] for j in det.f]
        det_output = det(x)
        
        return det_output, y
    
    def _forward_seg(self, det_output, y, rois=None):
        # 1. training: rois is required, return x, masks
        # 2. validation: rois is provided, return (output, x), masks
        # 3. inference: rois not provided, nms on output, return (output_filtered, x), masks_filtered
        det = self.model[self.det_l]
        seg = self.model[self.seg_l]
        if det.f != -1:  # if not from previous layer
            x = y[dt.f] if isinstance(det.f, int) else [x if j == -1 else y[j] for j in det.f]
        if self.training:
            assert rois is not None, "rois can't be empty in training mode!"
            mask_logits = seg(x, rois)
            return det_output, mask_logits
        else:
            if rois is None:  # inference only, use nms on results
                # det_output has boxes+scores, anchor_features in inference
                preds, _ = det_output
                preds = non_max_suppression(preds, **self.nms_params)
                det_output = preds, _
                rois = [_[:, :4] for _ in preds]
                mask_logits = seg(x, rois)

            return det_output, mask_logits

    def _forward_once(self, x, rois=None, profile=False, visualize=False):
        det_output, y = self._forward_det(x, profile=profile, visualize=visualize)
        mask_logits = None
        if self.seg_l:
            det_output, mask_logits = self._forward_seg(det_output, y, rois=rois)
        
        return det_output, mask_logits

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

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
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

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.det_l is not None:
            m = self.model[self.det_l]  # Detect()
            if isinstance(m, Detect):
                m.stride = fn(m.stride)
                m.grid = list(map(fn, m.grid))
                if isinstance(m.anchor_grid, list):
                    m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    def postprocess_detections(self, preds, masks_logits=None):
        outputs = []
        if masks_logits is not None:
            n_obj_per_image = [len(_) for _ in preds]
            # num_masks = mask_probs.shape[0]
            # labels = torch.cat(labels)
            # index = torch.arange(num_masks, device=labels.device)
            # mask_probs = mask_probs[index, labels][:, None]
            mask_probs = masks_logits.sigmoid().split(n_obj_per_image, dim=0)
        else:
            mask_probs = [None] * len(preds)
        
        for p, m in zip(preds, mask_probs):
            boxes = p[:, :4]
            scores = p[:, 4]
            labels = p[:, 5].long()
            r = {'boxes': boxes, 'scores': scores, 'labels': labels}
            
            if m is not None:
                if self.model[self.seg_l].nc == self.model[self.det_l].nc:
                    index = torch.arange(len(m), device=labels.device)
                    m = m[index, labels][:, None]
                r['masks'] = m
                # r['masks'] = paste_masks_in_image(m, boxes, img_shape, padding=1).squeeze(1)
            outputs.append(r)

        return outputs


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.ModuleList(layers), sorted(save)


class EnsembleMask(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()
    
    def forward(self, x, augment=False, profile=False, visualize=False):
        # we use all module for detection but self[-1] mask header
        det_output, mask_features = [], None
        for mi, module in enumerate(self):
            if hasattr(module, '_forward_det'):
                y, mask_features = module._forward_det(x, profile, visualize)
            else:  # compatible with old model
                y = module.forward(x, augment, profile, visualize)
            det_output.append(y[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        det_output = (torch.cat(det_output, 1), None)  # nms ensemble
        
        mask_logits = None
        if mask_features is not None and self[-1].seg_l:
            det_output, mask_logits = self[-1]._forward_seg(det_output, mask_features)
        
        return det_output, mask_logits  # inference, train output
    
    def postprocess_detections(self, preds, mask_logits=None):
        return self[-1].postprocess_detections(preds, mask_logits)



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
