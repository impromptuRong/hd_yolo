# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
from torch import Tensor
from typing import Tuple, List, Dict, Optional, Union

from .. import LOGGER, check_version, load_cfg

from .layers import *
from .yolo_head import Detect
from .utils_general import make_divisible
from .utils_torch import fuse_conv_and_bn, initialize_weights, model_info, scale_img
from .loss import DetLoss, SegLoss
# from .utils.plots import feature_visualization

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


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

    def _forward_backbone(self, x: torch.Tensor, visualize=False) -> Dict[int, torch.Tensor]:
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

        return y

    def _forward_once(self, x: torch.Tensor, targets=None, profile=False, visualize=False, compute_masks=False):
        bs, ch, h, w = x.shape
        image_shapes = [(h, w) for i in range(bs)]
        y = self._forward_backbone(x, visualize=visualize)  # intermediate layers

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
            task_losses, task_outputs = header(task_features, image_shapes, task_gts, compute_masks)
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
