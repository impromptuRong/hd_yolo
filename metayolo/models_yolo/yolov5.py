# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import os
import sys
from copy import deepcopy
from torch import Tensor
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union, Iterable, Iterator

from .. import LOGGER, check_version, load_cfg

from .layers import *
from .yolo_head import Detect
from .utils_general import make_divisible
from .utils_torch import fuse_conv_and_bn, initialize_weights, model_info, scale_img
# from .utils.plots import feature_visualization

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

## This one works, but we prefer to split backbone and fpn into two pieces.
# class YoloList(nn.ModuleList):
#     def __init__(self, modules=None, return_layers=None):
#         super().__init__(modules)
#         self.save = return_layers

#     def forward(self, x, visualize=False) -> Dict[int, torch.Tensor]:
#         if not isinstance(x, dict):
#             y: Dict[int, torch.Tensor] = {-1: x}
#         else:
#             y = x

#         for m in self:
#             if isinstance(m.f, int):
#                 y[-1] = m(y[m.f])
#             else:
#                 y[-1] = m([y[_] for _ in m.f])
#             if m.i in self.save:
#                 y[m.i] = y[-1]
# #             if visualize:
# #                 feature_visualization(y[-1], m.type, m.i, save_dir=visualize)

#         return {k: y[k] for k in self.save if k in y}


class CSPDarkNet(nn.Sequential):
    def __init__(self, modules: Optional[Iterable[nn.Module]] = None,
                 return_layers: Optional[List] = None) -> None:
        super().__init__(*modules)
        self.save = return_layers or [len(self)-1]
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        y: List[torch.Tensor] = []
        for m in self:
            x = m(x)
            y.append(x)

        return {k: y[k] for k in self.save}


class FPN(nn.Sequential):
    def __init__(self, modules: Optional[Iterable[nn.Module]] = None,
                 return_layers: Optional[List] = None) -> None:
        super().__init__(*modules)
        self.save = return_layers or [len(self)-1]

    def forward(self, x: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        for m in self:
            if isinstance(m.f, int):
                x[-1] = m(x[m.f])
            else:
                x[-1] = m([x[_] for _ in m.f])
            if m.i in self.save:
                x[m.i] = x[-1]

        return {k: x[k] for k in self.save}


def build_network(cfg, hyp, is_scripting=False):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    gd, gw, ch = cfg['depth_multiple'], cfg['width_multiple'], [cfg['ch']]

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, _ in enumerate(cfg['backbone'] + cfg['fpn'] + cfg['headers']):
        f, n, m, args = _[0], _[1], _[2], _[3]  # from, number, module, args, (tag)
        tag = _[4] if len(_) > 4 else None
        header_args = _[5] if len(_) > 5 else None  # header only
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            if isinstance(a, str) and a in cfg:
                args[j] = cfg[a]
            # try:
            #     args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            # except NameError:
            #     pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m is Detect:
            args = [[ch[x] for x in f]] + args
            if isinstance(args[1], int):  # number of anchors, don't use for anchor free model
                args[1] = [list(range(args[1] * 2))] * len(f)
            # get task header hyperparameters
            tag = tag or 'det'
            loss_keys = ['box', 'cls', 'cls_pw', 'cls_cw', 'obj', 'obj_pw', 'mask', 
                         'iou_t', 'anchor_t', 'fl_gamma', 'label_smoothing',]
            nms_keys = ['conf_thres', 'iou_thres', 'max_det']
            loss_hyp = {k: hyp[tag][k] for k in loss_keys if k in hyp[tag]}
            nms_params = {k: hyp[tag][k] for k in nms_keys if k in hyp[tag]}
            multi_label = False or hyp[tag]['multi_label']
            if isinstance(args[-1], int):  # if it's an int, all class goes to same mask class
                args[-1] = {cl: args[-1] for cl in range(args[-2]+1)}
            # do modification about mask here before re-write parser
            # args[-1] = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: -1}  # {1: 1, 2: 2, 3: 3, 4: 4,}  #
            # args[-1] = {cl: 0 for cl in range(args[-2]+1)}
            # args[-1] = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 5, 6: 5}
            # args[-1] = {1: 1, 2: 2, 3: 3, 4: 4, 5: 2, 6: -1, 7: 0}  # nucls_paper mask_indices
            m_ = m(*args, multi_label=multi_label, 
                   nms_params=nms_params, loss_hyp=loss_hyp,
                   is_scripting=is_scripting,
                  )  # module
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

    modules, save = layers, sorted(save)
    n1, n2, n3 = len(cfg['backbone']), len(cfg['fpn']), len(cfg['headers'])
    backbone = CSPDarkNet(modules[:n1], [_ for _ in save if _ < n1])
    fpn = FPN(modules[n1:n1+n2], save)
    headers = nn.ModuleDict({m.tag: m for m in modules[n1+n2:]})

    return backbone, fpn, headers

## Not working for scripting, can't access _modules, can't access inherited member/fun
# from torch._jit_internal import _copy_to_script_wrapper
# class Backbone(nn.Module):
#     def __init__(self, modules: Optional[Iterable[nn.Module]] = None,
#                  return_layers: Optional[List] = None) -> None:
#         super().__init__()
#         if modules is not None:
#             for idx, m in enumerate(modules):
#                 setattr(self, str(idx), m)
#             self._len = len(modules)
#         else:
#             self._len = 0
#         self.save = return_layers or [len(self)-1]

#     @_copy_to_script_wrapper
#     def __len__(self) -> int:
#         return self._len

#     @_copy_to_script_wrapper
#     def __iter__(self) -> Iterator[nn.Module]:
#         return iter(self._modules.values())

#     def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
#         # y: Dict[int, torch.Tensor] = {-1: x}
#         y: List[torch.Tensor] = []
#         for k, m in self._modules.items():
#             x = m(x)
#             y.append(x)

#         return {k: y[k] for k in self.save}

