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


# def build_network(cfg, hyp, is_scripting=False):  # model_dict, input_channels(3)
#     LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
#     gd, gw, ch = cfg['depth_multiple'], cfg['width_multiple'], [cfg['ch']]

#     LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")

    
def build_network(config, channels, num_classes, anchors, num_layers):
    depth_mul = cfg['depth_multiple']  # config.model.depth_multiple
    width_mul = cfg['width_multiple']  # config.model.width_multiple
    
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    num_anchors = config.model.head.anchors
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    
    if 'CSP' in config.model.backbone.type:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    head_layers = build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max)

    head = Detect(num_classes, anchors, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


