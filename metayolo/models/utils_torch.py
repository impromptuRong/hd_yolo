import math
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import LOGGER, check_version

try:
    from torchvision.ops import FrozenBatchNorm2d
except:
    from torchvision.ops.misc import FrozenBatchNorm2d

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')


# torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
if check_version(torch.__version__, '1.10.0'):
    def torch_meshgrid(*tensors):
        return torch.meshgrid(*tensors, indexing='ij')
else:
    def torch_meshgrid(*tensors):
        return torch.meshgrid(*tensors)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''

    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop


def freeze_params(model, layers=[]):
    """ Freeze parameters in given layers.
        layers: list of strings of layer/module
        layers = ['backbone', 'neck', 'header.det.m', ...]
    """
    if not layers:
        return model

    for k, v in model.named_parameters():
        # v.requires_grad = True  # don't unfreeze previous freezed layer
        if any(k.startswith(_ + '.') for _ in layers):
            LOGGER.info(f"freeze param: {k}")
            v.requires_grad = False

    return model


def freeze_bn(model, layers=[]):
    """ Freeze bn.running_mean and bn.running_var. (inplace)
        We replace nn.BatchNorm2d with torchvision.ops.FrozenBatchNorm2d
        Call m.eval() after model.train() in the trianing loop only work on single GPU,
        but failed to freeze running_mean and running_var in DDP.
    """
    if not layers:
        return model

    replace_layers = {}
    for k, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) and any((k == _) or k.startswith(_+'.') for _ in layers):
            replace_layers[k] = FrozenBatchNorm2d(m.num_features, m.eps).to(m.weight.device)
            replace_layers[k].load_state_dict(m.state_dict())

    for k, m in replace_layers.items():
        path = k.split('.')
        parent_node = model
        for attr in path[:-1]:
            parent_node = getattr(parent_node, str(attr))
        LOGGER.info(f"Replace layer: {k} with FrozenBatchNorm2d")
        setattr(parent_node, str(path[-1]), m)

    return model


def freeze_bn_dont_use(model, layers=[]):
    """ call eval to stop updating running_mean and running_var. """
    if not layers:
        return model

    for k, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) and any(k.startswith(_) for _ in layers):
            m.eval()

    return model


def freeze_detection(model, freeze_bn=True):
    # freeze all parameters 
    for param in model.model[:model.seg_l].parameters():
        param.requires_grad = False
    
    # freeze all bn layers stats
    if freeze_bn:
        for module in model.model[:model.seg_l].modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    return model


def one_hot_labels(x, num_classes=None):
    """ labels start from 1 to num_classes, one_hot to N*(num_classes+1).
        Put all negative/others/unclassified/out-range, etc in first column.
    """
    num_classes = num_classes or x.max()
    x = torch.where((x > 0) & (x <= num_classes), x, 0)
    return torch.nn.functional.one_hot(x, num_classes=num_classes+1)
