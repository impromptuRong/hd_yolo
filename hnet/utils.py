import math
import yaml
import torch
import torchvision
import torch.nn.functional as F
import numbers
import itertools

# from .torch_layers import *
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Dict, Optional


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p


def split_by_sizes(x, sizes):
    assert len(x) == sum(sizes)
    ends = list(itertools.accumulate(list(sizes)))
    starts = [0] + list(ends[:-1])
    
    return [x[s:e] for s, e in zip(starts, ends)]


def get_size(x, default=None):
    # if x is None and default is None:
    #     raise ValueError(f"input and default cannot both be None.")
    x = x if x is not None else default
    
    return (x, x) if isinstance(x, numbers.Number) else x


def sliding_window_scanner(image_size, roi_size=None, overlap=0):
    if roi_size is None:
        return torch.tensor([[0., 0., image_size[0], image_size[1]]], dtype=torch.float32)
    
    image_size = get_size(image_size)
    h, w = image_size[0], image_size[1]
    roi_size = get_size(roi_size)
    roi_h, roi_w = roi_size[0], roi_size[1]
    
    if w > roi_w:
        x0 = torch.arange(0, w, roi_w-overlap, dtype=torch.float32)
    else:
        x0 = torch.zeros((1,), dtype=torch.float32)  # ignore overlap if roi_w >= w
    if h > roi_h: 
        y0 = torch.arange(0, h, roi_h-overlap, dtype=torch.float32)
    else:
        y0 = torch.zeros((1,), dtype=torch.float32)  # ignore overlap if roi_h >= h
    
    y0, x0 = torch.meshgrid(y0, x0)
    x0, y0 = x0.reshape(-1), y0.reshape(-1)
    x1, y1 = x0 + roi_w, y0 + roi_h

    boxes = torch.stack((x0, y0, x1, y1), dim=1)
    boxes = torchvision.ops.boxes.clip_boxes_to_image(boxes, (h, w))

    return boxes


def extract_roi_feature_maps(x, image_size, roi_sizes=None, targets=None, scale=1.0, output_size=None):
    """
        Extract ROIs from Feature Maps (FPN).
        Args:
            x (OrderedDict or List): {'0': [N, c, h, w],}/[[N, c, h, w],] feature maps for each feature level.
            image_size (Tuple):  the input image shape (img_h_after_transform, img_w_after_transform).
            roi_sizes (Optional, List(Tuple)): specific to task_id, [(roi_h_after_transform, roi_w_after_transform)].
                roi_sizes are only used if targets is None. function will split the whole feature map into small
                patches by function sliding_window_scanner. If both targets and roi_sizes are None, roi_sizes will 
                set to image_size for each image: just roi_align all feature maps to output_size.
            targets (Optional, List[List[Dict]]): list of annotations. function will extract all rois from
                each feature maps based on the stride factor of original image_size. If None, function will consider
                roi_size. (See roi_size for details).
            scale (Optinal, 1.0): the scale of feature_map/image_size. Assume feature_map is enlarged to 40x, but 
                image is a 20x image, scale=2.0. 
            output_size (Optional, Tuple): the output_size of roi_align. If output_size is None, output size will as 
                same as x in each layer.
            
        Returns:
            results (OrderedDict[Tensor]): feature maps for ROI on FPN layers.
                They are ordered from highest resolution first.
    """
    if isinstance(x, torch.Tensor):
        feature_names, features = None, [x]
    elif isinstance(x, dict):
        feature_names, features = list(zip(*x.items()))
    else:
        feature_names, features = None, x
    
    num_images = features[0].shape[0]
    device = features[0].device
    image_size = get_size(image_size)
    output_size = get_size(output_size)

    boxes, task_targets, num_anns_per_target = [], [], []
    if targets is None:  # we only use roi_size if target==None
        if not isinstance(roi_sizes, list):
            roi_size = get_size(roi_sizes, image_size)
            roi_sizes = [roi_size] * num_images
            # None, 640, (640, 640), [640, 320], [(640, 640), (320, 320)]
        assert len(roi_sizes) == num_images
        for roi_size in roi_sizes:
            rois = sliding_window_scanner(image_size, get_size(roi_size), overlap=0).to(device)
            anns = [{'roi': _} for _ in rois]
            
            boxes.append(rois)
            task_targets.extend(anns)
            num_anns_per_target.append(len(anns))
    else:
        assert len(targets) == num_images
        for t in targets:
            if isinstance(t, torch.Tensor):
                rois = t
                anns = [{'roi': _} for _ in rois]
            elif len(t):
                rois = torch.stack([ann['roi'] for ann in t]).to(device)
                anns = [ann for ann in t]
            else:
                rois = torch.zeros((0, 4)).to(device) # create an empty tensor
                anns = []
            
            boxes.append(rois)
            task_targets.extend(anns)
            num_anns_per_target.append(len(anns))

    # print("*******utils_o", boxes, [_['roi'] for _ in task_targets], num_anns_per_target)
    ## calculate scale_factors for each feature map:
    task_features = []
    for f in features:
        scale_h, scale_w = f.shape[2]/(image_size[0]*scale), f.shape[3]/(image_size[1]*scale)
        roi_multiplier = f.new([scale_w, scale_h, scale_w, scale_h])
        rois = [_ * roi_multiplier * scale for _ in boxes]
        if output_size is None:
            o_size = (f.shape[2], f.shape[3])  # keep original feature map size
        else:
            o_size = (round(output_size[0] * scale_h), round(output_size[1] * scale_w))
        print(f"Line152: image=({image_size}, {scale}), fmap={f.shape}, rois={rois}, output_size={o_size}")
        task_features.append(torchvision.ops.roi_align(f, rois, o_size, aligned=True))  #.to(device2)
    
    if feature_names is not None:
        task_features = OrderedDict({n: f for n, f in zip(feature_names, task_features)})
    
    return task_features, task_targets, num_anns_per_target


def mosaic_roi_feature_maps(x, image_size, targets, roi_size, output_size, fpn_scale=1.0):
    """ 
        Build a mosaic feature map and gts extractor.
        Args:
            x (OrderedDict or List): {'0': [N, c, h, w],}/[[N, c, h, w],] feature maps for each feature level.
            image_size (Tuple):  the input image size (img_h_after_transform, img_w_after_transform).
            targets (Optional, List[List[Dict]]): list of annotations. function will extract all rois from
                each feature maps based on the stride factor of original image_size. If None, function will consider
                roi_size. (See roi_size for details).
            
            roi_size: roi_size to extract under image_size (not fpn size). Function will crop roi_size in 
                image_size (roi_size * fpn_scale in x) and targets. Then roi_align to output_size.
            output_size: For each pyramid layer, output_size = roi_size * header.amp / img.amp * pyra_multiplier
                Use the formula to calculate roi_size and output_size for the function.
            fpn_scale (Optional, 1.0): the scale of feature_map/image_size. Assume feature_map is enlarged to 40x, but 
                image is a 20x image, scale=2.0. 
        
        Returns:
            results (OrderedDict[Tensor]): feature maps for ROI on FPN layers.
                They are ordered from highest resolution first.
        
        Logics:
            1. extract all rois in targets original image_size. (h, w, 20x)
            2. crop/pad/mosaic all rois to roi_size.
            3. rois_fpn = rois * fpn_scale (fpn.amp/img.amp).
            4. roi_align rois_fpn to output_size*pyra_multiplier on all pyramid layer. 
        Under developing crop/pad/mosaic, paste 9 random feature maps into a large feature maps
    """
    if isinstance(x, torch.Tensor):
        feature_names, features = None, [x]
    elif isinstance(x, dict):
        feature_names, features = list(zip(*x.items()))
    else:
        feature_names, features = None, x
    
    num_images = features[0].shape[0]
    device = features[0].device
    image_size = get_size(image_size)
    roi_size = get_size(roi_size)
    output_size = get_size(output_size)
    
    boxes, task_targets, num_anns_per_target = [], [], []
    for t in targets:
        if t is None:
            rois = torch.zeros((0, 4)).to(device) # create an empty tensor
            anns = []
        elif isinstance(t, torch.Tensor):
            rois = t
            anns = [{'roi': _} for _ in rois]
        elif len(t):
            rois = torch.stack([ann['roi'] for ann in t]).to(device)
            anns = [ann for ann in t]
        else:
            rois = torch.zeros((0, 4)).to(device) # create an empty tensor
            anns = []
        boxes.append(rois)
        task_targets.extend(anns)
        num_anns_per_target.append(len(anns))

    # Add a temporary assert to trigger: output_size != roi_size * output_scale
    # Removed this part after finish coding random_cropping/random_padding/mosaic
    # if roi_h, roi_w < output_size, randomly crop, pad, mosaic to output_size.
    for rois in boxes:
        assert (rois[:,3]-rois[:,1]).float().allclose(torch.tensor(roi_size[0])), f"{rois} and {roi_size} are not matching!"
        assert (rois[:,2]-rois[:,0]).float().allclose(torch.tensor(roi_size[1])), f"{rois} and {roi_size} are not matching!"

    task_features = []
    for f in features:
        pyra_h, pyra_w = f.shape[2]/(image_size[0]*fpn_scale), f.shape[3]/(image_size[1]*fpn_scale)  # relative to fpn
        pyra_multiplier = f.new([pyra_w, pyra_h, pyra_w, pyra_h])
        rois = [_ * fpn_scale * pyra_multiplier for _ in boxes]
        o_size = (round(output_size[0] * pyra_h), round(output_size[1] * pyra_w))
        # print(f"Line234: image={image_size}, fmap={f.shape, fpn_scale}")
        # print(f"Line234: io_size={roi_size, output_size, o_size}, rois={rois}")
        task_features.append(torchvision.ops.roi_align(f, rois, o_size, aligned=True))

    if feature_names is not None:
        task_features = OrderedDict({n: f for n, f in zip(feature_names, task_features)})
    
    return task_features, task_targets, num_anns_per_target


## roi_align use aligned=True here
# a = torch.tensor([[
#     [1.,2,3,4,5,6,7,],
#     [2,3,4,5,6,7,8,],
#     [3,4,5,6,7,8,9,],
#     [4,5,6,7,8,9,10,],
#     [5,6,7,8,9,10,11,],
#     [6,7,8,9,10,11,12,],
# ]])[None]
# print(a.shape)
# boxes = [torch.tensor([[0., 0, 2, 2], [2, 3, 4, 5], [5,4,7,6]])]
# pp = torchvision.ops.roi_align(a, boxes, (2, 2), sampling_ratio=-1, aligned=True)
# for _ in pp:
#     print(_)
