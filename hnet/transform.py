import math
import torch
import torchvision
import numpy as np
import itertools

from torch import nn, Tensor
from typing import List, Tuple, Dict, Optional

from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import *
# from .roi_heads import paste_masks_in_image

def split_by_sizes(x, sizes):
    assert len(x) == sum(sizes)
    ends = list(itertools.accumulate(list(sizes)))
    starts = [0] + list(ends[:-1])
    
    return [x[s:e] for s, e in zip(starts, ends)]

def get_img_pad_width(input_size, output_size, pos='center'):
    # output_size = output_size + input_size[len(output_size):]
    output_size = np.maximum(input_size, output_size)
    if pos == 'center':
        l = np.floor_divide(output_size - input_size, 2)
    elif pos == 'random':
        l = [np.random.randint(0, _ + 1) for _ in output_size - input_size]
    return list(zip(l, output_size - input_size - l))


def align_roi_to_divisible(roi_box, roi_size=None, size_divisible=32):
    """ Align roi coordinates to divisible. 
        If roi_size is given, will crop to roi_size.
    """
    stride = float(size_divisible)
    w0, h0, w1, h1 = roi_box
    
    ## get new roi coords:
    w0_new = int(math.floor(w0 / stride) * stride)
    h0_new = int(math.floor(h0 / stride) * stride)
    w1_new = int(math.ceil(w1 / stride) * stride)
    h1_new = int(math.ceil(h1 / stride) * stride)
    
    if roi_size is not None:
        roi_h, roi_w = roi_size
        assert roi_h % stride == 0 and roi_w % stride == 0
        h_s = (h1_new - h0_new - roi_h)/stride
        w_s = (w1_new - w0_new - roi_w)/stride
        
        if h1_new - h1 >= h0 - h0_new:
            up, down = h_s//2, h_s-h_s//2
        else:
            down, up = h_s//2, h_s-h_s//2
        if w1_new - w1 >= w0 - w0_new:
            left, right = w_s//2, w_s-w_s//2
        else:
            right, left = w_s//2, w_s-w_s//2

        w0_new = int(w0_new + left * stride)
        h0_new = int(h0_new + up * stride)
        w1_new = int(w1_new - right * stride)
        h1_new = int(h1_new - down * stride)
    
    return [(h0-h0_new, h1_new-h1), (w0-w0_new, w1_new-w1)]


def pad_image(x, padding):
    """ F.pad: (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back) 
        Support negative padding.
    """
    (h_0, h_1), (w_0, w_1) = padding[-2:]
    return torch.nn.functional.pad(x, (w_0, w_1, h_0, h_1))


def pad_boxes(boxes, padding, image_size=None):
    """ Support negative padding. Crop box if image_size is given.
        a = torch.Tensor([1,3,5,7])
        x1, y1, x2, y2 = a.unbind(-1)
        torch.stack((x1, y1, x2, y2), dim=-1)

        b = torch.Tensor([[1,3,5,7],[2,4,6,8]])
        x1, y1, x2, y2 = b.unbind(-1)
        torch.stack((x1, y1, x2, y2), dim=-1)
    """
    b_w0, b_h0, b_w1, b_h1 = boxes.unbind(-1)
    (h_0, h_1), (w_0, w_1) = padding[-2:]
    b_w0_new = b_w0 + w_0
    b_h0_new = b_h0 + h_0
    b_w1_new = b_w1 + w_0
    b_h1_new = b_h1 + h_0
    
    if image_size is not None:
        h_min = w_min = 0
        h_max = image_size[0] + h_0 + h_1
        w_max = image_size[1] + w_0 + w_1    
        b_w0_new = torch.clamp(b_w0_new, w_min, w_max)
        b_h0_new = torch.clamp(b_h0_new, h_min, h_max)
        b_w1_new = torch.clamp(b_w1_new, w_min, w_max)
        b_h1_new = torch.clamp(b_h1_new, h_min, h_max)
    
    return torch.stack((b_w0_new, b_h0_new, b_w1_new, b_h1_new), dim=-1)


def pad_annotation(ann, padding):
    """ Pad annotation. 
        The function will keep ann['roi'] vs. ann['size'] ratio after padding.
        For each annotation, we first resize from ann['size'] to ann['roi'], 
        then pad to (h_ann_new, w_ann_new), then resize back with scales.
    """
    ann = {k: v for k, v in ann.items()}  # make a shallow copy
    w0, h0, w1, h1 = ann['roi']  # roi coords in image
    h_ann, w_ann = ann['size']  # annotation size
    scale_h, scale_w = (h1 - h0)/h_ann, (w1 - w0)/w_ann  # scales for ann['size'] to ann['roi']
    
    # New size after padding
    (h_0, h_1), (w_0, w_1) = padding[-2:]  # padding based on coords in image
    h0_new, h1_new = h0 - h_0, h1 + h_1
    w0_new, w1_new = w0 - w_0, w1 + w_1
    ann['roi'] = ann['roi'].new([w0_new, h0_new, w1_new, h1_new])
    
    # New ann_size after padding
    h_ann_new, w_ann_new = (h1_new - h0_new)/scale_h, (w1_new - w0_new)/scale_w
    ann['size_ori'] = ann['size']
    ann['size'] = ann['size'].new([h_ann_new, w_ann_new])
    
    # Resize from ann['size'] to ann['roi'], pad to (h_ann_new, w_ann_new), resize back with scales
    if 'boxes' in ann:
        boxes = ann['boxes']
        if h_ann != h1 - h0 or w_ann != w1 - w0:
            scales = boxes.new([scale_w, scale_h, scale_w, scale_h])
            boxes = pad_boxes(boxes*scales, padding, (h_ann_new, w_ann_new))/scales
        else:
            boxes = pad_boxes(boxes, padding, (h_ann_new, w_ann_new))
    if 'masks' in ann:
        masks = ann['masks']
        if h_ann != h1 - h0 or w_ann != w1 - w0:
            masks = torch.nn.functional.interpolate(
                masks[:, None].float(), size=((h1-h0).item(), (w1-w0).item()), mode='bilinear')
            masks = pad_image(masks, padding)
            masks = torch.nn.functional.interpolate(
                masks, size=tuple(ann['size'].tolist()), mode='bilinear')[:, 0] #.byte()
        else:
            masks = pad_image(masks, padding)
        ann['masks'] = masks
    
    return ann


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(-1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=-1)


def project_roi_boxes_to_image(boxes, roi, target_size):
    target_h, target_w = target_size[0], target_size[1]
    roi_x0, roi_y0, roi_x1, roi_y1 = roi
    roi_h, roi_w = roi_y1 - roi_y0, roi_x1 - roi_x0
    
    ## resize boxes from target_size to roi_size
    boxes = resize_boxes(boxes, (target_h, target_w), (roi_h, roi_w))
    ## get bbox absolute location in the whole image
    x0, y0, x1, y1 = boxes.unbind(-1)
    x0_abs, y0_abs, x1_abs, y1_abs = x0 + roi_x0, y0 + roi_y0, x1 + roi_x0, y1 + roi_y0
    
    return torch.stack((x0_abs, y0_abs, x1_abs, y1_abs), dim=-1)


def project_image_boxes_to_roi(boxes, roi, target_size):
    target_h, target_w = target_size[0], target_size[1]
    roi_x0, roi_y0, roi_x1, roi_y1 = roi
    roi_h, roi_w = roi_y1 - roi_y0, roi_x1 - roi_x0
    
    ## get bbox relative location in new roi
    x0, y0, x1, y1 = boxes.unbind(-1)
    x0, y0, x1, y1 = x0 - roi_x0, y0 - roi_y0, x1 - roi_x0, y1 - roi_y0
    ## resize boxes from new_roi_size to new_target_size
    boxes = torch.stack((x0, y0, x1, y1), dim=-1)
    boxes = resize_boxes(boxes, (roi_h, roi_w), (target_h, target_w))

    return boxes


# def project_masks_on_boxes(masks, boxes, matched_idxs, M):
#     # type: (Tensor, Tensor, Tensor, int) -> Tensor
#     """
#         Given segmentation masks and the bounding boxes corresponding
#         to the location of the masks in the image, this function
#         crops and resizes the masks in the position defined by the
#         boxes. This prepares the masks for them to be fed to the
#         loss computation as the targets.
#     """
#     matched_idxs = matched_idxs.to(boxes)
#     rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
#     gt_masks = gt_masks[:, None].to(rois)
#     return roi_align(gt_masks, rois, (M, M), 1.)[:, 0]



# def resize_image_and_masks(image: Tensor, self_min_size: float, self_max_size: float,
#                            target: Optional[Dict[str, Tensor]] = None,
#                            fixed_size: Optional[Tuple[int, int]] = None,
#                           ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
#     return _resize_image_and_masks(image, self_min_size, self_max_size, target, fixed_size)


class GeneralizedTransform(GeneralizedRCNNTransform):
#     def __init__(self, min_size, max_size, image_mean, image_std, size_divisible=32, fixed_size=None):
#         super(GeneralizedRCNNTransform, self).__init__()
#         if not isinstance(min_size, (list, tuple)):
#             min_size = (min_size,)
#         self.min_size = min_size
#         self.max_size = max_size
#         self.image_mean = image_mean
#         self.image_std = image_std
#         self.size_divisible = size_divisible
#         self.fixed_size = fixed_size
    
    def forward(self, images, targets=None, roi_sizes=None):
        """ Do the following transform:
            1. Pad images to (self.max_size, self.max_size) and change annotation rois.
               target['size_ori'] stores original size, ann['roi_ori'] stores the original coords.
               annotations roi and its members doesn't change in this step.
            2. Pad (crop) ann['roi'] to roi_sizes[task_id] (maybe divisible) (for FPN, 32 etc).
               And rescale all annotations to roi_sizes[task_id]. ann['size'] will change to roi_sizes[task_id],
               ann['size_ori'] stores the original annotation size. 
        """
        max_size = self.max_by_axis([list(img.shape) for img in images])
        # max_size = list(max_size)
        # max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        output_size = (3, self.max_size, self.max_size)
        # stride = float(size_divisible)
        
        images = [_ for _ in images]
        self.pad_images_and_targets(images, output_size, pos='center', targets=targets)
        image_sizes_list = [(img.shape[-2], img.shape[-1]) for img in images]
        images = torch.stack(images)
#         image_sizes_list: List[Tuple[int, int]] = []
#         for image_size in image_sizes:
#             assert len(image_size) == 2
#             image_sizes_list.append((image_size[0], image_size[1]))
        
        image_list = ImageList(images, image_sizes_list)
        self.pad_annotation_to_divisible(targets, roi_sizes, size_divisible=32)
        
        return image_list, targets
    
    def pad_images_and_targets(self, images, output_size, pos='center', targets=None):
        """ Function modifies images and targets inplace. """
        for i in range(len(images)):
            image = images[i]
            padding = get_img_pad_width(list(image.shape), output_size, pos=pos)
            images[i] = pad_image(image, padding)
            
            if targets is not None:
                target = targets[i]
                target['size_ori'] = target['size']
                target['size'] = target['size_ori'].new([image.shape[-2], image.shape[-1]])
                for task_id, anns in target['anns'].items():
                    for ann in anns:
                        ann['roi_ori'] = ann['roi']
                        ann['roi'] = pad_boxes(ann['roi'], padding)
    
    def pad_annotation_to_divisible(self, targets, roi_sizes=None, size_divisible=32):
        """ Function modifies targets inplace. 
            Pad each roi to size_divisible (for FPN)
            Not necessary with RoIAlign.
        """
        for target in targets:
            for task_id, anns in target['anns'].items():
                roi_size = roi_sizes.get(task_id, None) if roi_sizes is not None else None
                for _, ann in enumerate(anns):
                    padding = align_roi_to_divisible(ann['roi'], roi_size, size_divisible)
                    anns[_] = pad_annotation(ann, padding)


# class GeneralizedTransform(torch.nn.Module):
#     """ pad image """
#     def __init__(self, min_size, max_size, image_mean, image_std, size_divisible=32, fixed_size=None):
#         super(GeneralizedTransform, self).__init__()
#         if not isinstance(min_size, (list, tuple)):
#             min_size = (min_size,)
#         self.min_size = min_size
#         self.max_size = max_size
#         self.image_mean = image_mean
#         self.image_std = image_std
#         self.size_divisible = size_divisible
#         self.fixed_size = fixed_size

#     def max_by_axis(self, the_list):
#         # type: (List[List[int]]) -> List[int]
#         maxes = the_list[0]
#         for sublist in the_list[1:]:
#             for index, item in enumerate(sublist):
#                 maxes[index] = max(maxes[index], item)
#         return maxes
    
#     def batch_images(self, images, size_divisible=32):
#         # type: (List[Tensor], int) -> Tensor
#         if torchvision._is_tracing():
#             # batch_images() does not export well to ONNX
#             # call _onnx_batch_images() instead
#             return self._onnx_batch_images(images, size_divisible)

#         max_size = self.max_by_axis([list(img.shape) for img in images])
#         stride = float(size_divisible)
#         max_size = list(max_size)
#         max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
#         max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

#         batch_shape = [len(images)] + max_size
#         batched_imgs = images[0].new_full(batch_shape, 0)
#         for img, pad_img in zip(images, batched_imgs):
#             pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

#         return batched_imgs
    
#     def forward(self, images, targets=None):
#         images = [img for img in images]
#         image_sizes = [img.shape[-2:] for img in images]
        
#         images = self.batch_images(images, size_divisible=self.size_divisible)
#         image_sizes_list: List[Tuple[int, int]] = []
#         for image_size in image_sizes:
#             assert len(image_size) == 2
#             image_sizes_list.append((image_size[0], image_size[1]))

#         image_list = tmdet.image_list.ImageList(images, image_sizes_list)
        
#         return image_list, targets


# ## filter out small objects: min_area=100, max_area=10000, h, w >= 10
#         min_area, max_area = kwargs['min_area'], kwargs['max_area']
#         min_h, min_w = kwargs['min_h'], kwargs['min_w']
#         r_area = targets['area']/h/w
#         r_h = (targets['boxes'][:, 3] - targets['boxes'][:, 1])/h
#         r_w = (targets['boxes'][:, 2] - targets['boxes'][:, 0])/w
        
#         keep_indices = (r_area >= min_area) & (r_area < max_area) & (r_h >= min_h) & (r_w >= min_w)
#         targets = filter_tensor_targets(targets, keep_indices)
        
#         display_detection(image.permute(1, 2, 0).numpy(), targets['boxes'].numpy(), targets['labels'].numpy(), 
#                           kwargs['label_to_val'], masks=targets['masks'].numpy())
        
#         ## For RCNN training only, add a bbox with label=0 to indicate background
#         if not len(targets['boxes']):
#             targets['boxes'] = torch.tensor([[0, 0, w-1, h-1]], dtype=torch.float32)
#             targets['labels'] = torch.tensor([0], dtype=torch.int64)
#             targets['masks'] = torch.ones((1, h, w), dtype=torch.uint8)
#             targets['area'] = torch.tensor([h*w], dtype=torch.int64)
#             targets['iscrowd'] = torch.zeros((1,), dtype=torch.int64)
        