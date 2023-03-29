import os
import sys
import math
import time
import yaml
import random

import torch
import torchvision
import numpy as np
import pandas as pd
# import skimage
# import skimage.io
# import matplotlib.pyplot as plt
from collections import defaultdict


from torchvision.transforms import ToTensor
from torchvision.models.detection.roi_heads import paste_masks_in_image

from . import LOGGER, load_cfg
from .engines.augmentations import resample_segments, box_candidates
from .engines.image_utils import *
from .engines.torch_utils import collate_fn, torch_distributed_zero_first


META_INFO = {
    'detSC': {
        'classes': ['tumor', 'stromal', 'sTILs', 'other'],
        'labels_color': {-100: '#949494', 1: '#00ff00', 2: '#ff0000', 3: '#0000ff', 4: '#0094e1'},
        'labels_text': {-100: 'unlabeled', 1: 'tumor', 2: 'stromal', 3: 'sTILs', 4: 'other'}
    }
}
# COLORS_SEG = {
#     'bg': [255, 255, 255],
#     'tumor_area': [0, 255, 0], 
#     'stroma_area': [255, 0, 0], 
#     'necrosis_area': [0, 148, 255], 
#     'blood_area': [255, 0, 255],
#     'steatosis_area': [0, 255, 255], 
#     'pseudogland_gap': [255, 255, 0], 
#     'germinal_center': [148, 0, 255], 
#     'lymphocyte_cluster': [0, 0, 255],
# }

# COLORS_DET = {
#     'tumor': [0, 255, 0],  # Liver
#     'stroma': [255, 0, 0],  # Stroma
#     'lymphocyte': [0, 0, 255],  # Lymphocyte
#     'macrophage': [255, 255, 0],  # Macrophage
#     'blood': [255, 0, 255],  # Blood cell
#     'dead': [0, 148, 225],  # Dead nuclei
#     "unlabeled": [148, 148, 148],  # unlabeled in case
# }


def pad_annotation(ann, pad_var):
    new_ann = {**ann}  # avoid inplace modification
    if 'size' in ann:
        new_h = ann['size'][0] + pad_var[0][0] + pad_var[0][1]
        new_w = ann['size'][1] + pad_var[1][0] + pad_var[1][1]
        new_ann['size'] = update_size(ann['size'], [new_h, new_w])
    if 'boxes' in ann:
        new_ann['boxes'] = ann['boxes'] + np.array([pad_var[1][0], pad_var[0][0], pad_var[1][0], pad_var[0][0]], np.float32)
    if 'masks' in ann:
        new_masks = []
        for mask in ann['masks']:
            if mask is not None:
                new_m = mask.poly()
                new_m.size = [mask.size[0]+pad_var[0][0]+pad_var[0][1], mask.size[1]+pad_var[1][0]+pad_var[1][1]]
                new_m.m = [_ + np.array([pad_var[1][0], pad_var[0][0]]) for _ in new_m.m]
                new_masks.append(new_m.convert(mask.mode))
            else:
                new_masks.append(None)
        new_ann['masks'] = new_masks
        # new_ann['masks'] = np.pad(ann['masks'], pad_var, mode=mode, **pars)
    return new_ann
    

def pad_image_target(image, target, pad_width, mode='constant', **kwargs):
    """ Pad image and target by pad_width. 
        Use numpy pad mode on channel last format: [top, bottom, left, right, front, back]. 
        Assume each annotation are recorded under image shape.
    """
    if np.array(pad_width).sum():
        if mode == 'constant':
            pars = {'constant_values': kwargs.get('cval', 0.0)}
        elif mode == 'linear_ramp':
            pars = {'end_values': kwargs.get('end_values', 0.0)}
        elif mode == 'reflect' or mode == 'symmetric':
            pars = {'reflect_type': kwargs.get('reflect_type', 'even')}
        else:
            pars = {'stat_length': kwargs.get('stat_length', None)}

        if image is not None:
            pad_width = pad_width + [(0, 0)] * (image.ndim - len(pad_width))
            pad_var = pad_width[:image.ndim]
            image = np.pad(image, pad_var, mode=mode, **pars)
        if target is not None:
            pad_var = pad_width
            target = {**target}
            if 'size' in target:
                new_h = target['size'][0] + pad_var[0][0] + pad_var[0][1]
                new_w = target['size'][1] + pad_var[1][0] + pad_var[1][1]
                target['size'] = update_size(target['size'], [new_h, new_w])
            if 'anns' in target:
                target['anns'] = {k: [pad_annotation(ann, pad_var) for ann in v] 
                                  for k, v in target['anns'].items()}

    return image, target


def pad_image_target_if_needed(image, target, size, pos='center', mode='constant', **kwargs):
    """ Pad image and target to given size if needed. """
    pad_width = get_pad_width(image.shape, output_size=get_size(size), pos=pos)
    return pad_image_target(image, target, pad_width, mode='constant', **kwargs)


def remove_invalid_objects(ann, image_size=None, filter_fn=None):
    image_size = get_size(image_size)
    new_ann = {**ann}
    if image_size is not None:
        h, w = image_size[0], image_size[1]
        new_ann['boxes'] = np.clip(ann['boxes'], 0, [w, h, w, h])

    if filter_fn is None:
        filter_fn = lambda x: (x['boxes'][:, 0] < x['boxes'][:, 2]) & (x['boxes'][:, 1] < x['boxes'][:, 3])

    keep_idx = np.where(filter_fn(ann))[0]
    new_ann['boxes'] = new_ann['boxes'][keep_idx]
    new_ann['labels'] = new_ann['labels'][keep_idx]
    if 'masks' in new_ann:
        new_ann['masks'] = [new_ann['masks'][_] for _ in keep_idx]

    return new_ann


def crop_annotation(ann, crop_var):
    new_ann = {**ann}
    if 'size' in ann:
        new_h = ann['size'][0] - crop_var[0][0] - crop_var[0][1]
        new_w = ann['size'][1] - crop_var[1][0] - crop_var[1][1]
        new_ann['size'] = update_size(ann['size'], [new_h, new_w])
    if 'boxes' in ann:
        new_ann['boxes'] = ann['boxes'] - np.array([crop_var[1][0], crop_var[0][0], crop_var[1][0], crop_var[0][0]], np.float32)
    if 'masks' in ann:
        new_masks = []
        for mask in ann['masks']:
            if mask is not None:
                new_m = mask.poly()
                new_m.size = [mask.size[0]-crop_var[0][0]-crop_var[0][1], mask.size[1]-crop_var[1][0]-crop_var[1][1]]
                new_m.m = [np.clip(_ - np.array([crop_var[1][0], crop_var[0][0]], np.float32), [0., 0.], [new_m.size[1], new_m.size[0]]) for _ in new_m.m]
                new_masks.append(new_m.convert(mask.mode))
            else:
                new_masks.append(None)
        new_ann['masks'] = new_masks
        # new_ann['masks'] = skimage.util.crop(ann['masks'], crop_var, copy=False)

    return new_ann


def crop_image_target(image, target, crop_width, remove_invalid=True):
    """ Crop image and target by crop_width. 
        Use numpy crop mode on channel last format: [top, bottom, left, right, front, back].
        Assume each annotation are recorded under image shape.
    """
    if np.array(crop_width).sum():
        if image is not None:
            crop_width = crop_width + [(0, 0)] * (image.ndim - len(crop_width))
            crop_var = crop_width[:image.ndim]
            image = skimage.util.crop(image, crop_var, copy=False, order='K')
        if target is not None:
            target = {**target}
            crop_var = crop_width
            if 'size' in target:
                new_h = target['size'][0] - crop_var[0][0] - crop_var[0][1]
                new_w = target['size'][1] - crop_var[1][0] - crop_var[1][1]
                target['size'] = update_size(target['size'], [new_h, new_w])
            if 'anns' in target:
                target['anns'] = {k: [crop_annotation(ann, crop_var) for ann in v] 
                                  for k, v in target['anns'].items()}
                if remove_invalid and image is not None:
                    target['anns'] = {k: [remove_invalid_objects(ann, image.shape) for ann in v] 
                                      for k, v in target['anns'].items()}

    return image, target


def crop_image_target_if_needed(image, target, size, pos='center', remove_invalid=True):
    """ Crop image and target to given size if needed. """
    crop_width = get_crop_width(image.shape, output_size=get_size(size), pos=pos)
    return crop_image_target(image, target, crop_width, remove_invalid=remove_invalid)


def rescale_annotation(ann, size=None, scale=1.0):
    if size is not None:
        if isinstance(size, numbers.Number):
            size = [size, size]
        size = np.array(size, np.int32)
    if scale is not None:
        if isinstance(scale, numbers.Number):
            scale = [scale, scale]
        scale = np.array(scale, np.float32)
        if (scale == 1.0).all():
            scale = None

    new_ann = {**ann}
    if size is not None and 'size' in ann:
        new_ann['size'] = update_size(ann['size'], size)
    if scale is not None and 'boxes' in ann:
        new_ann['boxes'] = ann['boxes'] * scale[[1,0,1,0]]
    if 'masks' in ann:
        new_masks = []
        for mask in ann['masks']:
            if mask is not None:
                new_m = mask.poly()
                if size is not None:
                    new_m.size = [size[0], size[1]]
                if scale is not None:
                    new_m.m = [_ * scale[[1,0]] for _ in new_m.m]
                new_masks.append(new_m.convert(mask.mode))
            else:
                new_masks.append(None)
        new_ann['masks'] = new_masks
        # dtype = ann['masks'].dtype
        # new_ann['masks'] = skimage.transform.rescale(ann['masks'], scale, order=0, preserve_range=True)

    return new_ann


def resize_image_target(image, target, size=None, scale=None, order=1):
    """ Resize/Rescale image and target. 
        Assume each annotation are recorded under image shape.
    """
    assert size is not None or scale is not None, f"size and scale cannot both be None."

    h, w = image.shape[:-1]
    output_size, scale_factor, recompute_scale_factor = _process_size_and_scale_factor([h, w], size, scale)
    h_new, w_new = output_size
    scale_h, scale_w = scale_factor
    order = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 3: cv2.INTER_CUBIC}[order]

    if h_new != h or w_new != w:
        if image is not None:
            image = cv2.resize(image, (w_new, h_new), interpolation=order)
            # too slow
            # dtype = image.dtype
            # image = skimage.transform.rescale(image, scale, order=1, anti_aliasing=True, multichannel=True)
            # image = img_as(dtype)(image)
        if target is not None:
            target = {**target}
            if 'size' in target:
                target['size'] = update_size(target['size'], [h_new, w_new])
            if 'anns' in target:
                target['anns'] = {k: [rescale_annotation(ann, [h_new, w_new], [scale_h, scale_w]) for ann in v] 
                                  for k, v in target['anns'].items()}

    return image, target


# Borrow from pytorch interpolate.
def _process_size_and_scale_factor(input_size, size=None, scale_factor=None):
    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    dim = len(input_size)
    if size is not None and scale_factor is not None:
        raise ValueError(f"Only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "size shape must match input shape. " "Input is {}D, size is {}".format(dim, len(size))
                )
            output_size = size
        else:
            output_size = [size for _ in range(dim)]
        recompute_scale_factor = True

    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "scale_factor shape must match input shape. "
                    "Input is {}D, scale_factor is {}".format(dim, len(scale_factor))
                )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
        output_size = [d * scale for scale, d in zip(scale_factors, input_size)]
        recompute_scale_factor = np.any(output_size % np.round(output_size))
    else:
        raise ValueError(f"Either size or scale_factor should be defined")

    ## recompute scale factor
    output_size = [round(_) for _ in output_size]
    if recompute_scale_factor:
        scale_factors = [d_new/d for d_new, d in zip(output_size, input_size)]

    return output_size, scale_factors, recompute_scale_factor


def random_projective(image, target, hyp, output_shape=None, cval=0.):
    h, w = image.shape[:-1]
    h_new, w_new = (h, w) if output_shape is None else get_size(output_shape)

    tf_pars = random_transform_pars([h, w], [h_new, w_new], hyp)
    M = estimate_matrix(tf_pars)

    # Transform label coordinates
    image = warp_image(image, M, output_size=[h_new, w_new], cval=cval)
    if target is not None:
        target = {**target}

        for task_id, ann_list in target['anns'].items():
            new_ann_list = []
            for ann in ann_list:
                boxes = ann['boxes'].numpy()
                labels = ann['labels']
                masks = ann['masks']
                new_boxes = np.zeros((len(boxes), 4))
                new_masks = []                
                for idx, (box, mask) in enumerate(zip(boxes, masks)):
                    polys = resample_segments(mask.m) if mask else [box[[0,1,0,3,2,3,2,1]].reshape(-1, 2)]
                    new_polys = [warp_coords(p, M) for p in polys]
                    new_mask = Mask(new_polys, [h_new, w_new], 'poly')
                    new_masks.append(new_mask if mask else None)
                    new_boxes[idx] = new_mask.box()

                # filter candidates
                area_thr = np.array([0.01 if _ else 0.10 for _ in new_masks])
                keep_idx = box_candidates(box1=boxes.T * tf_pars['scale'], box2=new_boxes.T, area_thr=area_thr)
                keep_idx = np.where(keep_idx)[0]
                new_ann_list.append({**ann, 'size': [h_new, w_new], 'boxes': torch.as_tensor(new_boxes[keep_idx]), 
                                     'labels': labels[keep_idx], 'masks': [new_masks[_] for _ in keep_idx]})
            target['anns'][task_id] = new_ann_list

    return image, target


def hflip_image_target(image, target):
    if image is not None:
        image = image[:, ::-1, ...]
    if target is not None:
        target = {**target}
        if 'anns' in target:
            target['anns'] = {k: [hflip_annotation(ann) for ann in v] 
                              for k, v in target['anns'].items()}

    return image, target


def hflip_annotation(ann):
    new_ann = {**ann}
    if 'boxes' in ann:
        h, w = ann.get('size', [1.0, 1.0])  # treat as normalized box if no size
        new_ann['boxes'] = abs(ann['boxes'][:,[2,1,0,3]] - np.array([w,0,w,0]))
    if 'masks' in ann:
        new_ann['masks'] = [mask.hflip() if mask is not None else None for mask in ann['masks']]

    return new_ann


def vflip_image_target(image, target):
    if image is not None:
        image = image[::-1, ...]
    if target is not None:
        target = {**target}
        if 'anns' in target:
            target['anns'] = {k: [vflip_annotation(ann) for ann in v] 
                              for k, v in target['anns'].items()}

    return image, target


def vflip_annotation(ann):
    new_ann = {**ann}
    if 'boxes' in ann:
        h, w = ann.get('size', [1.0, 1.0])  # treat as normalized box if no size
        new_ann['boxes'] = abs(ann['boxes'][:,[0,3,2,1]] - np.array([0,h,0,h]))
    if 'masks' in ann:
        new_ann['masks'] = [mask.vflip() if mask is not None else None for mask in ann['masks']]

    return new_ann


def transpose_image_target(image, target):
    if image is not None:
        image = image.swapaxes(0, 1)
    if target is not None:
        target = {**target}
        if 'size' in target:
            target['size'] = update_size(target['size'], [target['size'][1], target['size'][0]])
        if 'anns' in target:
            target['anns'] = {k: [transpose_annotation(ann) for ann in v] 
                              for k, v in target['anns'].items()}

    return image, target


def transpose_annotation(ann):
    new_ann = {**ann}
    new_ann['size'] = update_size(ann['size'], [ann['size'][1], ann['size'][0]])
    if 'boxes' in ann:
        new_ann['boxes'] = ann['boxes'][:, [1,0,3,2]]
    if 'masks' in ann:
        new_ann['masks'] = [mask.t() if mask is not None else None for mask in ann['masks']]

    return new_ann


def random_flip(image, target, hflip=0.5, vlip=0.5, transpose=0.5):
    if random.random() < hflip:
        image, target = hflip_image_target(image, target)

    if random.random() < vlip:
        image, target = vflip_image_target(image, target)

    if random.random() < transpose:
        image, target = transpose_image_target(image, target)

    return image, target


def filter_annotation(ann, keep_indices):
    res = {**ann}
    if 'boxes' in ann:
        res['boxes'] = ann['boxes'][keep_indices]
    if 'labels' in ann:
        res['labels'] = ann['labels'][keep_indices]
    if 'masks' in ann:
        res['masks'] = [ann['masks'][_] for _ in keep_indices]
    if 'area' in ann:
        res['area'] = ann['area'][keep_indices]

    return res


def merge_annotations(anns, size):
    """ Merge annotations under same task_id. 
        anns will be: {'task_id1': [{'boxes', }, {}], 'task_id2': [], ...}
        Assume all annotations are under same ROI and have same size.
    """
    new_anns = {}
    for k, ann_list in anns.items():
        boxes, labels, masks = [], [], []
        for ann in ann_list:
            boxes.extend(list(ann['boxes']))
            labels.extend(list(ann['labels']))
            masks.extend(list(ann['masks']) if 'masks' in ann else [None] * len(ann['boxes']))
        new_anns[k] = [{'roi': [0, 0, size[1], size[0]], 'size': size, 
                        'boxes': torch.stack(boxes), 'labels': torch.stack(labels), 'masks': masks}]

    return new_anns


def target_to_tensors(x, normalize_box=False, mask_order=1):
    """ If n_classes is give, will transfer to one_hot encoding. """
    res = {'image_id': torch.as_tensor([x['image_id']], dtype=torch.int64), 
           'size': torch.as_tensor(x['size'], dtype=torch.int64),
           'anns': defaultdict(list), # 'anns': [], 
          }
    mask_order = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 3: cv2.INTER_CUBIC}[mask_order]

    for task_id, anns in x['anns'].items():
        for ann in anns:
            if task_id.startswith('det'):
                ann_new = {
                    # 'roi': torch.as_tensor(ann['roi'], dtype=torch.int64),
                    'size': torch.as_tensor(ann['size'], dtype=torch.int64),
                    # 'image_id': torch.as_tensor([ann['image_id']], dtype=torch.int64),
                    'boxes': torch.as_tensor(ann['boxes'], dtype=torch.float32), 
                    'labels': torch.as_tensor(ann['labels'], dtype=torch.int64),
                    # 'iscrowd': torch.zeros((len(ann['boxes']),), dtype=torch.int64),
                }
                if 'masks' in ann:
                    M = 28
                    new_masks = torch.zeros((len(ann['masks']), M, M), dtype=torch.float32)
                    for idx, (mask, box) in enumerate(zip(ann['masks'], ann['boxes'])):
                        if mask:
                            box_mask = mask.mask().m.astype(np.float32)
                            if box_mask.sum() >= 25:  # ignore some artifacts
                                x0, y0, x1, y1 = box.round().type(torch.int64)
                                try:
                                    m = cv2.resize(box_mask[y0:y1, x0:x1], (M, M), interpolation=mask_order)
                                except:
                                    m = np.zeros([M, M])
                                new_masks[idx] = torch.as_tensor(m)
                    ann_new['masks'] = new_masks
                # normalize boxes by size
                if normalize_box:
                    ann_new['boxes'] = ann_new['boxes'] / ann_new['size'][[1, 0, 1, 0]]
            elif task_id.startswith('seg'):
                ann_new = {
                    # 'roi': torch.as_tensor(ann['roi'], dtype=torch.int64),
                    'size': torch.as_tensor(ann['size'], dtype=torch.int64),
                    # 'image_id': torch.as_tensor([ann['image_id']], dtype=torch.int64),
                }
                m = [_.mask().m for _ in ann['masks']]
                ann_new['masks'] = torch.as_tensor(m, dtype=torch.float32)
                ann_new['areas'] = torch.sum(ann_new['masks'], dim=(1, 2))
            elif task_id.startwith('cl'):
                ann_new = {
                    # 'roi': torch.as_tensor(ann['roi'], dtype=torch.int64),
                    'size': torch.as_tensor(ann['size'], dtype=torch.int64),
                    # 'image_id': torch.as_tensor([ann['image_id']], dtype=torch.int64),
                    'label': torch.as_tensor([ann['labels']], dtype=torch.int64),
                }
            else:
                raise ValueError(f"Task: {task_id} is not supported!")

            res['anns'][task_id].append(ann_new)

    return res


def train_proc(img, ann, hyp):
    # Color Augmentation and Normalization
    color_aug = hyp.get('color_aug', 'hsv')
    if color_aug == 'jitter':
        img, = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.15, 0.1), p=1.0)([img])
    elif color_aug == 'hsv':
        img = random_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'], p=1.0)
    elif color_aug == 'dodge':
        img, = ColorDodge(global_mean=0.01, channel_mean=0.01, channel_sigma=0.2, p=1.0)([img])
    
    # projective transformation
    cval = hyp.get('cval', 0.5)
    diag = int(np.linalg.norm(img.shape[:-1]).round() + 10) # keep whole image with border
    out_size = hyp.get('patch_size', diag)
    img, ann = random_projective(img, ann, hyp, output_shape=out_size, cval=cval)
    img, ann = random_flip(img, ann, hflip=hyp['fliplr'], vlip=hyp['flipud'], transpose=hyp['transpose'])

    return img, ann


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data, hyp, processor=train_proc, normalize_box=True, in_memory=True):
        self.kwargs = self.hyp = hyp
        self.processor = processor
        # self.image_reader = skimage.io.imread
        self.albu = self._albumentation_tf()
        self.normalize_box = normalize_box

        self.root = './'
        if isinstance(data, str):
            self.root = os.path.dirname(data)
            data = pd.read_csv(data)
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')

        ## map images with annotations
        self.images = []
        self.annotations = []
        self.ann_cache = []
        self.id_map = {} # {image_id_name: pos in self.images, ann_id_name: pos in self.data}
        for ann_idx, info in enumerate(data):
            image_path, image_id, ann_id = info['image_path'], info['image_id'], info['ann_id']
            # image_global_label = 'tumor'  # apply info['image_label'] for image classification

            # Add new image to self.images and self.id_map
            if image_id not in self.id_map:
                self.id_map[image_id] = len(self.images)
                new_image = {'image_id': image_id, 'image_path': image_path, 'anns': [], 'kwargs': {}, }
                self.images.append(new_image)
            image_idx = self.id_map[image_id]

            # add new annotations to self.annotations and self.id_map
            self.id_map[ann_id] = ann_idx
            self.annotations.append({**info, 'image_idx': image_idx,})
            self.images[image_idx]['anns'].append(ann_idx)

            if in_memory:
                self.ann_cache.append(self.load_annotation(ann_idx))

    def _albumentation_tf(self):
        try:
            import albumentations as A
            # check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            tf = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                # A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.1),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)
            ])
        except:
            tf = None
        
        return tf

    def __len__(self):
        return len(self.images)

    def load_annotation(self, ann_idx):
        ann_info = self.annotations[ann_idx]
        ann_id = ann_info['ann_id']
        task_id = ann_info['task_id']

        anns = torch.load(os.path.join(self.root, ann_info['ann_path']))
        if task_id.startswith('det'):
            anns['masks'] = [Mask(_, anns['size'].tolist(), ann_info['mask_mode']) for _ in anns['masks']] # if _ else None 
        elif task_id.startswith('seg'):
            anns['masks'] = [Mask(_, anns['size'].tolist(), ann_info['mask_mode']) for _ in anns['masks']] # if _ else None 
        elif task_id.startswith('cl'):
            anns = anns
            # label = image_info['ann_path']
            # ann = {**ann, 'labels': label}
        else:
            raise ValueError(f"Task: {task_id} is not supported!")
        del anns['roi']  # remove roi here
        anns['image_id'] = ann_idx  # add annotation id

        return anns

    def load_image_and_target(self, idx):
        # load image
        image_info = self.images[idx]
        image = cv2.imread(os.path.join(self.root, image_info['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = self.image_reader(os.path.join(self.root, image_info['image_path']))
        # image = img_as(np.float32)(rgba2rgb(image))   # [..., ::-1]
        kwargs = {**self.kwargs, **image_info['kwargs'], 'image_idx': idx}

        # load annotations
        target = {'image_id': idx, 'size': image.shape[:-1], 'anns': defaultdict(list),}
        for ann_idx in image_info['anns']:
            ann_info = self.annotations[ann_idx]
            if self.ann_cache:
                anns = self.ann_cache[ann_idx]
            else:
                anns = self.load_annotation(ann_idx)
            task_id = ann_info['task_id']
            target['anns'][task_id].append(anns)

        return image, target, kwargs

    def __getitem__(self, idx):
        patch_size = self.kwargs['patch_size']
        img_size = self.kwargs['img_size']
        keep_res = self.kwargs.get('keep_res', -1)
        border = self.kwargs.get('border', 10)
        cval = self.kwargs.get('cval', 0.5)

        if self.processor is not None:  # training
            k_mosaic = int(self.kwargs['k_mosaic'])
            indices = [idx] + random.choices(range(len(self)), k=k_mosaic**2-1)
            img_block, ann_dict = [[None] * k_mosaic for _ in range(k_mosaic)], defaultdict(list)
            # speed = [time.time()]
            for rc, img_idx in enumerate(random.sample(indices, len(indices))):
                r, c = rc // k_mosaic, rc % k_mosaic
                img, tgt, kwargs = self.load_image_and_target(img_idx)
                # display_image_and_target(img, tgt, META_INFO)
                # speed.append(time.time())
                if self.processor is not None:  # each image, target are under patch_size
                    img = self.albu(image=img)['image']
                    img, tgt = self.processor(img, tgt, self.kwargs)
                # speed.append(time.time())

                if keep_res > 0.:  # keep different image under same resolution
                    img, tgt = resize_image_target(img, tgt, scale=keep_res)
                    output_size = int(patch_size * keep_res)
                    # pos = (('t' if r else 'b') + ('r' if c else 'l'))
                    img, tgt = pad_image_target_if_needed(img, tgt, output_size, pos='random', mode='constant', cval=cval)
                    img, tgt = crop_image_target_if_needed(img, tgt, output_size, pos='random', remove_invalid=True)
                    if border:
                        pad_width = [(border, border), (border, border)]
                        img, tgt = pad_image_target(img, tgt, pad_width, mode='constant', cval=cval)
                else:  # ignore resolution, resize images to the same size
                    img, tgt = resize_image_target(img, tgt, size=patch_size)

                # pad target according to mosaic row/col
                pad_var = [(r * img.shape[0], (k_mosaic-1-r) * img.shape[0]), 
                           (c * img.shape[1], (k_mosaic-1-c) * img.shape[1])]
                _, tgt = pad_image_target(None, tgt, pad_var, mode='constant')

                img_block[r][c] = img
                for k, v in tgt['anns'].items():
                    ann_dict[k].extend(v)
                # speed.append(time.time())
                # print(f"one patch speed: ", [e-s for s, e in zip(speed[:-1], speed[1:])])
                # speed = [time.time()]

            # s = time.time()
            image = np.concatenate([np.concatenate(row, axis=1) for row in img_block], axis=0)
            target = {'image_id': idx, 'size': image.shape[:-1], 'anns': merge_annotations(ann_dict, image.shape[:-1])}
            # diag = int(np.linalg.norm(img.shape[:-1]).round()) # keep whole image with border
            # image, target = self.processor(image, target, self.kwargs)
            image, target = crop_image_target_if_needed(image, target, img_size, pos='random', remove_invalid=True)
            # print(f"merge patches:", time.time() - s)
        else:  # validation
            image, target, kwargs = self.load_image_and_target(idx)
            if keep_res > 0.:
                image, target = resize_image_target(image, target, scale=keep_res)
                image, target = pad_image_target_if_needed(image, target, img_size, pos='center', mode='constant', cval=cval)
                image, target = crop_image_target_if_needed(image, target, img_size, pos='center', remove_invalid=True)
            else:  # ignore resolution, resize images to the same size
                image, target = resize_image_target(image, target, size=img_size)

        # remove very small objects
        # s = time.time()
        filter_fn = lambda x: (x['boxes'][:,0] < x['boxes'][:,2]-10) & (x['boxes'][:,1] < x['boxes'][:,3]-10)
        target['anns'] = {k: [remove_invalid_objects(ann, image.shape, filter_fn=filter_fn) for ann in v] 
                          for k, v in target['anns'].items()}
        # print(f"filter small objects:", time.time() - s)

        # s = time.time()
        image = ToTensor()(image.copy()).type(torch.float32)
        # print(f"convert image to tensor:", time.time() - s)

        # s = time.time()
        target = target_to_tensors(target, normalize_box=self.normalize_box)
        # print(f"convert target to tensor:", time.time() - s)

        # display_image_and_target(image, target, meta_info)
        return image, target


def display_image_and_target(image, target, meta_info, plot=True, verbose=1, **kwargs):
    ## display by tasks:
    image_id = target['image_id']
    image_size = target['size']
    if verbose:
        print(f"Image: id={image_id}, size={image_size}, stats={image_stats(image)}")
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()
    # plt.imshow(image)
    # plt.show()
    
    n_plots = sum([len(_) for _ in target['anns'].values()])
    r, c = ((n_plots + 3)//4, 4) if n_plots > 4 else (1, n_plots)
    # fig, axes = plt.subplots(r, c, figsize=(6 * c, 6 * r))
    fig, axes = plt.subplots(r, c, figsize=(12, 12))
    axes = axes.ravel() if n_plots > 1 else [axes]

    idx = 0
    for task_id, anns in target['anns'].items():
        if verbose:
            print(f"Task: {task_id}")
        if task_id.startswith('seg'):
            for ann in anns:
                if 'roi' in ann:
                    x0, y0, x1, y1 = ann['roi']
                    patch = image[int(y0):int(y1), int(x0):int(x1), :]
                else:
                    x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
                    patch = image
                if verbose:
                    print(f"patch_roi={(x0, y0, x1, y1)}, patch_stats={image_stats(patch)}, ann_size={ann['size']}")
                labels_color = meta_info.get(task_id, {}).get('labels_color', None)
                labels_text = meta_info.get(task_id, {}).get('labels_text', None)
                h_ann, w_ann = np.array(ann['size'])
                patch_resize = cv2.resize(patch, (w_ann, h_ann), interpolation=cv2.INTER_LINEAR)
                # patch_resize = skimage.transform.resize(patch, ann['size'], order=3) # resize patch to annotation size

                axes[idx].imshow(patch_resize)
                axes[idx].set_title(task_id)
                overlay_segmentations(
                    axes[idx], masks=ann['masks'], labels=None, 
                    labels_color=labels_color, labels_text=labels_text,
                )
                idx += 1

        elif task_id.startswith('det'):
            for ann in anns:
                if 'roi' in ann:
                    x0, y0, x1, y1 = ann['roi']
                    patch = image[int(y0):int(y1), int(x0):int(x1), :]
                else:
                    x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
                    patch = image
                if verbose:
                    print(f"patch_roi={(x0, y0, x1, y1)}, patch_stats={image_stats(patch)}, ann_size={ann['size']}")
                labels_color = meta_info.get(task_id, {}).get('labels_color', None)
                labels_text = meta_info.get(task_id, {}).get('labels_text', None)
                h_ann, w_ann = np.array(ann['size'])
                patch_resize = cv2.resize(patch, (w_ann, h_ann), interpolation=cv2.INTER_LINEAR)
                # patch_resize = skimage.transform.resize(patch, ann['size'], order=3) # resize patch to annotation size

                boxes, labels, masks = ann['boxes'], ann['labels'], None
                if labels.dim() > 1:  # one_hot vectors
                    labels = [int(torch.where(x)[-1][-1]) if x[1:].sum() else -100 for x in labels]
                if boxes.max() <= 1.0:  # normalized box
                    boxes *= np.array([w_ann, h_ann, w_ann, h_ann])
                if 'masks' in ann:
                    masks = ann['masks']
                    if isinstance(masks, torch.Tensor) and (masks.shape[-1] != w_ann or masks.shape[-2] != h_ann):
                        # masks is saved as K*M*M, paste back to h_ann, w_ann image
                        masks = paste_masks_in_image(masks[:, None, ...], boxes, (h_ann, w_ann), padding=1).squeeze(1)

                axes[idx].imshow(patch_resize)
                axes[idx].set_title(task_id)
                overlay_detections(
                    axes[idx], bboxes=boxes, labels=labels, masks=masks,
                    labels_color=labels_color, labels_text=labels_text,
                    **kwargs, # show_bboxes=False, show_texts=False,
                )
                idx += 1
    
    if isinstance(plot, str) and plot:
        plt.savefig(plot)
        plt.close()
    elif plot is True:
        plt.show()
        plt.close()


class InfiniteDataLoader(torch.utils.data.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def create_dataloader(path, batch_size, hyp=None, augment=False, cache=True, 
                      rank=-1, workers=8, image_weights=False, prefix='', shuffle=False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = TorchDataset(path, hyp, processor=(train_proc if augment else None), 
                               normalize_box=augment, in_memory=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle and sampler is None,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=collate_fn)

    return dataloader, dataset


def load_dataset_info(data_path):
    # load data.yaml
    # root = os.path.dirname(os.path.abspath(data_path))
    root = './'
    data = load_cfg(data_path)
    for k in 'train', 'val', 'test':
        if k in data:
            data[k] = os.path.join(root, data[k])
    meta_info = load_cfg(os.path.join(root, data['meta_info']))

    # load tasks metainfo
    tasks = {}
    for task_id in data['tasks']:
        tasks[task_id] = meta_info[task_id]
    data['meta_info'] = tasks

    return data  # dictionary

