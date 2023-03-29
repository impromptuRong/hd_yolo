import os
import sys
import math
import time
import yaml

import torch
import torchvision
import numpy as np
import pandas as pd
# import skimage
# import skimage.io
# import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor
from torchvision.models.detection.roi_heads import paste_masks_in_image
from DIPModels.utils_g.utils_image import *


COLORS_SEG = {
    'bg': [255, 255, 255],
    'tumor_area': [0, 255, 0], 
    'stroma_area': [255, 0, 0], 
    'necrosis_area': [0, 148, 255], 
    'blood_area': [255, 0, 255],
    'steatosis_area': [0, 255, 255], 
    'pseudogland_gap': [255, 255, 0], 
    'germinal_center': [148, 0, 255], 
    'lymphocyte_cluster': [0, 0, 255],
}

COLORS_DET = {
    'tumor': [0, 255, 0],  # Liver
    'stroma': [255, 0, 0],  # Stroma
    'lymphocyte': [0, 0, 255],  # Lymphocyte
    'macrophage': [255, 255, 0],  # Macrophage
    'blood': [255, 0, 255],  # Blood cell
    'dead': [0, 148, 225],  # Dead nuclei
    "unlabeled": [148, 148, 148],  # unlabeled in case
}


#######################################
## Define dataset with the following structure
# image = np.zeros([4096, 4096, 3])  # 40x large patch
# # (feature_1: 1024*1024 10x, feature_1: 512*512 5x feature_3: 256*256 2.5x feature_4: 128*128 1.25x)
# target = {
#     "image_id": image_idx,
#     "size": image_size, # (4000, 4000)
#     "anns": {
#         "cl5x": [{ # a classification annotation, task id startswith("cl")
#             "roi": [roi_w0, roi_h0, roi_w1, roi_h1], # coords in original image (40x)
#             "size": ann_size, # not really matter for classification
#             "image_id": ann_idx, 
#             "labels": labels, # k*1
#         },],
#         "det40x": [{ # a detection annotation with N objects, task id startswith("det")
#             "roi": [roi_w0, roi_h0, roi_w1, roi_h1], # coords in original image (40x)
#             "size": ann_size, # annotated mask size (may not same as roi): a 2000x2000 roi with mask size=500x500
#             "image_id": ann_idx, 
#             "boxes": [box_w0, box_h0, box_w1, box_h1], # relative coords in patch_size, N*4
#             "labels": labels, # N,
#             "area": area, # N,
#             "iscrowd": iscrowd,
#             "masks": masks, # N * h * w, masks with "size", not "roi"
#         },],
#         "seg10x": [{ # a segmentation annotation, task id startswith("seg")
#             "roi": [roi_w0, roi_h0, roi_w1, roi_h1], # coords in original image (40x)
#             "size": ann_size, # annotated mask size (may not same as roi): a 2000x2000 roi with mask size=500x500
#             "image_id": ann_idx, 
#             "masks": masks, # masks same size as size, # N * h * w
#             "labels": labels, # N,
#         },],
#     ]
# }
#######################################

class ImageAmp(object):
    def __init__(self, img, amp):
        self.img = img
        self.amp = amp
    
    def to(self, device):
        self.img = self.img.to(device)
        return self


def collate_fn(batch):
    return tuple(zip(*batch))


def pad_image_and_target(image, target, size=None, pad_width=None, pos='center', mode='constant', stride=1, **kwargs):
    """ Pad target only apply on 'size' and ann['roi']. 
        ann['size'] won't be changed, and will keep 
        ann['boxes'], ann['masks'], etc. in its original scale.
        stride can avoid floating roi when resizing.
    """
    assert image.shape[0] == target['size'][0] and image.shape[1] == target['size'][1], f"image and target are not consistent."
    h, w = target['size'][0], target['size'][1]
    
    if pad_width is None:
        assert size is not None, f"pad_width and size cannot be both None."
        pad_width = get_pad_width(image.shape, size, pos='random', stride=stride)
    
    image = pad(image, pad_width=pad_width, mode=mode, **kwargs)
    h_new, w_new = image.shape[:-1]
    
    # avoid overriding original target
    new_target = {**target, 'size_ori': [h, w], 'size': [h_new, w_new], 'anns': defaultdict(list)}
    for task_id in target['anns']:
        for ann in target['anns'][task_id]:
            new_ann = {**ann, 'roi_ori': ann['roi'], 
                       'roi': [ann['roi'][0]+pad_width[1][0], ann['roi'][1]+pad_width[0][0], 
                               ann['roi'][2]+pad_width[1][0], ann['roi'][3]+pad_width[0][0],]
                      }
            new_target['anns'][task_id].append(new_ann)
    
    return image, new_target


def pad_annotation(ann, pad_width=None, pos='center', mode='constant', **kwargs):
    """ Pad annotation. is not changed. """
    x0, y0, x1, y1 = ann['roi']
    h_roi, w_roi = float(y1-y0), float(x1-x0)
    h, w = int(ann['size'][0]), int(ann['size'][1])
    
    # get scale for roi_size -> ann_size
    output_size, scale_factor, recompute_scale_factor = _process_size_and_scale_factor([h_roi, w_roi], [h, w])
    scale_h, scale_w = scale_factor
    
    # get pad_width for annotation
    (y0_p, y1_p), (x0_p, x1_p) = pad_width  # roi pad width
    y0_p_ann, y1_p_ann = round(float(y0_p * scale_h)), round(float(y1_p * scale_h))
    x0_p_ann, x1_p_ann = round(float(x0_p * scale_w)), round(float(x1_p * scale_w))
    
    # calculate new roi_size and new ann_size after padding
    x0_new, y0_new, x1_new, y1_new = x0 - x0_p, y0 - y0_p, x1 + x1_p, y1 + y1_p
    h_new, w_new = h + y0_p_ann + y1_p_ann, w + x0_p_ann + x1_p_ann

    new_ann = {**ann, 'roi_ori': ann['roi'], 'roi': [x0_new, y0_new, x1_new, y1_new],
               'size_ori': [h, w], 'size': [h_new, w_new],}
    if h_new != h or w_new != w:  # modify 'boxes' and 'masks'
        if 'boxes' in ann:
            # tensor + np.array is acceptable
            new_ann['boxes'] = ann['boxes'] + np.array([x0_p_ann, y0_p_ann, x0_p_ann, y0_p_ann], np.float32)
        if 'masks' in ann:
            new_ann['masks'] = []
            for mask in ann['masks']:
                if mask.mode.startswith('poly'):
                    m = [p + np.array([x0_p_ann, y0_p_ann]) for p in mask.m]
                elif mask.mode.startswith('rle'):
                    assert h == mask.size[0] and w == mask.size[1]
                    pwd = [(y0_p_ann, y1_p_ann), (x0_p_ann, x1_p_ann)]
                    m = rle_decode(mask.m, mask.size)
                    m = pad(m, pad_width=pwd, mode='constant')
                    m = rle_encode(m)['counts']
                else:
                    pwd = [(y0_p_ann, y1_p_ann), (x0_p_ann, x1_p_ann)]
                    m = pad(m, pad_width=pwd, mode='constant')
                
                new_ann['masks'].append(Mask(m, [h_new, w_new], mask.mode))
        
    return new_ann


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


def resize_target(target, size=None, scale_factor=None):
    """ Resize target only apply on 'size' and ann['roi']. 
        ann['size'] won't be changed, and will keep 
        ann['boxes'], ann['masks'], etc. in its original scale.
    """
    output_size, scale_factor, recompute_scale_factor = _process_size_and_scale_factor(target['size'], size, scale_factor)
    h, w = target['size']
    h_new, w_new = output_size
    scale_h, scale_w = scale_factor
    
    new_target = {**target, 'size_ori': [h, w], 'size': [h_new, w_new], 'anns': defaultdict(list)}
    for task_id in target['anns']:
        for ann in target['anns'][task_id]:
            new_ann = {**ann, 'roi_ori': ann['roi'], 
                       'roi': [ann['roi'][0] * scale_w, ann['roi'][1] * scale_h, 
                          ann['roi'][2] * scale_w, ann['roi'][3] * scale_h,],
                      }
            new_target['anns'][task_id].append(new_ann)
    
    return new_target


def resize_annotation(ann, size=None, scale_factor=None, mask_mode=None, mask_dtype=None, kwargs={}):
    """ Resize annotation. ann['roi'] is not changed. 
        mask_mode: change mask_mode if needed. Ext: switch boolean mask to float after resize.
        kwargs: parameters parsed to skimage.transform.resize
    """
    if size is None and scale_factor is None:  # by default, resize ann to the size of ann['roi']
        size = [float(ann['roi'][3] - ann['roi'][1]), float(ann['roi'][2] - ann['roi'][0])]
    
    output_size, scale_factor, recompute_scale_factor = _process_size_and_scale_factor(ann['size'], size, scale_factor)
    h, w = ann['size']
    h_new, w_new = output_size
    scale_h, scale_w = scale_factor
    
    new_ann = {**ann, 'size_ori': [h, w], 'size': [h_new, w_new]}
    if h_new != h or w_new != w:
        if 'boxes' in ann:
            # tensor * np.array is acceptable
            new_ann['boxes'] = ann['boxes'] * np.array([scale_w, scale_h, scale_w, scale_h], np.float32)
        if 'masks' in ann:
            new_ann['masks'] = []
            for mask in ann['masks']:
                mask = mask.convert(mask_mode, mask_dtype)
                if mask.mode.startswith('poly'):
                    m = [p * np.array([scale_w, scale_h]) for p in mask.m]
                elif mask.mode.startswith('rle'):
                    assert h == mask.size[0] and w == mask.size[1]
                    m = rle_decode(mask.m, mask.size)  # rle only support binary
                    pars = {'order': 0, 'anti_aliasing': False, 'preserve_range': True, **kwargs}
                    m = skimage.transform.resize(m, [h_new, w_new], **pars)
                    m = rle_encode(m)['counts']
                else:
                    pars = {
                        'order': 0 if mask.m.dtype == bool else 1,
                        'anti_aliasing': False if mask.m.dtype == bool else True,
                        'preserve_range': True,
                        **kwargs,
                    }
                    m = skimage.transform.resize(mask.m, (h_new, w_new), **pars)
                new_ann['masks'].append(Mask(m, [h_new, w_new], mask.mode))
    
    return new_ann


def target_to_tensors(x):
    res = {'image_id': torch.as_tensor([x['image_id']], dtype=torch.int64), 
           'size': torch.as_tensor(x['size'], dtype=torch.int64),
           'anns': defaultdict(list), # 'anns': [], 
          }
    
    for task_id, anns in x['anns'].items():
        for ann in anns:
            if task_id.startswith('det'):
                ann_new = {
                    'roi': torch.as_tensor(ann['roi'], dtype=torch.int64),
                    'size': torch.as_tensor(ann['size'], dtype=torch.int64),
                    'image_id': torch.as_tensor([ann['image_id']], dtype=torch.int64),
                    'boxes': torch.as_tensor(ann['boxes'], dtype=torch.float32), 
                    'labels': torch.as_tensor(ann['labels'], dtype=torch.int64),
                    # 'iscrowd': torch.zeros((len(ann['boxes']),), dtype=torch.int64),
                }
                # normalize boxes
                ann_new['boxes'] = ann_new['boxes'] / ann_new['size'][[1, 0, 1, 0]]
                # shrink mask size
                if 'masks' in ann:
                    M = 28
                    m = [_.mask(np.float32).m[None] for _ in ann['masks']]
                    m = torch.as_tensor(m, dtype=torch.float32)
                    rois = [_[None] for _ in ann['boxes']]
                    ann_new['masks'] = torchvision.ops.roi_align(m, rois, (M, M), aligned=True)[:, 0]
                    ann_new['area'] = torch.sum(ann_new['masks'], dim=(1, 2))
                else:
                    boxes = ann['boxes']
                    ann_new['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            elif task_id.startswith('seg'):
                ann_new = {
                    'roi': torch.as_tensor(ann['roi'], dtype=torch.int64),
                    'size': torch.as_tensor(ann['size'], dtype=torch.int64),
                    'image_id': torch.as_tensor([ann['image_id']], dtype=torch.int64),
                }
                m = [_.mask().m for _ in ann['masks']]
                ann_new['masks'] = torch.as_tensor(m, dtype=torch.float32)
                ann_new['areas'] = torch.sum(ann_new['masks'], dim=(1, 2))
            elif task_id.startwith('cl'):
                ann_new = {
                    'roi': torch.as_tensor(ann['roi'], dtype=torch.int64),
                    'size': torch.as_tensor(ann['size'], dtype=torch.int64),
                    'image_id': torch.as_tensor([ann['image_id']], dtype=torch.int64),
                    'label': torch.as_tensor([ann['labels']], dtype=torch.int64),
                }
            else:
                raise ValueError(f"Task: {task_id} is not supported!")

            res['anns'][task_id].append(ann_new)
    
    return res


def filter_annotation(ann, keep_indices):
    res = {}
    if 'boxes' in ann:
        res['boxes'] = ann['boxes'][keep_indices]
    if 'labels' in ann:
        res['labels'] = ann['labels'][keep_indices]
    if 'masks' in ann:
        res['masks'] = ann['masks'][keep_indices]
    if 'area' in ann:
        res['area'] = ann['area'][keep_indices]
    
    return {**targets, **res}


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations, meta_info, processor=None, in_memory=True, **kwargs):
        self.root = root
        self.kwargs = kwargs
        self.processor = processor
        self.image_reader = skimage.io.imread
        
        if isinstance(meta_info, str):
            # with open("./data/meta_info.yaml", "r") as f:
            with open(meta_info, "r") as f:
                meta_info = yaml.safe_load(f)
        self.meta_info = meta_info
        
        if isinstance(annotations, str):
            annotations = pd.read_csv(annotations)
        if isinstance(annotations, pd.DataFrame):
            annotations = annotations.to_dict('records')
        
        ## map images with annotations
        self.images = []
        self.annotations = []
        self.ann_cache = []
        self.id_map = {} # {image_id_name: pos in self.images, ann_id_name: pos in self.annotations}
        for ann_idx, info in enumerate(annotations):
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
        
    def __len__(self):
        return len(self.images)
    
    def load_annotation(self, ann_idx):
        ann_info = self.annotations[ann_idx]
        ann_id = ann_info['ann_id']
        task_id = ann_info['task_id']

        anns = torch.load(os.path.join(self.root, ann_info['ann_path']))
        if task_id.startswith('det'):
            anns['masks'] = [Mask(_, anns['size'], ann_info['mask_mode']) for _ in anns['masks']]
        elif task_id.startswith('seg'):
            anns['masks'] = [Mask(_, anns['size'], ann_info['mask_mode']) for _ in anns['masks']]
        elif task_id.startswith('cl'):
            anns = anns
            # label = image_info['ann_path']
            # ann = {**ann, 'labels': label}
        else:
            raise ValueError(f"Task: {task_id} is not supported!")

        anns['image_id'] = ann_idx  # add annotation id
        
        return anns
    
    def load_image_and_target(self, idx):
        # load image
        image_info = self.images[idx]
        image = self.image_reader(os.path.join(self.root, image_info['image_path']))
        image = img_as(np.float32)(rgba2rgb(image))
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
        image, target, kwargs = self.load_image_and_target(idx)
        if self.processor is not None:
            image, target = self.processor(image, target, **kwargs)
        
        image = ToTensor()(image.copy()).type(torch.float32)
        target = target_to_tensors(target)
        
        return image, target


def display_image_and_target(image, target, meta_info, **kwargs):
    ## display by tasks:
    image_id = target['image_id']
    image_size = target['size']
    print(f"Image: id={image_id}, size={image_size}, stats={image_stats(image)}")
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.show()
    
    for task_id, anns in target['anns'].items():
        print(f"Task: {task_id}")
        if task_id.startswith('seg'):
            for ann in anns:
                x0, y0, x1, y1 = ann['roi']
                patch = image[int(y0):int(y1), int(x0):int(x1), :]
                print(f"patch_roi={(x0, y0, x1, y1)}, patch_stats={image_stats(patch)}, ann_size={ann['size']}")
                
                labels_color = meta_info.get(task_id, {}).get('labels_color', None)
                labels_text = meta_info.get(task_id, {}).get('labels_text', None)
                patch_resize = skimage.transform.resize(patch, ann['size'], order=3) # resize patch to annotation size
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                ax.imshow(patch_resize)
                overlay_segmentations(
                    ax, masks=ann['masks'], labels=None, 
                    labels_color=labels_color, labels_text=labels_text,
                )
                
                plt.show()

        elif task_id.startswith('det'):
            for ann in anns:
                x0, y0, x1, y1 = ann['roi']
                h_ann, w_ann = ann['size']
                patch = image[int(y0):int(y1), int(x0):int(x1), :]
                print(f"patch_roi={(x0, y0, x1, y1)}, patch_stats={image_stats(patch)}, ann_size={ann['size']}")
                
                labels_color = meta_info.get(task_id, {}).get('labels_color', None)
                labels_text = meta_info.get(task_id, {}).get('labels_text', None)
                patch_resize = skimage.transform.resize(patch, [h_ann, w_ann], order=3) # resize patch to annotation size
                
                boxes, labels, masks = ann['boxes'], ann['labels'], ann['masks']
                if boxes.max() <= 1.0:  # normalized box
                    boxes *= np.array([w_ann, h_ann, w_ann, h_ann])
                if isinstance(masks, torch.Tensor) and (masks.shape[-1] != w_ann or masks.shape[-2] != h_ann):
                    # masks is saved as K*M*M, paste back to h_ann, w_ann image
                    masks = paste_masks_in_image(masks[:, None, ...], boxes, (h_ann, w_ann), padding=1).squeeze(1)
                fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                ax.imshow(patch_resize)
                overlay_detections(
                    ax, bboxes=boxes, labels=labels, masks=masks,
                    labels_color=labels_color, labels_text=labels_text,
                    # show_bboxes=False, show_texts=False,
                )
                
                plt.show()

