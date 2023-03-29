import torch
import torchvision
import numpy as np
import pandas as pd
import pickle
import time
import albumentations as A
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from DIPModels.utils_g.utils_image import *

SEED = None


def collate_fn(batch):
    return tuple(zip(*batch))


## don't use if masks are truncated, as it will ruin correct bbox
def generate_objects_from_masks(masks, labels, image_size):
    objects = {'bboxes': [], 'masks': [], 'labels': [], 'areas': [], 'image_size': image_size}
    for mask, label in zip(masks, labels):
        area = np.sum(mask)
        objects['masks'].append(mask)
        objects['labels'].append(label)
        objects['areas'].append(area)
        
        if area > 0:
            pos = np.where(mask > 0)
            bbox = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
        else:
            bbox = [-1, -1, -1, -1]
        objects['bboxes'].append(bbox)
    
    return objects


def objects_to_tensor_targets(objects, image_id, image_size, bbox_mode='yxyx'):
    num_objs = len(objects['boxes'])
    assert len(objects['labels']) == num_objs
    
    h, w = image_size[0], image_size[1]
    boxes = torch.as_tensor(objects['boxes'], dtype=torch.float32)
    labels = torch.as_tensor(objects['labels'], dtype=torch.int64)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
    image_id = torch.tensor([image_id])
    image_size = torch.as_tensor([int(h), int(w)])
    
    res = {"boxes": boxes, "labels": labels, 
           "area": area, "image_id": image_id, 
           "size": image_size, "iscrowd": iscrowd}
    
    if 'masks' in objects:
        assert len(objects['masks']) == num_objs
        res['masks'] = torch.as_tensor(objects['masks'], dtype=torch.uint8)
        # res['area'] = torch.sum(res['masks'], dim=(1, 2))
    
    return res


def filter_tensor_targets(targets, keep_indices):
    res = {}
    if 'boxes' in targets:
        res['boxes'] = targets['boxes'][keep_indices]
    if 'labels' in targets:
        res['labels'] = targets['labels'][keep_indices]
    if 'masks' in targets:
        res['masks'] = targets['masks'][keep_indices]
    if 'area' in targets:
        res['area'] = targets['area'][keep_indices]
    
    return {**targets, **res}


def remove_invalid_objects(objects, image_size=None):
    if image_size is not None:
        h, w = image_size[0], image_size[1]
        objects["boxes"][:,0] = np.clip(objects["boxes"][:,0], 0, w)
        objects["boxes"][:,1] = np.clip(objects["boxes"][:,1], 0, h)
        objects["boxes"][:,2] = np.clip(objects["boxes"][:,2], 0, w)
        objects["boxes"][:,3] = np.clip(objects["boxes"][:,3], 0, h)
    
    boxes = objects['boxes']
    keep_idx = np.logical_and(boxes[:, 0] < boxes[:, 2], boxes[:, 1] < boxes[:, 3])
    return {k: v[keep_idx] for k, v in objects.items()}


def format_annotations(x, image_size, val_to_label, **kwargs):
    res = {}
    
    ## boxes
    res["boxes"] = np.array(x["boxes"])
    res["labels"] = np.array([val_to_label[_] for _ in x["labels"]])
    if "masks" in x:
        masks = []
        for mask in x["masks"]:
            if mask is None:
                mask = np.zeros(image_size)
            elif mask.shape[-1] == 2:  # a polygon
                # skimage x y is reverted
                mask = 1.0 * skimage.draw.polygon2mask(image_shape=image_size, polygon=mask[:,[1,0]])
            else:
                # add a resize if necessary
                mask = mask
            masks.append(mask)
        res["masks"] = np.array(masks)
    
    res = remove_invalid_objects(res, image_size)
    
    return res
        

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, ann_dir, meta_info, processor=None, **kwargs):
        self.processor = processor
        self.kwargs = kwargs
        
        self.image_ids = []
        self.images = []
        self.indices = {}
        
        meta_info = meta_info.drop_duplicates()  # has duplicate lines...
        for _, info in meta_info.iterrows():
            image_id = info['fovname']
            assert image_id not in self.indices, f"duplicated image_id: {image_id}"
            
            roi = [info['xmin'], info['ymin'], info['xmax'], info['ymax']]
            image_path = os.path.join(image_dir, f"{image_id}.png")
            
            if ann_dir is not None:
                df = pd.read_csv(os.path.join(ann_dir, f"{image_id}.csv"), index_col=0)
                boxes = 1.0 * df[['xmin', 'ymin', 'xmax', 'ymax']].values
                labels = df['group'].values
                masks = []
                for _, entry in df[['type', 'coords_x', 'coords_y']].iterrows():
                    if entry['type'] == "polyline":
                        coords_x = [float(_) for _ in entry['coords_x'].split(',')]
                        coords_y = [float(_) for _ in entry['coords_y'].split(',')]
                        # some bad annotations contradict to 'type'
                        if len(np.unique(coords_x)) < 4:
                            masks.append(None)
                        else:
                            masks.append(np.stack([coords_x, coords_y], axis=-1))
                    else:
                        masks.append(None)
                annotations = {"boxes": boxes, "labels": labels, "masks": masks}
            else:
                annotations = None
            new_image = {
                "image_id": image_id, 
                "data": (image_path, annotations), 
                "kwargs": {"roi": roi},
            }
            
            self.images.append(new_image)
            self.indices[image_id] = len(self.images) - 1
    
    def load_image_and_objects(self, idx):
        image_info = self.images[idx]
        ## Read in image and masks
        image_file, annotations = image_info['data']
        patch_roi = image_info["kwargs"]["roi"]
        image = self.kwargs['image_reader'](image_file)
        image = img_as('float')(rgba2rgb(image))
        
        kwargs = {**self.kwargs, **image_info['kwargs'], 'image_idx': idx,
                  'image_size': (image.shape[0], image.shape[1])}
        if annotations is not None:
            objects = format_annotations(annotations, **kwargs)
        else:
            objects = None
        
        return image, objects, kwargs
    
    def add_annotations(self, annotations):
        for image_id, bbox, label in annotations:
            if image_id in self.indices:
                idx = self.indices[image_id]
                _, ann = self.images[idx]['data']
                ann['boxes'] = np.append(ann['boxes'], [bbox], axis=0)
                ann['labels'] = np.append(ann['labels'], label)
                ann['masks'].append(None)
        
        return self
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image, objects, kwargs = self.load_image_and_objects(idx)
        
        ## processor will keep empty box after augmentation
        if self.processor is not None:
            image, objects = self.processor(image, objects, **kwargs)
        
        ## To torch Tensor
        h, w = image.shape[0], image.shape[1]
        image = ToTensor()(image.copy()).type(torch.float)
        targets = objects_to_tensor_targets(objects, idx, (h, w))
#         display_detection(image.permute(1, 2, 0).numpy(), targets['boxes'].numpy(), targets['labels'].numpy(), 
#                           kwargs['label_to_val'], masks=targets['masks'].numpy())
        
        ## filter out small objects: min_area=100, max_area=10000, h, w >= 10
        min_area, max_area = kwargs['min_area'], kwargs['max_area']
        min_h, min_w = kwargs['min_h'], kwargs['min_w']
        r_area = targets['area']/h/w
        r_h = (targets['boxes'][:, 3] - targets['boxes'][:, 1])/h
        r_w = (targets['boxes'][:, 2] - targets['boxes'][:, 0])/w
        
        keep_indices = (r_area >= min_area) & (r_area < max_area) & (r_h >= min_h) & (r_w >= min_w)
        targets = filter_tensor_targets(targets, keep_indices)
        
        ## For RCNN training only, add a bbox with label=0 to indicate background
        if not len(targets['boxes']):
            targets['boxes'] = torch.tensor([[0, 0, w-1, h-1]], dtype=torch.float32)
            targets['labels'] = torch.tensor([0], dtype=torch.int64)
            targets['masks'] = torch.ones((1, h, w), dtype=torch.uint8)
            targets['area'] = torch.tensor([h*w], dtype=torch.int64)
            targets['iscrowd'] = torch.zeros((1,), dtype=torch.int64)
        
        return image, targets
    
    def filter_images(self, fn):
        new_images, new_indices = [], {}
        for _ in self.images:
            if not fn(_):
                new_images.append(_)
                new_indices[_['image_id']] = len(new_images) - 1
        self.images = new_images
        self.indices = new_indices
        
        return self
    
    def display(self, indices=None, results=None, plot_raw=False, plot_image=True, plot_gt=True):
        if indices is None:
            indices = range(len(self))
        elif isinstance(indices, numbers.Number):
            indices = np.random.choice(len(self), indices)
        
        ## get display colors and texts
        labels_color = self.kwargs['labels_color']
        labels_text = {i+1: _ for i, _ in enumerate(self.kwargs['labels'])}
        
        for i in indices:
            ## raw image and info
            image_info = self.images[i]
            image_id, (image_file, masks_file) = image_info['image_id'], image_info['data']
            print("========================")
            print(f"image {i}, image_id: {image_id}")
            
            ## readed image and info
            image, targets = self[i]
            image = image.permute(1, 2, 0).numpy()
            image = image * self.kwargs['std'] + self.kwargs['mean']
            
            n_plots = plot_raw + plot_image + plot_gt + (results is not None)
            sub_fig_size = int(24/n_plots)
            fig, ax = plt.subplots(1, n_plots, figsize=(24, sub_fig_size))
            ax = ax.tolist()[::-1] if n_plots > 1 else [ax]
            
            if plot_raw:
                raw_image = self.kwargs['image_reader'](image_file)
                ax[-1].imshow(raw_image)
                ax.pop()
            
            if plot_image:
                ax[-1].imshow(image)
                ax.pop()
            
            if plot_gt:
                print("*********************")
                print(image_stats(image))
                print(image_stats(targets['boxes']))
                print(image_stats(targets['labels']))
                print(image_stats(targets['masks']))
                print("*********************")
                t_boxes = targets['boxes'].numpy()
                t_labels = targets['labels'].numpy()
                t_masks = targets['masks'].numpy() if 'masks' in targets else None
                ax[-1].imshow(image)
                overlay_detections(ax[-1], bboxes=t_boxes, labels=t_labels, masks=t_masks, scores=None,
                                   labels_color=labels_color, labels_text=labels_text,
                                   show_bboxes=True, show_texts=False, show_masks=True, show_scores=False,
                                  )
                ax.pop()
            
            if results is not None:
                outputs = results[i]
                o_boxes = outputs['boxes'].numpy()
                o_labels = outputs['labels'].numpy()
                o_masks = outputs['masks'].numpy() if 'masks' in outputs else None
                ax[-1].imshow(image)
                overlay_detections(ax[-1], bboxes=o_boxes, labels=o_labels, masks=o_masks, scores=None,
                                   labels_color=labels_color, labels_text=labels_text,
                                   show_bboxes=True, show_texts=False, show_masks=True, show_scores=False,
                                  )
                ax.pop()
                
            plt.show()


class DanyunTestDataset(TorchDataset):
    def __init__(self, image_dir, ann_dir, processor=None, **kwargs):
        self.processor = processor
        self.kwargs = kwargs
        
        self.image_ids = []
        self.images = []
        self.indices = {}
        
        for image_file_name in os.listdir(image_dir):
            if image_file_name.endswith('.png'):
                image_id = image_file_name.split('.png')[0]
                image_path = os.path.join(image_dir, "{}.png".format(image_id))
                ann_path = os.path.join(ann_dir, "{}.pkl".format(image_id))
                
                new_image = {
                    "image_id": image_id, 
                    "data": (image_path, ann_path), 
                    "kwargs": {},
                }
            
                self.images.append(new_image)
                self.indices[image_id] = len(self.images) - 1
                
    def load_image_and_objects(self, idx):
        image_info = self.images[idx]
        
        ## Read in image
        image_path, ann_path = image_info['data']
        image = self.kwargs['image_reader'](image_path)
        image = img_as('float')(rgba2rgb(image))
        
        ## load and process gt      
        gt = pickle.load(open(ann_path, 'rb'))
        masks = np.moveaxis(gt['masks'], -1, 0)
        boxes = [get_mask_bbox(_) for _ in masks]
        masks = [_ if tp == 'polyline' else np.zeros_like(_)
                 for _, tp in zip(masks, gt['lb_typs'])]
        annotations = {'boxes': np.array(boxes), 'labels': gt['class_nms'], 'masks': masks}
        
        kwargs = {**self.kwargs, **image_info['kwargs'], 'image_idx': idx,
                  'image_size': (image.shape[0], image.shape[1])}
        if annotations is not None:
            objects = format_annotations(annotations, **kwargs)
        else:
            objects = None
        
        return image, objects, kwargs


def train_processor(image, objects, **kwargs):
    mean = kwargs['mean']
    std = kwargs['std']
    output_size = kwargs['output_size']
    color_aug = kwargs['color_aug']
    
    ## Color Augmentation and Normalization
    if color_aug == 'dodge':
        image, = ColorDodge(global_mean=0.01, channel_mean=0.01, channel_sigma=0.2, p=0.5)([image])
    elif color_aug == 'jitter':
        image, = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.15, 0.1), p=0.5)([image])
    image, = Normalize(mean=mean, std=std)([image])
    
    ## shape augmentation
    has_mask = 'masks' in objects
    tf = A.Compose([
        # A.LongestMaxSize(max_size=400, p=1),
        A.PadIfNeeded(output_size[0], output_size[1], border_mode=0, p=1),
        A.RandomCrop(output_size[0], output_size[1], p=1),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
                      mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=0.7),
        # rotate will ruin bounding box, so don't do rotation, use mask for tight bbox
        A.Affine(scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)}, translate_percent=(-0.1, 0.1), rotate=None, shear=None, p=1.0),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=3, p=1.0),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['bbox_ids']))
    
    # albumentation failed to remove masks when remove bboxes, use bbox_ids to hack...
    params = {'masks': [_ for _ in objects['masks']]} if has_mask else {}
    transformed = tf(image=image, bboxes=objects['boxes'], bbox_ids=np.arange(len(objects['boxes'])), **params)
    
    image = np.array(transformed['image'])
    keep = np.array(transformed['bbox_ids'])
    objects = {
        'boxes': np.array(transformed['bboxes']),
        'labels': objects['labels'][keep],
    }
    if has_mask:
        objects['masks'] = np.array(transformed['masks'])[keep]

    return image, objects


def val_processor(image, objects, **kwargs):
    mean = kwargs['mean']
    std = kwargs['std']
    output_size = kwargs['output_size']
    color_aug = kwargs['color_aug']
    
    ## Color Augmentation and Normalization
    if color_aug == 'dodge':
        image, = ColorDodge(global_mean=0.01, channel_mean=0.01, channel_sigma=0.2, p=0.5)([image])
    elif color_aug == 'jitter':
        image, = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.15, 0.1), p=0.5)([image])
    image, = Normalize(mean=mean, std=std)([image])

    has_mask = 'masks' in objects
    tf = A.Compose([
        A.PadIfNeeded(output_size[0], output_size[1], border_mode=0, p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['bbox_ids']))
    
    # albumentation failed to remove masks when remove bboxes, use bbox_ids to hack...
    params = {'masks': [_ for _ in objects['masks']]} if has_mask else {}
    transformed = tf(image=image, bboxes=objects['boxes'], bbox_ids=np.arange(len(objects['boxes'])), **params)
    
    image = np.array(transformed['image'])
    keep = np.array(transformed['bbox_ids'])
    objects = {
        'boxes': np.array(transformed['bboxes']),
        'labels': objects['labels'][keep],
    }
    if has_mask:
        objects['masks'] = np.array(transformed['masks'])[keep]

    return image, objects


def train_processor_ms(image, objects, **kwargs):
    image, objects = train_processor(image, objects, **kwargs)
    
    ## shrink to 20x, order = 0 will severely trunk/shrink mask
    factor = np.random.uniform(0.5, 1.0)
    image = skimage.transform.rescale(image, factor, preserve_range=True, anti_aliasing=True, multichannel=True, order=3)
    objects['masks'] = [skimage.transform.rescale(_, factor, preserve_range=True, order=1) > 0.5 for _ in objects['masks']]
    objects['bboxes'] = np.array(objects['bboxes']) * factor

    return image, objects


def train_processor_mask(image, objects, **kwargs):
    mean = kwargs['mean']
    std = kwargs['std']
    output_size = kwargs['output_size']
    color_aug = kwargs['color_aug']
    
    has_mask = 'masks' in objects
    tf = A.Compose([
        A.PadIfNeeded(output_size[0], output_size[1], border_mode=0, p=1),
        A.RandomCrop(output_size[0], output_size[1], p=1),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['bbox_ids']))
    
    # albumentation failed to remove masks when remove bboxes, use bbox_ids to hack...
    params = {'masks': [_ for _ in objects['masks']]} if has_mask else {}
    transformed = tf(image=image, bboxes=objects['boxes'], bbox_ids=np.arange(len(objects['boxes'])), **params)
    
    image = np.array(transformed['image'])
    keep = np.array(transformed['bbox_ids'])
    objects = {
        'boxes': np.array(transformed['bboxes']),
        'labels': objects['labels'][keep],
    }
    if has_mask:
        objects['masks'] = np.array(transformed['masks'])[keep]

    return image, objects



####################################################################################
####################################################################################
def get_mask_ious(y_true, y_pred):
    # a_and_b = torch.einsum('ihw,jhw->ij', y_true, y_pred)
    n_true, n_pred = len(y_true), len(y_pred)
    y_true = y_true.flatten(1).unsqueeze(1).expand(-1, n_pred, -1)
    y_pred = y_pred.flatten(1).unsqueeze(0).expand(n_true, -1, -1)
    
    inter = (y_true * y_pred).sum(dim=-1)
    union = (y_true + y_pred).sum(dim=-1) - inter + 1e-8
    
    return inter/union


def evaluate_detection(target, output, classes, iou_threshold=0.5, iou_type='boxes'):
    # import networkx as nx
    
    # get ious: n_target * n_preds
    if iou_type=='masks' and ('masks' in output and 'masks' in target): # use mask iou
        ious = get_mask_ious(target['masks'], output['masks'])
    else: # use box iou
        ious = torchvision.ops.box_iou(target['boxes'], output['boxes'])
    n_true, n_pred = ious.shape[0], ious.shape[1]
    
    ## Cover gt with prediction, this will only focus on recall and ignore precision
    true_label = target['labels']
    pred_label = output['labels']
    if n_true > 0 and n_pred > 0:
        # matcher = Matcher(high_threshold=iou_threshold, low_threshold=0.0)
        # matched_idxs = matcher(ious) # -2: between low and high, -1: below low, >0: matched_ids, 0: bg,
        matched_ious, matched_idxs = ious.max(1)
        pred_label = pred_label[matched_idxs]
        pred_label[matched_ious < iou_threshold] = -1
    else:
        pred_label = -torch.ones_like(true_label)
        matched_ious = torch.zeros_like(true_label) * 1.0
    res = {'y_true': true_label, 'y_pred': pred_label, 'ious': matched_ious}
    
    ## per_class summary with best bipartite match, includes precision and recall
    stats_per_class = {}
    for c in classes:
        t_idx, o_idx = target['labels'] == c, output['labels'] == c
        n1, n2 = t_idx.sum().item(), o_idx.sum().item()
        matched, m_iou = [], 0.
        if n1 > 0 and n2 > 0:
            ious_c = ious[t_idx][:,o_idx]
            matched_ious, matched_idxs = ious_c.max(1)
            matched = matched_ious[matched_ious >= iou_threshold]
            m_iou = matched.mean().item() if len(matched) else 0.0
        
        stats_per_class[c] = [len(matched), n1, n2, m_iou]

#             edges = [(x+1, -(y+1), {"weight": -ious_c[x][y]})
#                      for x in range(n1) for y in range(n2) if ious_c[x][y] >= iou_threshold]
            
#             if edges:
#                 G = nx.Graph()
#                 G.add_nodes_from([_+1 for _ in range(n1)], bipartite=0)
#                 G.add_nodes_from([-(_+1) for _ in range(n2)], bipartite=1)
#                 G.add_edges_from(edges)
                
#                 for nodes in nx.connected_components(G):
#                     if len(nodes) >= 2:
#                         flow = nx.bipartite.minimum_weight_full_matching(G.subgraph(nodes))
#                         matched.extend([(k-1, -v-1) for k, v in flow.items() if k > 0])
                
#                 m_iou = np.mean([ious_c[k][v] for k, v in matched])
#         stats_per_class[c] = [len(matched), n1, n2, m_iou]
    
    return res, stats_per_class


def weighted_average_pr(dataset, results, classes, iou_threshold=0., class_weights="area"):
    stats_logger = defaultdict(list)
    labels_all, areas_all = [], []
    for (image, target), output in zip(dataset, results):
        res, stats_per_class = evaluate_detection(target, output, classes, iou_threshold=iou_threshold)
        labels_all.append(target['labels'])
        areas_all.append(target['area'])
        for k, v in stats_per_class.items():
            stats_logger[k].append(v)

    ## per class precision and recall
    if class_weights is None or class_weights == "none":
        class_weights = {k: 1./len(stats_logger) for k in stats_logger}
    else:
        if class_weights == "area":
            per_class = pd.DataFrame.from_dict({
                'labels': torch.cat(labels_all).numpy(), 
                'areas': torch.cat(areas_all).numpy(),
            }).groupby('labels')['areas'].sum()
            class_weights = (per_class/per_class.sum()).to_dict()
        elif class_weights == "n_obj":
            per_class = pd.DataFrame.from_dict({
                'labels': torch.cat(labels_all).numpy(), 
            }).groupby('labels')['labels'].size()
            class_weights = (per_class/per_class.sum()).to_dict()
    
    ## normalize weights
    total = sum(class_weights.values())
    class_weights = {k: 1.0*v/total for k, v in class_weights.items()}
    
    stats = {}
    for k, v in stats_logger.items():
        v = np.array(v)
        precision = v[:,0].sum()/(v[:,2].sum() + 1e-8)
        recall = v[:,0].sum()/(v[:,1].sum() + 1e-8)
        m_iou = (v[:,3] * v[:,0]).sum()/(v[:,0].sum() + 1e-8)
        stats[k] = np.array([precision, recall, m_iou])
    
    class_weights = {k: v for k, v in class_weights.items() if k >= 0}
    ave_stats = sum([stats[k] * class_weights[k]for k in class_weights])
    f = 2 * ave_stats[0] * ave_stats[1]/(ave_stats[0] + ave_stats[1])
    
    return {'precision': ave_stats[0], 'recall': ave_stats[1], 'w_miou': ave_stats[2], 'f_score': f}


def weighted_accuracy(y_pred, y_true, weight=None):
    assert len(y_pred) == len(y_true)
    if len(y_pred) == 0:
        return 0.
    
    if weight is not None:
        w = torch.tensor(weight, device=y_pred.device)[y_true]
        c = (w * (y_true == y_pred)).sum() * 1.0
        t = w.sum()
    else:
        c = (y_true == y_pred).sum() * 1.0
        t = y_pred.shape[0]
    
    return c/t


def get_shidan_stats(y_true, y_pred, ious, num_classes=6):
    counts = [(y_true == _).sum().item() for _ in range(1, num_classes+1)]
    class_weights = [1./_ if _ > 0 else 0. for _ in counts]
    print(class_weights)
    
    matched_ids = y_pred != 0
    mean_iou = ious[matched_ids].mean().item()
    coverage = matched_ids.sum().item()/len(y_true)
    # set negative (-100) to 0 and give 0 weight
    accuracy = weighted_accuracy(y_true[matched_ids].clip(0), y_pred[matched_ids], [0.] + class_weights)
    
    # cm = confusion_matrix(classes_matched_gt, classes_matched)
    # Coverage, accuracy, mIOU
    
#     # tumor, stroma, lympho coverage
#     core_nuclei_idx = y_true_core != 4
#     coverage_core = matched_ids[core_nuclei_idx].sum()/core_nuclei_idx.sum()
#     accuracy_core = weighted_accuracy(y_true_core[matched_ids], y_pred_core[matched_ids], [0.] + class_weights + [1.])
    
    return coverage, accuracy, mean_iou
####################################################################################
####################################################################################



####################################################################################
####################################################################################
def reduce_confusion_matrix(cm, labels):
    if not isinstance(labels, dict):
        label_x = label_y = labels
    else:
        label_x, label_y = labels['x'], labels['y']
    res = np.zeros([len(label_x)+1, len(label_y)+1])
    
    res[:-1, :-1] = cm.loc[label_x, label_y].values
    res[:-1, -1] = cm.drop(label_y, axis=1).loc[label_x,].values.sum(1)
    res[-1, :-1] = cm.drop(label_x, axis=0)[label_y].sum(0)
    res[-1, -1] = cm.drop(label_y, axis=1).drop(label_x, axis=0).values.sum()

    return pd.DataFrame(res, index=label_x + ["others"], columns=label_y + ["others"])


def summarize_confusion_matrix(cm, labels, core_labels=['tumor', 'stromal', 'sTILs']):
    assert len(cm) == len(labels) and len(cm[0]) == len(labels)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    coverage = 1 - cm['missing'].values.sum()/cm.values.sum()

    # In gt, unlabeled will be merged into "others" and thus be ignored.
    # In pred, unlabeled is merged into "others", we treat it as "others" nuclei
    cm_core = reduce_confusion_matrix(cm, core_labels + ['missing'])
    cm_core = cm_core.drop("missing", axis=0).drop("others", axis=0)
    K = len(np.diag(cm_core))
    accuracy = np.diag(cm_core.values).sum()/cm_core.values.sum()  # overall accuracy (consider missing)
    accuracy_c = np.diag(cm_core.values).sum()/cm_core.values[:K, :K].sum()  # HD-staining accuracy (ignore missing)

    precision = np.diag(cm_core.values)/cm_core.values.sum(0)[:K]
    recall = np.diag(cm_core.values)/cm_core.values.sum(1)[:K]
    f = 2 * precision * recall/(precision + recall)

    return {'coverage': coverage, 'accuracy_c': accuracy_c, 'accuracy': accuracy, 
            'cm': cm, 'cm_core': cm_core,
            **{('precision', n): _ for n, _ in zip(core_labels, precision)}, 
            **{('recall', n): _ for n, _ in zip(core_labels, recall)}, 
            **{('f1', n): _ for n, _ in zip(core_labels, f)}, }


def summarize_precision_recall(stats_list, labels_text):
    stat_sum = defaultdict(list)
    for stat in stats_list:
        for k, v in stat.items():
            stat_sum[k].append(v)
    
    res = {}
    for k, v in stat_sum.items():
        tmp = np.array(v)
        n_matched, n_true, n_pred, m_iou = tmp[:,0].sum(), tmp[:,1].sum(), tmp[:,2].sum(), tmp[:,3].mean()
        precision = n_matched/n_pred if n_pred > 0 else np.nan
        recall = n_matched/n_true if n_true > 0 else np.nan
        f = 2 * precision * recall/(precision + recall)
        res[labels_text[k]] = {'precision': precision, 'recall': recall, 'f1': f, 'miou': m_iou}
        
    return res


def summarize_mcc(y_true, y_pred, core_labels=['tumor', 'stromal', 'sTILs']):
    """ addon for NuCLS paper comparison. """
    res = {}
    indexing = [_ in core_labels for _ in y_true]
    y_true_core = [val for tag, val in zip(indexing, y_true) if tag]
    y_pred_core = [val for tag, val in zip(indexing, y_pred) if tag]
    res['mcc'] = matthews_corrcoef(y_true_core, y_pred_core)  

    for nuclei_type in core_labels:
        y_true_binary = [_ == nuclei_type for _ in y_true_core]
        y_pred_binary = [_ == nuclei_type for _ in y_pred_core]
        res[('mcc', nuclei_type)] = matthews_corrcoef(y_true_binary, y_pred_binary)
    
    return res


def evaluate_results_new(dataset, results, iou_threshold=0.5, 
                         core_labels=['tumor', 'stromal', 'sTILs'], 
                         label_converter={}, meta_info=None, 
                         display_results=False):
    assert len(dataset) == len(results), f"dataset and results are not compatible."
    if meta_info is None:
        labels_color = dataset.kwargs['labels_color']
        labels_text = dataset.kwargs['labels_text']
    else:
        labels_color = meta_info['labels_color']
        labels_text = meta_info['labels_text']
    # add a label convert to merge labels if needed
    label_converter_true = label_converter.get('true', lambda x: x)
    label_converter_pred = label_converter.get('pred', lambda x: x)

    # For confusion matrix: add a class for missing detection (label=-1)
    cm_labels_text = {**labels_text, -1: 'missing'}
    cm_labels = list(cm_labels_text.values())

    cm_list, stats_list, y_trues, y_preds, y_ious = [], [], [], [], []
    for idx in range(len(dataset)):
        ## read raw image and raw masks
        image_info = dataset.images[idx]
        image_path, objects = image_info['data']
        image_id = image_info['image_id']
        raw_image = rgba2rgb(skimage.io.imread(image_path))

        image, target = dataset[idx]
        image = np.moveaxis(image.numpy(), 0, -1)
        output = results[idx]
        target['labels'] = label_converter_true(target['labels'])
        output['labels'] = label_converter_pred(output['labels'])

#         ## Don't run extra nms, directly apply to model, iou_threshold means something else
#         if iou_threshold_nms:
#             keep = torchvision.ops.nms(output['boxes'], output['scores'], iou_threshold=iou_threshold_nms)
#             output['boxes'] = output['boxes'][keep]
#             output['labels'] = output['labels'][keep]
#             output['scores'] = output['scores'][keep]
#             if 'masks' in output:
#                 output['masks'] = output['masks'][keep]
        
        ## run evaluation
        res, stats_per_class = evaluate_detection(target, output, classes=list(labels_text), iou_threshold=iou_threshold)
        y_true = [cm_labels_text[_] for _ in res['y_true'].numpy()]
        y_pred = [cm_labels_text[_] for _ in res['y_pred'].numpy()]
        y_ious.extend([_ for _ in res['ious'].numpy()])
        y_trues.extend(y_true)
        y_preds.extend(y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        cm_list.append(cm)
        stats_list.append(stats_per_class)
        
        ## display
        if display_results:
            print("===============================")
            print(image_id)

            cm_stats = summarize_confusion_matrix(cm, cm_labels, core_labels=core_labels)
            print(f"coverage: {cm_stats['coverage']}, accuracy: {cm_stats['accuracy']}")
            # plot_confusion_matrix(cm.values, cm.columns)
            display(cm_stats['cm'])
            display(cm_stats['cm_core'])
            fig, axes = plt.subplots(1, 3, figsize=(24, 24))
            ax = axes.ravel()
            ax[0].imshow(raw_image)
            ax[1].imshow(image)
            overlay_detections(
                ax[1], bboxes=target['boxes'], labels=target['labels'],
                masks=target.get('masks', None), scores=None,  # target['masks'].numpy()
                labels_color=labels_color, labels_text=labels_text,
                show_bboxes=True, show_texts=False, show_masks=True, show_scores=False,
            )
            ax[2].imshow(image)
            overlay_detections(
                ax[2], bboxes=output['boxes'], labels=output['labels'],
                masks=output.get('masks', None), scores=output['scores'],
                labels_color=labels_color, labels_text=labels_text,
                show_bboxes=True, show_texts=False, show_masks=True, show_scores=False,
            )
            
            if isinstance(display_results, str):
                figure_file_name = os.path.join(display_results, "figures", f"{image_id}.png")
                plt.savefig(figure_file_name)
                plt.close()
            else:
                plt.show()
    
    sum_stats_1 = summarize_confusion_matrix(sum(cm_list), cm_labels, core_labels=core_labels)
    sum_stats_1['preds'] = {'y_true': y_trues, 'y_pred': y_preds}
    sum_stats_1['miou'] = np.array(y_ious).mean()
    sum_stats_1 = {**sum_stats_1, **summarize_mcc(y_trues, y_preds, core_labels=core_labels)}

    sum_stats_2 = summarize_precision_recall(stats_list, labels_text)
    
    return cm_list, stats_list, sum_stats_1, sum_stats_2


####################################################################################
####################################################################################

