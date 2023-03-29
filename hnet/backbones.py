import timm
import numbers
import torch
import torchvision
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from collections import OrderedDict
from torchvision.ops.feature_pyramid_network import *
from yolov5.yolo import Model as yolo_backbone

# FeaturePyramidNetwork, LastLevelP6P7(256, 256), LastLevelMaxPool

def to_device(x, device):
    """ Move objects to device.
        1). if x.to(device) is valid, directly call it.
        2). if x is a function, ignore it
        3). if x is a dict, list, tuple of objects.
            recursively send elements to device.
        Function makes a copy of all non-gpu objects of x. 
        It will skip objects already stored on gpu. 
    """
    try:
        return x.to(device)
    except:
        if not callable(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    x[k] = to_device(v, device)
            else:
                x = type(x)([to_device(v, device) for v in x])
    return x

class ROIExtractor(nn.Module):
    def anchor_generator(self, image_shape, roi_size, overlap=0):
        h, w = image_shape[0], image_shape[1]
        roi_h, roi_w = roi_size[0], roi_size[1]
        x0 = torch.arange(0, w, roi_w-overlap, dtype=torch.float32)
        y0 = torch.arange(0, h, roi_h-overlap, dtype=torch.float32)
        y0, x0 = torch.meshgrid(y0, x0)
        x0, y0 = x0.reshape(-1), y0.reshape(-1)
        x1, y1 = x0 + roi_w, y0 + roi_h
        
        boxes = torch.stack((x0, y0, x1, y1), dim=1)
        boxes = torch.ops.boxes.clip_boxes_to_image(boxes, (h, w))
        
        return boxes
    
    def forward(self, x, image_shape, roi_size, feature_maps=None, targets=None):
        """
            Extract ROIs from FPN.
            Args:
                x (OrderedDict): {'0': [N, c, h, w]}feature maps for each feature level.
                image_shape (Tuple):  the input image shape (img_h_after_transform, img_w_after_transform).
                roi_size (Tuple): specific to task_id, (roi_h_after_transform, roi_w_after_transform).
                feature_maps: (OrderedDict[str, int]): feature_map names -> indices mapping.
                targets (Optional, List[List[Dict]]): list of annotations. 
                    If None, extractor will split feature map into (overlapped) ROI with roi_size.
            Returns:
                results (OrderedDict[Tensor]): feature maps for ROI on FPN layers.
                    They are ordered from highest resolution first.
        """
        feature_names, features = list(zip(*x.items()))
        num_images = features[0].shape[0]
        device = features[0].device
        device2 = torch.device('cuda:1')
        
        if targets is not None:
            assert len(targets) == num_images
        else:
            targets = [None] * num_images
        
        boxes, task_targets = [], []
        for t in targets:
            if t is None:
                rois = self.anchor_generator(image_shape, roi_size, overlap=0).to(device)
                anns = [{'roi': _} for _ in rois]
            elif len(t):
                rois = torch.stack([ann['roi'] for ann in t]).to(device)
                anns = [ann for ann in t]
            else:
                rois = torch.zeros((0, 4)).to(device) # create an empty tensor
                anns = []
            boxes.append(rois)
            task_targets.extend(anns)

        ## calculate scale_factors for each feature map:
        results = OrderedDict({})
        for k, f in zip(feature_names, features):
            scale_h, scale_w = f.shape[2]/image_shape[0], f.shape[3]/image_shape[1]
            output_size = (int(roi_size[0] * scale_h), int(roi_size[1] * scale_w))
            roi_multiplier = f.new([scale_w, scale_h, scale_w, scale_h])
            rois = [_ * roi_multiplier for _ in boxes]
            results[k] = torchvision.ops.roi_align(f, rois, output_size, aligned=True)#.to(device2)
        
        task_features = OrderedDict({k: results[v] for k, v in feature_maps.items()})
        # task_targets = to_device(task_targets, device2)
        
        return task_features, task_targets


class BasicFeaturePyramidNetwork(FeaturePyramidNetwork):
    def forward(self, x):
        x = OrderedDict((str(k), v) for k, v in enumerate(x))
        return super().forward(x)


# backbone_type='tv_resnet50'
class DynamicFeaturePyramidNetwork(FeaturePyramidNetwork):    
    def get_result_from_inner_blocks(self, x, idx, boxes, roi_size):
        out = torchvision.ops.roi_align(x, boxes, roi_size, aligned=True)
#         out = [F.interpolate(_[None, :, int(x0):int(x1), int(y0):int(y1)], size=roi_size, mode="nearest")[0]
#                for _, rois in zip(x, boxes) for y0, x0, y1, x1 in rois]
#         out = torch.stack(out)
        
        return super().get_result_from_inner_blocks(out, idx)
    
    def forward(self, x, image_shape, roi_size, feature_maps=None, targets=None):
        """
            Computes the FPN for a set of feature maps.
            Args:
                x (Tuple, List): feature maps for each feature level.
                image_shape (Tuple):  the input image shape (img_h_after_transform, img_w_after_transform).
                roi_size (Tuple): specific to task_id, (roi_h_after_transform, roi_w_after_transform).
                feature_maps: (OrderedDict[str, int]): feature_map names -> indices mapping.
                targets (Optional, List[List[Dict]]): list of annotations.
            Returns:
                results (OrderedDict[Tensor]): feature maps after FPN layers.
                    They are ordered from highest resolution first.
        """
        ## calculate scale_factors for each feature map:
        roi_sizes, roi_boxes = [], []
        for f in x:
            scale_h, scale_w = f.shape[2]/image_shape[0], f.shape[3]/image_shape[1]
            roi_sizes.append((int(roi_size[0] * scale_h), int(roi_size[1] * scale_w)))
            ## calculate roi on each feature map
            if targets is not None:
                boxes = [torch.stack([ann['roi'] for ann in t]) * f.new([scale_w, scale_h, scale_w, scale_h]) 
                         for t in targets]
            else:
                boxes = None
            roi_boxes.append(boxes)
        
        ## Last feature layer
        results = []
        last_inner = self.get_result_from_inner_blocks(x[-1], -1, roi_boxes[-1], roi_sizes[-1])
        results.append(self.get_result_from_layer_blocks(last_inner, -1))
        
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx, roi_boxes[idx], roi_sizes[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))            
        
        task_features = OrderedDict({k: results[v] for k, v in feature_maps.items()})
        task_targets = [ann for t in targets for ann in t]
        
        return task_features, task_targets


## EfficientPS: Efficient Panoptic Segmentation
class DoubleFeaturePyramidNetwork(torch.nn.Module):
    pass
