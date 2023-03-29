import math
import yaml
import torch
import torchvision
import torch.nn.functional as F

from typing import List, Tuple, Dict, Optional

from .torch_layers import *
from .transform import *
from .backbones import *
from .detection.mask_rcnn import MaskRCNN
from .segmentation.panoptic_seg import PanopticSeg
# from .fcos_utils import *
# from .detr_utils import *


class ClassificationModule(torch.nn.Module):
    pass


class GraphModule(torch.nn.Module):
    pass


class ConstrainModule(torch.nn.Module):
    def __init__(self, config):
        super(ConstrainModule, self).__init__()
        self.config = config
        self.graph = self.build_bipartite_graph(config['graph'])
    
    def build_bipartite_graph(self, x):
        if isinstance(x, dict):
            assert 'edges' in x and 'values' in x
            edges = torch.tensor(x['edges'], dtype=torch.int64)
            values = torch.tensor(x['values'])
        else:
            x = torch.tensor(x).to_sparse()
            edges = x.indices().t()
            values = x.values()
        
        return {'edges': edges, 'values': values}
    
    def match_rois(self, bbox, rois):
        res = []
        x0, y0, x1, y1 = bbox
        for idx, roi in enumerate(rois):
            if x0 >= roi[0] and y0 >= roi[1] and x1 <= roi[2] and y1 <= roi[3]:
                res.append(idx) # bbox is in rois[idx]

        return res

    def compute_probs(self, det_class_probs, det_mask_probs, seg_mask_probs):
        """
            det_class_probs: (N_obj, det_class), 
            det_mask_probs: (N_obj, det_class, h, w), sigmoid probability
            seg_mask_probs: (N_obj, det_class, seg_class, h, w), softmax probability
        """
        probs = []
        for (i, j), v in zip(self.graph['edges'], self.graph['values']):
            area = (seg_mask_probs[:, j, i, :, :] * det_mask_probs[:, j, :, :]).sum()
            p_area = area/det_mask_probs[:, j, :, :].sum()
            p_i_j = p_area * det_class_probs[:, j]
            probs.append(p_i_j)

        return sum(probs)

    def forward(self, outputs_0, outputs_1, targets_0, targets_1):
        """ output_1: [output, det_class]
            det_mask_logits: (N_obj, det_class, h, w)
            seg_mask_logits: (N_obj, seg_class, h, w), seg_mask_logits roi_align to proposals
        """
        probs = []

        for seg_preds, det_preds, seg_gts, det_gts in zip(outputs_0, outputs_1, targets_0, targets_1):
            seg_rois = [_['roi'] for _ in seg_gts]
            # project bboxes in det_res into seg_res
            for det_pred, det_gt in zip(det_preds, det_gts):
                obj_masks = det_pred['masks']
                obj_probs = det_pred['logits'].softmax(1)
                n_classes_seg = self.config['num_classes_root'] # boxes.shape[1]
                n_classes_det = self.config['num_classes_node'] #masks.shape[1]
                m0, m1 = obj_masks.shape[-2], obj_masks.shape[-1]

                # match det roi to seg roi
                seg_idx = self.match_rois(det_gt['roi'], seg_rois)[0]
                boxes_abs = project_roi_boxes_to_image(
                    det_pred['boxes'], det_gt['roi'], 
                    target_size=self.config['target_size_node'], 
                )
                boxes = project_image_boxes_to_roi(
                    boxes_abs, seg_gts[seg_idx]['roi'], 
                    target_size=self.config['target_size_root'], 
                )

                # if use ground truth mask
                masks = seg_gts[seg_idx]['masks'][None]
                # if use predict
                # masks = seg_preds[seg_idx][None]
                obj_segs = torchvision.ops.roi_align(masks, [boxes.view(-1, 4)], (m0, m1), aligned=True)
                obj_segs = obj_segs.view(-1, n_classes_det, n_classes_seg, m0, m1)

                probs.append(self.compute_probs(obj_probs, obj_masks, obj_segs))

        probs = torch.cat(probs)
        losses = F.binary_cross_entropy(probs, torch.ones_like(probs))
        # losses = -torch.mean(torch.log(probs))

        return {"bce_loss": losses}


class HNet(torch.nn.Module):
    def __init__(self, configs):
        super(HNet, self).__init__()
        self.configs = configs
        self.transform = GeneralizedTransform(**configs['transform'])
        # self.hierarchy_loss = RuleBranch(configs['constrain'])
        
        ## Build backbones
        backbone_cfg = self.configs['backbone']
        if backbone_cfg['type'] == 'swin':
            self.backbone = SwinTransformer(**backbone_cfg['configs'])
            feature_channels = [96, 192, 384, 768]
            feature_maps = OrderedDict({'0': 0, '1': 1, '2': 2, '3': 3})
        else:
            self.backbone = timm.create_model(backbone_cfg['type'], pretrained=True, 
                                              features_only=True, **backbone_cfg['configs'])
            feature_channels = [_['num_chs'] for _ in self.backbone.feature_info][1:] # ignore 1st feature map
            feature_maps = OrderedDict({'0': 0, '1': 1, '2': 2, '3': 3})
        
        ## Build neck layers (FPN, etc)
        neck_cfg = self.configs['neck']
        if 'in_channels_list' not in neck_cfg['configs'] or neck_cfg['configs']['in_channels_list'] is None:
            neck_cfg['configs']['in_channels_list'] = feature_channels
        
        if neck_cfg['type'] == 'fpn':
            self.fpn = DynamicFeaturePyramidNetwork(
                neck_cfg['configs']['in_channels_list'],
                neck_cfg['configs']['out_channels'], 
            )
        else:
            raise ValueError("{} is not implemented yet!".format(neck_cfg['type']))
        
        ## Build forward task headers
        self.headers = torch.nn.ModuleDict({})
        self.roi_sizes = {}
        self.feature_maps = {}
        
        for task_id, cfg in self.configs['headers'].items():
            if 'in_channels' not in cfg['configs'] or cfg['configs']['in_channels'] is None:
                cfg['configs']['in_channels'] = neck_cfg['configs']['out_channels']
            if 'feature_maps' not in cfg['configs'] or cfg['configs']['feature_maps'] is None:
                cfg['configs']['feature_maps'] = feature_maps
            
            self.roi_sizes[task_id] = cfg['configs']['roi_size']
            self.feature_maps[task_id] = cfg['configs']['feature_maps']
            
            if cfg['type'] == 'PanopticSeg': # segmentation header
                self.headers[task_id] = PanopticSeg(cfg['configs'])
            elif cfg['type'] == 'MaskRCNN': # detection header
                self.headers[task_id] = MaskRCNN(cfg['configs'])
            elif cfg['type'] == 'CL': # classification header
                self.headers[task_id] = ClassificationModule(cfg['configs'])
            else:
                raise NotImplementedError()
            
            deep_update(cfg['configs'], self.headers[task_id].config)

        ## Build hierachical constrains cross headers
        self.constrains = torch.nn.ModuleDict({})
        if 'constrains' in configs and configs['constrains'] is not None:
            for edge, cfg in configs['constrains'].items():
                root, node = edge.split('_')
                cfg['target_size_root'] = self.headers[root].config['target_size']
                cfg['num_classes_root'] = self.headers[root].config['num_classes']
                cfg['target_size_node'] = self.headers[node].config['target_size']
                cfg['num_classes_node'] = self.headers[node].config['num_classes']
                self.constrains[edge] = ConstrainModule(cfg)
        
        self.debug_buffer = 0.
    
    @torch.jit.unused
    def eager_outputs(self, losses, outputs):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return outputs
    
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        num_images = len(images)
        if targets is not None:
            assert len(targets) == num_images
        
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        
        images, targets = self.transform(images, targets, roi_sizes=self.roi_sizes)
        image_size = images.tensors.shape[-2:]
        features = self.backbone(images.tensors)
        features = self.fpn(features)
#         if isinstance(features, torch.Tensor):
#             features = OrderedDict([('0', features)])
        
        losses, outputs, annotations = {}, {}, {}
        for task_id, header in self.headers.items():
            # extract feature_map and target for each header
            task_size = self.roi_sizes[task_id]
            feature_maps = self.feature_maps[task_id]
            if targets is not None:
                task_targets = [_['anns'][task_id] if task_id in _['anns'] else {} for _ in targets]
                num_anns_per_target = [len(_['anns'][task_id]) if task_id in _['anns'] else 0 for _ in targets]
                task_features, task_gts = self.fpn(features, image_size, task_size, feature_maps, task_targets)
            else:
                raise NotImplementedError("Developing...")
                task_targets = generate_rois(image_size, task_size)
                task_features, task_gts = self.fpn(features, image_size, task_size, feature_maps, task_targets)
                num_anns_per_target = len(task_gts)
            
            task_outputs, task_losses = header(task_features, task_size, task_gts)
            # update annotations, losses, outputs
            losses.update({"{}_{}".format(task_id, k): v for k, v in task_losses.items()})
            outputs[task_id] = split_by_sizes(task_outputs, num_anns_per_target)
            annotations[task_id] = task_targets
        
        for edge, constrain in self.constrains.items():
            root, node = edge.split('_')
            task_losses = constrain(outputs[root], outputs[node], annotations[root], annotations[node])
            # update contradiction losses
            losses.update({"{}_{}".format(edge, k): v for k, v in task_losses.items()})

        # outputs = self.transform.postprocess(outputs, images.image_sizes, original_image_sizes)
        
#         p = torch.rand(100).to(images.tensors.device)*(1-self.debug_buffer)+self.debug_buffer
#         self.debug_buffer += 0.001
#         losses.update({"seg10x_det40x_bce_loss": -torch.mean(torch.log(p))})
        
        # outputs = self.postprocess(outputs, task_sizes, original_image_sizes)
        return losses, outputs


    ## TODO: postprocess all results and apply constrains
    def postprocess(self, result, task_sizes, original_image_sizes=None):
        # result: List[Dict[str, Tensor]]
        # task_sizes: List[Tuple[int, int]]
        # original_image_size: List[Tuple[int, int]]
        if self.training:
            return result
        original_image_sizes = original_image_sizes or task_sizes
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, task_sizes, original_image_sizes)):            
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result
    
    def save_config(self, filepath):
        with open(filepath, 'w') as f:
            yaml.safe_dump(self.configs, f, default_flow_style=False)


def build_module(cfg):
    pass