from ..utils import *


def get_rcnn_config(config, num_classes, masks=True, keypoints=False):
    assert masks in [None, True, False]
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    default_config = {
        ## backbone
        # 'backbone': None,
        'featmap_names': None,
        'in_channels': None,
        ## rpn
        'rpn_params': {
            'anchor': {
                'sizes': [[32], [64], [128], [256], [512]],
                'aspect_ratios': [[0.5, 1.0, 2.0]] * 5,
            }, 
            'rpn': {
                'fg_iou_thresh': 0.7, 
                'bg_iou_thresh': 0.3,
                'batch_size_per_image': 256, 
                'positive_fraction': 0.5,

                'pre_nms_top_n': {'training': 2000, 'testing': 1000},
                'post_nms_top_n': {'training': 2000, 'testing': 1000},
                'nms_thresh': 0.7,
            },
        },
        ## transform
        'transform': {
            'min_size': 800, 'max_size': 1333, 
            'image_mean': [0.485, 0.456, 0.406], 
            'image_std': [0.229, 0.224, 0.225],
        },
        ## roi
        'roi_params': {
            ## roi predictor
            'roi': {
                # Faster R-CNN training
                'fg_iou_thresh': 0.5, 
                'bg_iou_thresh': 0.5,
                'batch_size_per_image': 512, 
                'positive_fraction': 0.25,
                'bbox_reg_weights': None,
                # Faster R-CNN inference
                'score_thresh': 0.05, 
                'nms_thresh': 0.5, 
                'detections_per_img': 100,
            },
            ## box predictor
            'box': {
                'num_classes': 2,
                'roi_output_size': 7, 
                'roi_sampling_ratio': 2,
                'layers': [1024, 1024], 
            },
            'mask': None,
            'keypoint': None, 
        },
    }

    ## mask predictor
    default_config_mask = {
        'num_classes': 2,
        'roi_output_size': 14, 
        'roi_sampling_ratio': 2,
        'layers': [256, 256, 256, 256],
        'dilation': 1,
        'dim_reduced': 256,
    }
    ## organize configs for masks
    has_mask = 'roi_params' in config and 'mask' in config['roi_params'] and config['roi_params']['mask'] is not None
    if (masks is False) or (masks is None and not has_mask):
        default_config_mask = None
    elif masks is not False and has_mask:
        deep_update(default_config_mask, config['roi_params']['mask'])

    ## keypoint predictor
    default_config_keypoint = {
        'num_keypoints': 19,
        'roi_output_size': 14, 
        'roi_sampling_ratio': 2,
        'layers': [512] * 8,
    }
    ## organize configs for keypoints
    has_kps = 'roi_params' in config and 'keypoint' in config['roi_params'] and config['roi_params']['keypoint'] is not None
    if (keypoints == 0) or (keypoints is None and not has_kps):
        default_config_keypoint = None
    elif (keypoints is None or keypoints > 0) and has_kps:
        deep_update(default_config_keypoint, config['roi_params']['keypoint'])

    deep_update(default_config, config)
    default_config['roi_params']['mask'] = default_config_mask
    default_config['roi_params']['keypoint'] = default_config_keypoint

    ## Fix num_classes in box and (maybe) mask, num_keypoints in keypoint
    if num_classes is not None:
        default_config['roi_params']['box']['num_classes'] = num_classes
        if 'mask' in default_config['roi_params'] and default_config['roi_params']['mask'] is not None:
            default_config['roi_params']['mask']['num_classes'] = num_classes
    if keypoints is not None and keypoints > 0:
        default_config['roi_params']['keypoint']['num_keypoints'] = keypoints

    return default_config


def load_pretrain(model, pretrained, layers="backbone+fpn+headers"):
    if pretrained:
        if isinstance(pretrained, str):
            weights = torch.load(pretrained, map_location='cpu')
        else:
            if model.roi_heads.has_mask():
                m = tmdet.maskrcnn_resnet50_fpn(
                    pretrained=True, progress=False, pretrained_backbone=False)
            elif model.roi_heads.has_keypoint():
                m = tmdet.keypointrcnn_resnet50_fpn(
                    pretrained=True, progress=False, pretrained_backbone=False)
            else:
                m = tmdet.fasterrcnn_resnet50_fpn(
                    pretrained=True, progress=False, pretrained_backbone=False)
            weights = m.state_dict()

        layers = set(layers.split('+'))
        if 'headers' not in layers:
            weights = {k: v for k, v in weights.items() 
                       if k.startswith('backbone')}
        if 'fpn' not in layers:
            weights = {k: v for k, v in weights.items() 
                       if not k.startswith('backbone.fpn')}
        if 'backbone' not in layers:
            weights = {k: v for k, v in weights.items() 
                       if not k.startswith('backbone') or k.startswith('backbone.fpn')}

        try:
            model.load_state_dict(weights, strict=False)
        except RuntimeError as e:
            print(e)


def project_roi_results_on_image(results, targets, image_shape=None):
    h, w = image_shape[0], image_shape[1]
    
    outputs = defaultdict(list)
    for r, t in zip(results, targets):
        x0, y0, x1, y1 = t['roi']
        r['boxes'][:, 0] += x0
        r['boxes'][:, 1] += y0
        r['boxes'][:, 2] += x0
        r['boxes'][:, 3] += y0
        
        if image_shape is not None:
            r['boxes'] = torchvision.ops.boxes.clip_boxes_to_image(r['boxes'], (h, w))
        
        for k, v in r.items():
            outputs[k].append(v)
    
    return {k: torch.cat(v) for k, v in outputs.items()}


class DetectionFeatureConnector(torch.nn.Module):
    def __init__(self, in_channels, out_channel, stride, feature_maps, mode='upsample'):
        """ For each feature maps, resize to match target size. 
            Up/down scale with nearest stride, then bilinear resize. 
            stride: resize feature 2 ** stride. Assume stride_w == stride_h for now. 
        Args:
            in_channels: int or List[int]
            out_channel: out_channel for task_features
            stride: up/down scale, log2
            feature_maps (dict): the feature_maps to keep {'task_feature_name': 'input_fpn_feature_name'/backbone_indices}
        """
        super().__init__()
        self.stride = stride
        self.feature_maps = feature_maps
        if isinstance(in_channels, numbers.Number):
            in_channels = [in_channels] * len(self.feature_maps)
        
        self.layers = torch.nn.ModuleDict()
        for idx, (in_c, featmap_name) in enumerate(zip(in_channels, self.feature_maps)):
            self.layers[featmap_name] = self.get_layer(in_c, out_channel, self.stride)
            
    
    def get_layer(self, in_channel, out_channel, stride):
        if in_channel != out_channel:
            blocks = [
                torch.nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1, bias=False),
                torch.nn.GroupNorm(num_groups=32, num_channels=out_channel),
                torch.nn.ReLU(inplace=True),
            ]
        else:
            blocks = []
        
        for _ in range(0, self.stride, 1):
            in_c = in_channel if _ == 0 else out_channel
            blocks += [
                torch.nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1, bias=False),
                torch.nn.GroupNorm(num_groups=32, num_channels=out_channel),
                torch.nn.ReLU(inplace=True),
                # torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ]
        for _ in range(0, self.stride, -1):
            # we simply use up sample here
            blocks += [
                torch.nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
                torch.nn.GroupNorm(num_groups=32, num_channels=out_channel),
                torch.nn.ReLU(inplace=True),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ]
        
        return torch.nn.Sequential(*blocks)
    
    def forward(self, features):
        return OrderedDict({k: self.layers[k](features[v]) for k, v in self.feature_maps.items()})



## different version has different forward, try another way for compatibility
class AnchorGenerator(tmdet.rpn.AnchorGenerator):
    def forward(self, task_size, feature_maps):
        """ Remove image_list from input: task_size: (roi_h, roi_w). """
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        num_images = feature_maps[0].shape[0]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(task_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(task_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype, device)
        
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(num_images):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        
        return anchors


class BoxPredictor(nn.Sequential):
    def __init__(self, in_channels, featmap_names, num_classes, 
                 roi_output_size=7, roi_sampling_ratio=2, layers=[1024, 1024]):
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=featmap_names, 
            output_size=roi_output_size, 
            sampling_ratio=roi_sampling_ratio)
        header = tmdet.faster_rcnn.TwoMLPHead(
            in_channels=in_channels * roi_pooler.output_size[0] ** 2,
            representation_size=layers[0],
        )
        predictor = tmdet.faster_rcnn.FastRCNNPredictor(
            in_channels=layers[-1], num_classes=num_classes)
        
        super(BoxPredictor, self).__init__(
            OrderedDict([
                ('box_roi_pool', roi_pooler),
                ('box_head', header),
                ('box_predictor', predictor),
            ])
        )