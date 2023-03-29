from .utils_seg import *

class PanopticSeg(torch.nn.Module):
    def __init__(self, config):
        super(PanopticSeg, self).__init__()
        self.config = config
        self.config['featmap_names'] = list(self.config['feature_maps'].keys())
        self.connector = PanopticFeatureConnector(
            config['in_channels'], config['in_channels'], config['feature_maps'], mode='bilinear',
        )
        
        layers = [
            torch.nn.Conv2d(config['in_channels'], config['num_classes'], kernel_size=1),
            torch.nn.Softmax2d(),
        ]
        scale_factor = config['scale_factor']
        if scale_factor is not None and scale_factor != 1:
            layers = [
                torch.nn.Upsample(scale_factor=scale_factor, mode=config['resize_mode'], align_corners=True)
            ] + layers
        self.layers = torch.nn.Sequential(*layers)
        self.criterion = SoftDiceLoss(config['class_weight'])
    
    def forward(self, features, image_size, roi_size, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        features = self.connector(features)
        task_features, task_gts, num_anns_per_target = \
            extract_roi_feature_maps(features, image_size, roi_size, targets=targets)
        res = self.layers(task_features['0'])
        
        losses = {}
        if targets is not None:
            masks = torch.stack([_['masks'] for _ in task_gts])
            h, w = masks.shape[-2], masks.shape[-1]
            res = torch.nn.functional.interpolate(
                res, size=(h, w), mode='bilinear', align_corners=True)
            losses.update({'soft_iou_loss': (1 + self.criterion(res, masks))})
        
        res = split_by_sizes(res, num_anns_per_target)
        
        return res, losses

