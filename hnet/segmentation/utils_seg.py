import torch
import numbers
from ..utils import *

class PanopticFeatureConnector(torch.nn.Module):
    def __init__(self, in_channels, out_channel, feature_maps, mode='bilinear'):
        """ Panoptic FPN: https://arxiv.org/pdf/1901.02446.pdf
            Starting from the deepest FPN level (at 1/32 scale), we perform three upsampling stages to yield a feature map at 1/4 scale, where each upsampling stage consists of 3×3 convolution, group norm [54], ReLU, and 2× bilinear upsampling. The result is a set of feature maps at the same 1/4 scale, which are then element-wise summed. A final 1×1 convolution, 4× bilinear upsampling, and softmax are used to generate the per-pixel class labels at the original image resolution. In addition to stuff classes, this branch also outputs a special 'other' class for all pixels belonging to objects. We use a standard FPN configuration with 256 output channels per scale, and our semantic segmentation branch reduces this to 128 channels.
        """
        super().__init__()
        if isinstance(in_channels, numbers.Number):
            in_channels = [in_channels] * len(feature_maps)
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.feature_maps = feature_maps
        
        self.layers = torch.nn.ModuleDict()
        for idx, (in_c, featmap_name) in enumerate(zip(in_channels, feature_maps)):
            blocks = [
                torch.nn.Conv2d(in_c, out_channel, 3, stride=1, padding=1, bias=False),
                torch.nn.GroupNorm(num_groups=32, num_channels=out_channel),
                torch.nn.ReLU(inplace=True),
            ]
            if idx > 0: 
                blocks += [torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),]
            
            for _ in range(idx-1):
                blocks += [
                    torch.nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
                    torch.nn.GroupNorm(num_groups=32, num_channels=out_channel),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ]
            
            self.layers[featmap_name] = torch.nn.Sequential(*blocks)
    
    def get_layer(self, in_channel, out_channel, stride):
        blocks = [
            torch.nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=32, num_channels=out_channel),
            torch.nn.ReLU(inplace=True),
        ]
        if idx > 0: 
            blocks += [torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),]

        for _ in range(idx-1):
            blocks += [
                torch.nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
                torch.nn.GroupNorm(num_groups=32, num_channels=out_channel),
                torch.nn.ReLU(inplace=True),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ]
        
        return torch.nn.Sequential(*blocks)
    
    def forward(self, features):
        res = [layer(features[k]) for k, layer in self.layers.items()]
        return {'0': sum(res)}


