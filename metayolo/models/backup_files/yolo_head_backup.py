from collections import OrderedDict

from .layers import *
from .utils_general import check_anchor_order, nms_per_image, xyxy2xywhn, xyxy2xywh, xywh2xyxy, paired_box_iou
from .utils_torch import torch_meshgrid, one_hot_labels
from .loss import DetLoss, SegLoss
# from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
# from torchvision.models.detection.roi_heads import paste_masks_in_image
from torch_scatter import scatter_max

ROI_ALIGN = False


class Detect(nn.Module):  # detection layer
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, ch, anchors, stride, nc, nc_masks=None, 
                 dim_reduced=256, mask_output_size=28, dilation=1, 
                 nms_params={}, loss_hyp={}):
        super().__init__()
        self.inplace = False  # use in-place ops (e.g. slice assignment)
        assert len(ch) == len(anchors) == len(stride), "ch, anchors, stride should have same length."

        ## anchors, strides, channels
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.ch = ch  # number of channels
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor

        stride = torch.tensor(stride).float()  # [8.0, 16.0, 32.0, 64.0]
        anchors = torch.tensor(anchors).float().view(len(anchors), -1, 2) / stride.view(-1, 1, 1)  # shape(nl,na,2)
        self.register_buffer('stride', stride)
        self.register_buffer('anchors', anchors)

        for idx in range(self.nl):
            # ny = nx = int(640/2**(3+i))
            # grid, anchor_grid = self._make_grid(self.anchors[i], self.stride[i], nx, ny)
            grid, anchor_grid = None, None
            self.register_buffer(f'grid_{idx}', grid)
            self.register_buffer(f'anchor_grid_{idx}', anchor_grid)
        check_anchor_order(self)

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self._initialize_biases()  # initialize bias for yolo detect layer

        # get losses
        autobalance = False
        ssi = list(self.stride).index(16)+1 if autobalance else 0  # stride 16 index layer
        self.det_loss = DetLoss(self.nc, self.nl, loss_hyp, ssi=ssi)
        self.seg_loss = SegLoss()
        self.nms_params = self.get_nms_params(nms_params)

        ## Add instance segmentation header
        if nc_masks:
            # assert nc_masks in [True, 1, self.nc], "If specified, nc_masks need to be True, 1, or same as detection classes! "
            self.nc_masks = self.nc if nc_masks is True else nc_masks  # number of classes for masks, same as nc, 1, or None
            self.mask_output_size = mask_output_size
            # self.m = nn.ModuleList(Conv(x, dim_reduced, k=3, act=False) for x in in_channels)  # mask fuse
            # self.seg = nn.ModuleList(Conv(self.ch[i], (self.ch[i-1] if i > 0 else dim_reduced), k=3, act=False) 
            #                          for i in range(self.nl))
            self.seg = nn.ModuleList(Conv(self.ch[i], dim_reduced, k=3, act=True) 
                                     for i in range(self.nl-1, -1, -1))  # top-down for easy scripting
            # self.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
#             self.seg_h = nn.Sequential(OrderedDict([
#                 ("maskrcnn_heads", MaskRCNNHeads(dim_reduced, (256, 256, 256, 256), 1)),
#                 ("maskrcnn_preds", MaskRCNNPredictor(256, 256, self.nc_masks)),
#             ]))
            self.seg_h = nn.Sequential(OrderedDict([
                ("maskrcnn_heads", MaskRCNNHeads(dim_reduced, (256, 256, 256, 256), 1)),
                ("maskrcnn_preds", MaskRCNNPredictor(256, 256, self.nc_masks)),
            ]))
        else:
            self.nc_masks = 0
            self.seg = None
            self.seg_h = None
            self.mask_output_size = None

    def forward(self, x: List[torch.Tensor], image_shapes, targets=None, compute_masks=True):
        # 1. training: targets rois are required, return x, masks
        # 2. validation: targets rois are provided, return (output, x), masks
        # 3. inference: rois not provided, nms on output, return (output_filtered, x), masks_filtered

        if self.training:
            assert targets is not None
            compute_losses, compute_outputs = True, False
        else:
            compute_losses, compute_outputs = targets is not None, True
        compute_masks = (self.nc_masks > 0) * compute_masks

        # Run det header on feature map
        dets = []
        for i, layer in enumerate(self.m):
            f = layer(x[i])  # conv
            bs, _, ny, nx = f.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            det = f.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            dets.append(det)

        # calculate predictions
        if self.training and not compute_masks:  # only in training and ignore masks, we don't need to compute proposals
            preds = []
        else:
            preds = self.compute_proposals(dets)  # [(bs, 20x20, xywhcoefs), (bs, 40x40, xywhcoefs), ...]

        if compute_masks > 0.:  # Run seg header on feature map
            panoptic_feature_maps = []
            for i, layer in enumerate(self.seg, 1):
                panoptic_feature_maps.append(layer(x[-i]))
            panoptic_feature_maps = panoptic_feature_maps[::-1]
        else:
            panoptic_feature_maps = None

        ## compute losses and outputs
        if compute_losses:
            losses = self.compute_losses(dets, preds, panoptic_feature_maps, image_shapes, targets, compute_masks=compute_masks)
        else:
            losses = {}

        if compute_outputs:
            outputs = self.compute_outputs(preds, panoptic_feature_maps, image_shapes, compute_masks=compute_masks)
        else:
            outputs = []

        return losses, outputs

    def compute_proposals(self, dets):
        preds = []
        for i in range(self.nl):
            y = dets[i].sigmoid()

            # update grid and anchor_grid
            grid_i = getattr(self, f'grid_{i}')
            if grid_i is None or grid_i.shape[2:4] != y.shape[2:4] or self.onnx_dynamic:
                bs, na, ny, nx, no = y.shape
                grid, anchor_grid= self._make_grid(self.anchors[i], self.stride[i], nx, ny)
                # setattr(self, f'grid_{i}', grid)
                # setattr(self, f'anchor_grid_{i}', anchor_grid)
            else:
                grid, anchor_grid = getattr(self, f'grid_{i}'), getattr(self, f'anchor_grid_{i}')

            if self.inplace:  # inplace cause gradient problem, don't use
                y[..., 0:2] = (y[..., 0:2] * 2. + grid) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
            else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                xy, wh, conf = y.tensor_split((2, 4), -1)  # torch 1.8.0
                xy = (xy * 2 + grid) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * anchor_grid  # wh
                y = torch.cat((xy, wh, conf), -1)
                preds.append(y)
        # print(f"compute_proposals", [y.shape for y in preds])

        return preds

    def compute_losses(self, dets, preds, features, image_shapes, targets, compute_masks=True):
        # gts: [img_id, x, y, w, h, one_hot_labels...]
        gts = torch.cat([
            torch.cat([torch.full_like(_['boxes'][:, :1], idx), xyxy2xywh(_['boxes'], clip=True, eps=0.0)], -1)
            for idx, _ in enumerate(targets)
        ])
        gt_labels = torch.cat([_['labels'] for _ in targets])
        tbox, tids, indices, anchors = self.matcher(dets, gts)  # targets, tcls start from 1

        # det loss
        tcls = [gt_labels[_] for _ in tids]
        det_loss, det_loss_items = self.det_loss(dets, tcls, tbox, indices, anchors)

        # seg loss
        if compute_masks > 0.:
            mask_features, proposals, gt_proposals, obj_ids = [], [], [], []
            for i in range(self.nl):
                y, f, obj_id = preds[i], features[i], tids[i]
                bs, nc, h, w = f.shape
                # which pred are matched with target in layer i: image_b, anchor_a, grid_j, grid_i
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                boxes = xywh2xyxy(y[b, a, gj, gi, :4].detach())
                gt_boxes = xywh2xyxy(gts[obj_id][:, 1:]*gts.new([w, h, w, h])*self.stride[i])
#                 print(f"For layer {i}/{self.nl}: boxes={boxes.shape}, gts={gts[obj_id].shape}")
#                 print(f"For layer {i}/{self.nl}: feature={f.shape}, scale=1/{self.stride[i]}")
                fmap = torchvision.ops.roi_align(f, torch.cat([b[:, None], gt_boxes], -1), self.mask_output_size//2, 
                                                 spatial_scale=1/self.stride[i], sampling_ratio=2, aligned=ROI_ALIGN)
                proposals.append(boxes)
                obj_ids.append(tids[i])
                gt_proposals.append(gt_boxes)
                mask_features.append(fmap)

            mask_features = torch.cat(mask_features)
            proposals = torch.cat(proposals)
            gt_proposals = torch.cat(gt_proposals)
            obj_ids = torch.cat(obj_ids)

            # Only keep good box for mask loss
            box_ious = paired_box_iou(proposals, gt_proposals)  # torchvision.ops.box_iou(boxes, gt_boxes).diag()
            max_ious, indices = scatter_max(box_ious, obj_ids)
            keep = indices[max_ious >= 0.8]  # trim un-matched box
            # print(f"{len(obj_ids)} -> {len(keep)}")

            mask_logits = self.seg_h(mask_features[keep])
            proposals, gt_proposals = proposals[keep], gt_proposals[keep]
            gt_labels = torch.cat(tcls)[keep]
            gt_masks = torch.cat([_['masks'] for _ in targets])[obj_ids[keep], None] * 1.0
            mask_targets = gt_masks

            # gt_masks = paste_masks_in_image(gt_masks, gt_proposals, [640, 640], padding=1).squeeze(1)
            # mask_targets = torchvision.ops.roi_align(gt_masks, proposals.split(1), 
            #                                          (mask_logits.shape[-2], mask_logits.shape[-1]), 1.0)

#             if True:
#                 mask_patches_1 = [m[..., int(y0):int(y1), int(x0):int(x1)]
#                                   for (x0, y0, x1, y1), m in zip(proposals, gt_masks)]
#                 mask_patches_2 = [m[..., int(y0):int(y1), int(x0):int(x1)]
#                                   for (x0, y0, x1, y1), m in zip(gt_proposals, gt_masks)]
#                 ious = [proposal_m.sum()/gt_m.sum() for proposal_m, gt_m in zip(mask_patches_1, mask_patches_2)]
#                 box_ious = paired_box_iou(proposals, gt_proposals)

#                 print("======================")
#                 import matplotlib.pyplot as plt
#                 print(mask_targets.shape, mask_logits.shape)
#                 fig, axes = plt.subplots(4, 6, figsize=(24, 16))
#                 ax = axes.ravel()
#                 for i in range(min(len(ax)//3, len(mask_targets))):
#                     ax[i*3].imshow(gt_masks[i][0].cpu().detach().numpy())
#                     ax[i*3].set_title(f"{ious[i]}")
#                     # ax[i*3+1].imshow(mask_patches_1[i].permute(1, 2, 0).cpu().detach().numpy())
#                     ax[i*3+1].set_title(f"{box_ious[i]}")
#                     # ax[i*3+2].imshow(mask_patches_2[i].permute(1, 2, 0).cpu().detach().numpy())
#                     # ax[i*3+2].set_title(f"{gt_boxes[i]}")
#                     ax[i*3+1].imshow(mask_targets[i][0].cpu().detach().numpy())
#                     ax[i*3+2].imshow(mask_logits.sigmoid()[i][0].cpu().detach().numpy())
#                 plt.show()
#                 print("======================")

            mask_loss = self.seg_loss(mask_logits, mask_targets, gt_labels)  #, gt_boxes=gt_proposals[keep],)
        else:
            mask_loss = torch.zeros_like(det_loss)

        return {'det_loss': det_loss, 'mask_loss': mask_loss, 'loss_items': {**det_loss_items, 'mask': mask_loss.detach()}}

    def multiscale_roi_align(self, features, boxes, levels, mapper=None):
        assert len(features) == self.nl
        num_rois = len(boxes)
        num_channels = features[0].shape[1]
        M = self.mask_output_size//2

        if mapper is not None:
            levels = mapper(boxes)

        dtype, device = features[0].dtype, features[0].device
        result = torch.zeros((num_rois, num_channels, M, M), dtype=dtype, device=device,)
        for i in range(self.nl):
            idx_in_level = torch.where(levels == i)[0]
            fmap = torchvision.ops.roi_align(features[i], boxes[idx_in_level], (M, M), 
                                             spatial_scale=1/self.stride[i], sampling_ratio=2, aligned=ROI_ALIGN)
            result[idx_in_level] = fmap.to(result.dtype)

        return result

    def compute_outputs(self, preds, features, image_shapes, compute_masks=True):
        # det_output has boxes + scores, anchor_features in inference
#         outputs = []
#         for i in range(self.nl):
#             y = preds[i]
#             bs, na, ny, nx, no = y.shape
#             grid = torch.cartesian_prod(torch.tensor([i]), torch.arange(na), torch.arange(ny), torch.arange(nx))
#             outputs.append(y.view(bs, -1, no))
#             indices.append(grid.to(y.device))
        preds = torch.cat([F.pad(y.view(y.shape[0], -1, self.no), [0,1], value=idx) 
                           for idx, y in enumerate(preds)], 1)
        outputs = nms_per_image(preds, nc=self.nc, **self.nms_params) # [[x0, y0, x1, y1, label, score, extra:],]
        results = [{'boxes': _[:, :4], 'labels': _[:, 4] + 1, 'scores': _[:, 5],} for _ in outputs]  # 'extra': _[:,6:]

        if compute_masks > 0.:
            n_obj_per_image = [len(_) for _ in outputs]
            outputs = torch.cat([F.pad(_, [1,0], value=idx) for idx, _ in enumerate(outputs)])  # pad batch_id in first column
            mask_features = self.multiscale_roi_align(features, boxes=outputs[:, :5], levels=outputs[:, 7])
            mask_logits = self.seg_h(mask_features)
            mask_probs = mask_logits.sigmoid().split(n_obj_per_image, dim=0)

            for r, m in zip(results, mask_probs):
                if self.nc <= self.nc_masks:
                    labels = r['labels']  # r['labels'] starts from 1
                    index = torch.arange(len(m), device=labels.device)
                    m = m[index, labels - 1][:, None]
                r['masks'] = m
                # r['masks'] = paste_masks_in_image(m, boxes, img_shape, padding=1).squeeze(1)

        return results

    def matcher(self, p, gts):
        device = gts.device
        na, nt = self.na, len(gts)  # number of anchors, number of gt objects 

        # append targets to (obj_id,img_id,x,y,w,h,one_hot_labels,...)
        targets = torch.cat([torch.arange(nt, device=device)[:,None], gts[:,:6]], -1)  # add obj_id, ignore label
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # na*nt, [[0,0,...],[1,1,...],[2,2,...]]
        targets = torch.cat([ai[:, :, None], targets.repeat(na, 1, 1)], 2)
        # [(anchor_id:0,obj_id,img_id,x,y,w,h), (anchor_id:1,obj_id,img_id,x,y,w,h), ...]

        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                           ], device=device).float() * g  # offsets

        tbox, tids, indices, anch = [], [], [], []
        for i in range(self.nl):
            anchors = self.anchors[i].to(device)
            f_h, f_w = p[i].shape[2:4]
            gain = torch.tensor([1, 1, 1, f_w, f_h, f_w, f_h], device=device)
            t = targets * gain

            if nt:
                # Matches
                r = t[:, :, 5:7] / anchors[:, None]  # gt_box wh ratio relative to anchor
                keep = torch.max(r, 1. / r).max(2)[0] < self.det_loss.hyp['anchor_t']  # keep non-flat box (na*nt)
                # keep = wh_iou(anchors, t[:, 5:7]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[keep]  # filter, n_matched_gt (< na*nt) * 8

                # Offsets
                gxy = t[:, 3:5]  # grid xy
                gxi = gain[[3, 4]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                keep = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[keep]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[keep]
                # print(f"For layer {i}/{self.nl}: matched {t.shape} targets")
                # if len(t) < 20:
                #     print(t)
            else:
                t = targets[0]
                offsets = 0

            # Define
            gxy = t[:, 3:5]  # grid xy
            gwh = t[:, 5:7]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            anchor_id, obj_id, image_id = t[:, :3].long().T
            # c = t[:, 7:]

            # Append
            indices.append((image_id, anchor_id, gj.clamp_(0, f_h - 1), gi.clamp_(0, f_w - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[anchor_id])  # anchors
            # tcls.append(c)  # class
            tids.append(obj_id)  # object_id for mapping gt_box and gt_mask

        return tbox, tids, indices, anch


    def _make_grid(self, anchors, stride, nx=20, ny=20):
        d, t = anchors.device, anchors.dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch_meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (anchors * stride).view((1, self.na, 1, 1, 2)).expand(shape)

        return grid, anchor_grid

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        for mi in self.m:  # from
            b = mi.bias.detach().view(self.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def get_nms_params(self, args={}):
        default_args = {'conf_thres': 0.15, 'iou_thres': 0.45, 'multi_label': False, 'max_det': 300}
        return {k: args.get(k, v) for k, v in default_args.items()}

    def merge_outputs(self, r):
        """ Merge outputs from different rois under same amp. 
            rois and outputs should under same amplification.
        """
        boxes = torch.cat([_['boxes'] + _['boxes'].new([_['roi'][0], _['roi'][1], _['roi'][0], _['roi'][1]]) for _ in r])
        labels = torch.cat([_['labels'] for _ in r])
        scores = torch.cat([_['scores'] for _ in r])
        res = {'boxes': boxes, 'labels': labels, 'scores': scores}
        
        if 'masks' in r[0]:
            res['masks'] = torch.cat([_['masks'] for _ in r])

        return res
    
    def rescale_outputs(self, r, scale=1.0):
        """ Rescale outputs to another amplification. """
        if scale != 1.0:
            r['boxes'] *= scale
        
        return r

