# For compatibility purpose only, make old models loadable with torch.load
from collections import OrderedDict, deque

from .layers import *
from .utils_general import check_anchor_order, nms_per_image, xyxy2xywhn, xyxy2xywh, xywh2xyxy, paired_box_iou
from .utils_torch import torch_meshgrid, one_hot_labels
from .loss import DetLoss, SegLoss
# from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
# from torchvision.models.detection.roi_heads import paste_masks_in_image
from torch_scatter import scatter_max
from .. import LOGGER, check_version


ROI_ALIGN = False


class BuffersDict(nn.Module):
    def __init__(self, x: Dict[str, Optional[torch.Tensor]] = {}):
        super().__init__()
        for k, v in x.items():
            self.register_buffer(k, v)


class Detect(nn.Module):  # detection layer
    # onnx_dynamic = False  # ONNX export parameter

    def __init__(self, ch: List[int], anchors: List[List[int]], strides: List[int], 
                 nc: int, masks: Dict[int, int] = {}, dim_reduced: int = 256, 
                 mask_output_size: int = 28, multi_label: bool = False, 
                 nms_params: Dict[str, float] = {},
                 loss_hyp: Dict[str, float] = {}, 
                 default_input_size: Optional[int] = 640,
                 is_scripting: bool = False,
                ):
        super().__init__()
        # self.inplace = is_scripting  # use in-place ops in scripting (e.g. slice assignment)
        assert len(ch) == len(anchors) == len(strides), f"ch, anchors, strides should have same length."

        ## anchors, strides, channels
        self.ch = ch  # number of channels
        self.nl = len(ch)  # number of detection layers
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.na = len(anchors[0]) // 2  # force using list of anchors in yolov5
        self.default_input_size = default_input_size  # default input size, for calculate grid, and ms+hier
        # if isinstance(anchors, (list, tuple)):  # number of anchors
        #     self.na = len(anchors[0]) // 2
        # else:
        #     self.na = anchors

        ## register a dictionary: {node: all_descendants}
        self.descendants: Dict[int, List[int]] = {}
        self.get_descendants(self.build_hierarchical_tree())

        ## register strides, anchors, grids, anchor_grids
        self.anchors = nn.ModuleList([])
        strides = torch.tensor(strides).float()  # [8.0, 16.0, 32.0, 64.0]
        anchors = torch.tensor(anchors).float().view(len(anchors), -1, 2) / strides.view(-1, 1, 1)  # shape(nl,na,2)
        for stride, anchor in zip(strides, anchors):
            if self.default_input_size is not None:
                ny = nx = int(self.default_input_size/stride)
                grid, anchor_grid = self._make_grid(anchor, stride, nx, ny)
            else:
                grid, anchor_grid = None, None
            buffers = BuffersDict({
                'stride': stride, 'anchor': anchor, 
                'grid': grid, 'anchor_grid': anchor_grid,
            })
            self.anchors.append(buffers)
        # self.register_buffer('strides', strides)
        # self.register_buffer('anchors', anchors)
        # self.register_buffer(f'grid_{idx}', grid)
        # self.register_buffer(f'anchor_grid_{idx}', anchor_grid)
        # check_anchor_order(self)  # don't need this

        self.m = self.build_det_layers()
        self.initialize_biases()  # initialize bias for yolo detect layer

        # get losses
        autobalance = False
        ssi = list(strides).index(16)+1 if autobalance else 0  # stride 16 index layer
        if not is_scripting:
            self.det_loss = DetLoss(self.nc, self.nl, loss_hyp, ssi=ssi)
        # Dictionary inputs to traced functions must have consistent type.
        self.nms_params: Dict[str, float] = self.get_nms_params(nms_params)
        self.multi_label: bool = multi_label

        ## Add instance segmentation header, mask_indices rule:
        # 0: object masks (general masks applied to "unclassified" objects), 
        # -1: means ignore masks, masks for this cls_xx will be all zero.
        # 1~nc_mask: something like {cls_01: mask_01, cls_02: mask_02, ...}, allow duplicate mask_id
        # if a cls_xx is not given, will be set to 0 (default general mask) by default
        mask_indices = torch.tensor([masks.get(idx, 0) for idx in range(self.nc+1)])
        self.nc_masks = mask_indices.max().item() + 1
        self.register_buffer('mask_indices', mask_indices)
        self.dim_reduced = dim_reduced

        if self.nc_masks > 0:
            self.mask_output_size = mask_output_size
            self.seg, self.seg_h = self.build_seg_layers()
            self.aligned = ROI_ALIGN  # avoid jit script error with global variable
            if not is_scripting:
                self.seg_loss = SegLoss(loss_hyp)
        else:
            self.mask_output_size = None
            self.seg, self.seg_h = None, None
            if not is_scripting:
                self.seg_loss = None

    def build_det_layers(self):
        return nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in self.ch)  # output conv

    def build_seg_layers(self):
        # self.m = nn.ModuleList(Conv(x, self.dim_reduced, kernel_size=3, act=False) for x in in_channels)  # mask fuse
        # self.seg = nn.ModuleList(Conv(self.ch[i], (self.ch[i-1] if i > 0 else self.dim_reduced), kernel_size=3, act=False) for i in range(self.nl))

        # self.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        # self.seg_h = nn.Sequential(OrderedDict([
        #     ("maskrcnn_heads", MaskRCNNHeads(self.dim_reduced, (256, 256, 256, 256), 1)),
        #     ("maskrcnn_preds", MaskRCNNPredictor(256, 256, self.nc_masks)),
        # ]))
        seg = nn.ModuleList(Conv(self.ch[i], self.dim_reduced, kernel_size=3, act=True) 
                            for i in range(self.nl-1, -1, -1))  # top-down for easy scripting
        seg_h = nn.Sequential(OrderedDict([
            ("maskrcnn_heads", MaskRCNNHeads(self.dim_reduced, (256, 256, 256, 256), 1)),
            ("maskrcnn_preds", MaskRCNNPredictor(256, 256, self.nc_masks)),
        ]))

        return seg, seg_h

    def forward(self, x: Dict[int, torch.Tensor], 
                targets: Optional[List[Dict[str, torch.Tensor]]] = None, 
                compute_masks: bool = True):
        # 1. training: targets rois are required, return x, masks
        # 2. validation: targets rois are provided, return (output, x), masks
        # 3. inference: rois not provided, nms on output, return (output_filtered, x), masks_filtered
        # Run det header on feature map
        dets: List[torch.Tensor] = []
        x = x[self.f] if isinstance(self.f, int) else [x[j] for j in self.f]
        for i, det_layer in enumerate(self.m):
            f = det_layer(x[i])  # conv
            bs, _, ny, nx = f.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            det = f.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            dets.append(det)

        if not torch.jit.is_scripting() and self.training:
            assert targets is not None
            compute_losses, compute_outputs = True, False
        else:
            compute_losses, compute_outputs = targets is not None, True
        compute_masks = (self.nc_masks > 0) and compute_masks

        # calculate predictions
        if not torch.jit.is_scripting() and self.training and not compute_masks:
            # only in training and ignore masks, we don't need to compute proposals
            preds = []
        else:
            # [(bs, 20x20, xywhcoefs), (bs, 40x40, xywhcoefs), ...]
            preds = self.compute_proposals(dets)

        # Run seg header on feature map
        mask_feature_maps: List[torch.Tensor] = []
        if compute_masks:
            for i, seg_layer in enumerate(self.seg, 1):
                mask_feature_maps.append(seg_layer(x[-i]))
        mask_feature_maps = mask_feature_maps[::-1]

        ## compute losses and outputs
        if not torch.jit.is_scripting() and compute_losses:
            losses = self.compute_losses(
                dets, preds, mask_feature_maps, targets, 
                compute_masks=compute_masks
            )
        else:
            losses = {}

        if compute_outputs:
            outputs = self.compute_outputs(preds, mask_feature_maps, compute_masks=compute_masks)
        else:
            outputs: List[Dict[str, Tensor]] = []

        return losses, outputs

    def compute_proposals(self, dets: List[torch.Tensor]) -> List[torch.Tensor]:
        preds: List[torch.Tensor] = []

        for i, buffer in enumerate(self.anchors):
            y = dets[i].sigmoid()
            bs, na, ny, nx, no = y.shape

            # update grid and anchor_grid
            # bug with getattr in scripting: second argument must be a string literal
            if buffer.grid is None:  # or self.onnx_dynamic:
                grid, anchor_grid = self._make_grid(buffer.anchor, buffer.stride, nx, ny)
            elif buffer.grid.shape[:2] != y.shape[2:4]:
                grid, anchor_grid = self._make_grid(buffer.anchor, buffer.stride, nx, ny)
            else:
                grid, anchor_grid = buffer.grid, buffer.anchor_grid

            grid = grid.expand(1, self.na, ny, nx, 2)
            anchor_grid = anchor_grid.view((1, self.na, 1, 1, 2)).expand(1, self.na, ny, nx, 2)
            if torch.jit.is_scripting():  # inplace cause gradient problem, don't use in training
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * buffer.stride  # xy
                y[..., 2:4] = (y[..., 2:4] * 2.) ** 2 * anchor_grid  # wh
            else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                xy, wh, conf = y.tensor_split((2, 4), -1)  # torch 1.8.0
                xy = (xy * 2. - 0.5 + grid) * buffer.stride  # xy
                wh = (wh * 2.) ** 2 * anchor_grid  # wh
                y = torch.cat((xy, wh, conf), -1)
            preds.append(y)

        return preds

    @torch.jit.unused
    def compute_losses(self, dets, preds, features, targets, compute_masks=True):
        # gts: [img_id, x, y, w, h, one_hot_labels...]
        gts = torch.cat([
            torch.cat([torch.full_like(_['boxes'][:, :1], idx), xyxy2xywh(_['boxes'], clip=True, eps=0.0)], -1)
            for idx, _ in enumerate(targets)
        ])
        gt_labels = torch.cat([one_hot_labels(_['labels'], self.nc) if _['labels'].dim() == 1 else _['labels'] for _ in targets])
        gt_labels_idx = torch.cat([_['labels'] for _ in targets])
        tbox, tids, indices, anchors = self.matcher(dets, gts)  # targets, tcls start from 1

        # det loss
        tcls = [gt_labels[_] for _ in tids]
        tcls_idx = [gt_labels_idx[_] for _ in tids]
        det_loss, det_loss_items = self.det_loss(dets, tcls, tbox, indices, anchors)

        # seg loss
        if compute_masks:
            mask_features, proposals, gt_proposals, obj_ids = [], [], [], []
            for i, buffer in enumerate(self.anchors):
                y, f, obj_id = preds[i], features[i], tids[i]
                bs, nc, h, w = f.shape
                # which pred are matched with target in layer i: image_b, anchor_a, grid_j, grid_i
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                boxes = xywh2xyxy(y[b, a, gj, gi, :4].detach())
                gt_boxes = xywh2xyxy(gts[obj_id][:, 1:]*gts.new([w, h, w, h])*buffer.stride)
                # print(f"For layer {i}/{self.nl}: boxes={boxes.shape}, gts={gts[obj_id].shape}")
                # print(f"For layer {i}/{self.nl}: feature={f.shape}, scale=1/{buffer.stride}")
                fmap = torchvision.ops.roi_align(f, torch.cat([b[:, None], gt_boxes], -1), self.mask_output_size//2, 
                                                 spatial_scale=1/buffer.stride, sampling_ratio=2, aligned=self.aligned)
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
            keep = indices[max_ious >= 0.8]  # trim un-matched box with iou smaller than 0.8
            # print(f"detected objects={len(obj_ids)} -> match_by_iou={len(keep)}")
            mask_logits = self.seg_h(mask_features[keep])
            proposals, gt_proposals = proposals[keep], gt_proposals[keep]
            mask_targets = torch.cat([_['masks'] for _ in targets])[obj_ids[keep], None] * 1.0
            gt_labels = torch.cat(tcls)[keep]
            ## Convert 28*28 masks in gt_box to 28*28 masks in pred_box
            # gt_masks = paste_masks_in_image(gt_masks, gt_proposals, [640, 640], padding=1).squeeze(1)
            # mask_targets = torchvision.ops.roi_align(gt_masks, proposals.split(1), 
            #                                          (mask_logits.shape[-2], mask_logits.shape[-1]), 1.0)

            ## Currently we use the lowest level label in hierachical labels.
            # We might want to get labels on all level for mask loss in future.
            gt_labels_hier = (gt_labels * torch.arange(self.nc+1, device=gt_labels.device)).max(-1)[1]
            mask_labels = self.mask_indices[gt_labels_hier]
            mask_loss = self.seg_loss(mask_logits, mask_targets, mask_labels)
        else:
            mask_loss = torch.zeros_like(det_loss)

        return {'det_loss': det_loss, 'mask_loss': mask_loss, 'loss_items': {**det_loss_items, 'mask': mask_loss.detach()}}

    def multiscale_roi_align(self, features: List[torch.Tensor], boxes: torch.Tensor, 
                             levels: torch.Tensor) -> torch.Tensor:
        assert len(features) == self.nl
        num_rois = len(boxes)
        num_channels = features[0].shape[1]
        M = self.mask_output_size//2

        ## TODO: add a function mapper to customize box to level
        # if mapper is not None:
        #     levels = mapper(boxes)

        dtype, device = features[0].dtype, features[0].device
        result = torch.zeros((num_rois, num_channels, M, M), dtype=dtype, device=device,)
        for i, buffer in enumerate(self.anchors):
            idx_in_level = torch.where(levels == i)[0]
            fmap = torchvision.ops.roi_align(features[i], boxes[idx_in_level], (M, M), 
                                             spatial_scale=1/buffer.stride, sampling_ratio=2, 
                                             aligned=self.aligned)
            result[idx_in_level] = fmap.to(result.dtype)

        return result

    def compute_outputs(self, preds: List[torch.Tensor], features: List[torch.Tensor], 
                        compute_masks: bool = True) -> List[Dict[str, torch.Tensor]]:
        # det_output has boxes + scores, anchor_features in inference
        # outputs = []
        # for i in range(self.nl):
        #     y = preds[i]
        #     bs, na, ny, nx, no = y.shape
        #     grid = torch.cartesian_prod(torch.tensor([i]), torch.arange(na), torch.arange(ny), torch.arange(nx))
        #     outputs.append(y.view(bs, -1, no))
        #     indices.append(grid.to(y.device))
        preds = torch.cat([F.pad(y.view(y.shape[0], -1, self.no), [0,1], value=float(idx)) 
                           for idx, y in enumerate(preds)], 1)
        outputs = nms_per_image(
            preds, nc=self.nc, 
            conf_thres=self.nms_params['conf_thres'],
            iou_thres=self.nms_params['iou_thres'],
            max_det=int(self.nms_params['max_det']), 
        )  # [[x0, y0, x1, y1, scores, extra:],]
        n_obj_per_image = [len(_['boxes']) for _ in outputs]

        if compute_masks and (sum(n_obj_per_image) > 0):
            # pad batch_id in first column
            proposals, levels = [], []  # Comprehension ifs are not supported yet
            for idx, _ in enumerate(outputs):
                if len(_['boxes']):
                    proposals.append(F.pad(_['boxes'], [1,0], value=float(idx)))
                    levels.append(_['extra'][:, 0])
            proposals = torch.cat(proposals)
            levels = torch.cat(levels)
            mask_features = self.multiscale_roi_align(features, boxes=proposals, levels=levels)
            mask_logits = self.seg_h(mask_features)
            mask_probs = mask_logits.sigmoid().split(n_obj_per_image, dim=0)
        else:
            mask_probs: List[torch.Tensor] = []  # avoid scripting error

        results = [{'boxes': _['boxes'], 'scores': _['scores'],} for _ in outputs]  # 'labels': _[:, 4] + 1, 'extra': _[:,6:]
        for idx, r in enumerate(results):
            r['scores'] = self.hierarchical_scores(r['scores'])
            if self.multi_label:
                r['labels'] = r['scores'] > self.nms_params['conf_thres']
            else:
                obj_scores = r['scores'][..., 0]
                cls_scores, cls_labels = r['scores'][..., 1:].max(1)
                r['scores'] = torch.where(cls_scores > self.nms_params['conf_thres'], cls_scores, obj_scores)
                r['labels'] = torch.where(cls_scores > self.nms_params['conf_thres'], cls_labels+1, -100)  # assign unclassified to -100
                if compute_masks and n_obj_per_image[idx] > 0:
                    m = mask_probs[idx]
                    mask_labels = self.mask_indices[r['labels'].clamp(min=0.)]   # convert -100 to 0, obj_labels 0(unclassified), 1~nc
                    # print(f"===", torch.stack([r['labels'], mask_labels], -1))
                    index = torch.arange(len(m), device=mask_labels.device)
                    r['masks'] = m[index, mask_labels][:, None]
                    # r['masks'] = paste_masks_in_image(m, boxes, img_shape, padding=1).squeeze(1)
                    r['masks'][mask_labels<0] = 0  # remove masks with labels == -1

        return results

    @torch.jit.unused
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
        for i, buffer in enumerate(self.anchors):
            anchors = buffer.anchor.to(device)
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

    def _make_grid(self, anchor: torch.Tensor, stride: torch.Tensor, 
                   nx: int = 20, ny: int = 20):
        d, t = anchor.device, anchor.dtype
        # shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        # yv, xv = torch_meshgrid(y, x)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2)
        anchor_grid = (anchor * stride)

        return grid, anchor_grid

    def initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, buffer in zip(self.m, self.anchors):
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / buffer.stride) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        for mi in self.m:  # from
            b = mi.bias.detach().view(self.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def get_nms_params(self, args={}):
        default_args = {'conf_thres': 0.15, 'iou_thres': 0.45, 'max_det': 300}
        return {k: float(args.get(k, v)) for k, v in default_args.items()}

    @torch.jit.unused
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

    @torch.jit.unused
    def rescale_outputs(self, r, scale=1.0):
        """ Rescale outputs to another amplification. """
        if scale != 1.0:
            r['boxes'] *= scale

        return r

    def hierarchical_scores(self, x: torch.Tensor) -> torch.Tensor:
        # Under python 3.6, self.descendants should be in correct order.
        # So we use inplace modification.
        for k, v in self.descendants.items():
            x[:, v] *= x[:, k:k+1]

        return x

    def get_descendants(self, node: Optional[Dict] = None):
        res: List[int] = []
        if node is not None:
            for k, v in node.items():
                res.append(k)
                c = self.get_descendants(v)
                if c:
                    self.descendants[k] = c
                    res += c

        return res

    def hierarchical_scores_bfs(self, x, inplace=True):
        # dynamic bfs deque, nested dictionary are not supported by jit. 
        # We predefine {node: childrens} with bottom up strategy.
        if not inplace:  # clone tensor
            x = x.detach().clone()

        # level order traversal
        classes = self.build_hierarchical_tree()
        queue = deque(classes.items())
        while queue:
            node, tree = queue.popleft()
            if tree:
                queue.extend(tree.items())
                x[..., list(tree.keys())] *= x[..., node:node+1]

        return x

    def build_hierarchical_tree(self):
        return {0: {_: None for _ in range(1, self.nc+1)}}  # default non-hierarchical structure


#         CLASSES = [
#             'tumor', 'stromal', 'immune cell', 'other', 'apoptotic body', 
#             'non-mitotic tumor', 'mitotic tumor', 'myeloid cell', 'sTILs', 'fibroblast', 
#             'vascular endothelium', 'myoepithelium', 'muscle', 'red blood cell', 'macrophage', 
#             'neutrophil', 'eosinophil', 'lymphocyte nuclei', 'plasma cell', 'normal epithelium',
#         ]

#         return {
#             0: {
#                 1: {
#                     6: {}, 
#                     7: {},
#                 },
#                 2: {
#                     10: {},
#                     11: {},
#                     12: {},
#                     13: {},
#                 },
#                 3: {
#                     8: {
#                         15: {},
#                         16: {},
#                         17: {},
#                     },
#                     9: {
#                         18: {},
#                         19: {},
#                     },
#                 },
#                 4: {
#                     14: {},
#                     20: {},
#                 },
#                 5: {},
#             }
#         }


