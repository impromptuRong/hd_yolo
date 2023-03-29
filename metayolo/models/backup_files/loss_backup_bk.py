# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Dict, Optional, Union

from .utils_general import bbox_iou


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class WeightReduceLoss(nn.Module):
    def __init__(self, loss_fn, weight: Optional[Tensor]=None, reduction: str='mean'):
        """ loss_fn should output elementwise results (reduction = 'none'). """
        super().__init__()
        self.loss_fn = loss_fn
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = self.loss_fn(input, target)
        if self.weight is not None:
            loss *= self.weight

        return self.reduce_loss(loss)

    def reduce_loss(self, loss):
        """Reduce loss as specified. "none", "mean" and "sum"."""
        reduction_enum = F._Reduction.get_enum(self.reduction)
        # none: 0, elementwise_mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class DetLoss(nn.Module):
    # Compute losses
    def __init__(self, det, hyp={}, autobalance=False):
        """ Rewrite loss function. 
        hyp: Dict[str, float], (cls_pw can be a list, but doesnt support jit in this case)
        slots in hyp:
            box (default 0.05): box loss gain
            cls (default 0.5): cls loss gain
            cls_pw (default 1.0): cls BCELoss positive_weight, 
                cls_pw > 1 increase recall, < 1 increase precision. Directly multiply to logits.
                FocalLoss use alpha to balance pos vs neg, can turn this to 1.0.
            cls_cw (default 1.0): cls BCELoss class_weights,
                apply class weight on poor classified class.
            obj (default 1.0): obj loss gain (scale with pixels)
            obj_pw (default 1.0): obj BCELoss positive_weight
            iou_t (default 0.20): IoU training threshold
            anchor_t (default 4.0): anchor-multiple threshold
            # anchors: 3  # anchors per output layer (0 to ignore)
            fl_gamma (default 0.0): focal loss gamma (efficientDet default gamma=1.5)
            label_smoothing (default 0.0): positive, negative BCE targets
        class_weights:
            give weights to different class, larget weight will focus on specific class.
            different from cls_pw.
        """
        super().__init__()
        self.gr, self.autobalance, self.sort_obj_iou = 1.0, autobalance, False
        self.hyp = self.get_hyp_params(hyp)

        # Define criteria
        cls_weight = torch.tensor(hyp['cls_cw'])
        cls_weight = cls_weight/cls_weight.sum() * (cls_weight > 0).sum()
        BCEcls = WeightReduceLoss(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyp['cls_pw']), reduction='none'), cls_weight)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyp['obj_pw']))
        if hyp['fl_gamma'] > 0.:  # Focal loss
            BCEcls = FocalLoss(BCEcls, hyp['fl_gamma'])
            BCEobj = FocalLoss(BCEobj, hyp['fl_gamma'])
        self.BCEcls, self.BCEobj = BCEcls, BCEobj

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=hyp['label_smoothing'])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def get_hyp_params(self, args={}):
        default_args = {'box': 0.05, 'cls': 0.05, 'obj': 1.0,
                        'cls_pw': 1.0, 'obj_pw': 1.0, 'cls_cw': 1.0, 'fl_gamma': 0.0, 
                        'iou_t': 0.20, 'anchor_t': 4.0, 'label_smoothing': 0.0}
        return {k: args.get(k, v) for k, v in default_args.items()}

    def forward(self, p, targets):  # predictions, targets, targets['labels'] start from 1
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets, tcls start from 1

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    keep = tcls[i] > 0   # filter out negative sample
                    if keep.any():  # BCEWithLogitsLoss gives nan for 0 objects
                        logits = pcls[keep]
                        t = torch.full_like(logits, self.cn, device=device)  # targets
                        t[range(keep.sum()), tcls[i][keep]-1] = self.cp  # label start from 1
                        lcls += self.BCEcls(logits, t).mean()  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, {'box': lbox.detach(), 'obj': lobj.detach(), 'cls': lcls.detach()}

    def build_targets(self, p, targets):
        device = targets.device
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                           ], device=device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i].to(device)
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask_logits, boxes, gt_masks, gt_labels, mask_patches=None):
        # type: (Tensor(b, nc, h_m, w_m), Tensor(b, 4), Tensor(b, h, w), Tensor(b,)) -> Tensor
        # project masks on rois, gt_labels start from 1
        bs, nc, h, w = mask_logits.shape
        mask_targets = torchvision.ops.roi_align(gt_masks, boxes.split(1), (h, w), 1.0)
        if nc > 1:  # multi masks
            index = torch.arange(gt_labels.shape[0], device=gt_labels.device)
            mask_logits = mask_logits[index, gt_labels-1][:, None]  # gt_labels start from 1

        # trim empty mask_targets:
        keep = (mask_targets.sum(dim=[1, 2, 3]) > 0) & (gt_labels < 4) # ignore blood, macrophage, dead mask
        mask_targets = mask_targets[keep]
        mask_logits = mask_logits[keep]

        # if mask_patches is not None:
        #     print("======================")
        #     import matplotlib.pyplot as plt
        #     mask_patches = mask_patches[keep]
        #     print(mask_targets.shape, mask_logits.shape, mask_patches.shape)
        #     fig, axes = plt.subplots(4, 6, figsize=(24, 16))
        #     ax = axes.ravel()
        #     for i in range(min(len(ax)//3, len(mask_targets))):
        #         ax[i*3].imshow(mask_patches[i].permute(1, 2, 0).cpu().detach().numpy())
        #         ax[i*3+1].imshow(mask_targets[i][0].cpu().detach().numpy())
        #         ax[i*3+2].imshow(mask_logits.sigmoid()[i][0].cpu().detach().numpy())
        #     plt.show()
        #     print("======================")

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            # print("No masks found!")
            return mask_logits.sum() * 0

        # return 1-mask_iou(mask_logits.sigmoid(), mask_targets, factor=0.0, eps=0.).mean()
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_targets)
        
        return mask_loss[None]  # consistent with det_loss

