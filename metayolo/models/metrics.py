# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from .utils_general import box_iou


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            nc, nn = self.nc, len(names)  # number of classes, names
            sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
            labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array,
                           annot=nc < 30,
                           annot_kws={
                               "size": 8},
                           cmap='Blues',
                           fmt='.2f',
                           square=True,
                           vmin=0.0,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------
def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


################################################
################################################
class APMeter(object):
    def __init__(self, labels_text={}):
        self.reset()
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.labels_text = labels_text

    def reset(self):
        """Resets the meter with empty member variables"""
        self.n_pred, self.n_true, self.n_match= 0, 0, 0

        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.y_pred = torch.LongTensor(torch.LongStorage())
        self.y_true = torch.LongTensor(torch.LongStorage())

        self.ious = torch.FloatTensor(torch.FloatStorage())
        self.m_pred = torch.LongTensor(torch.LongStorage())
        self.m_true = torch.LongTensor(torch.LongStorage())
        # self.weights = torch.FloatTensor(torch.FloatStorage())

    def add(self, output, target, iou_type='boxes'):
        o_scores, order = torch.sort(output['scores'], descending=True)
        o_labels = output['labels'][order]

        if iou_type == 'masks' and ('masks' in output and 'masks' in target): # use mask iou
            ious = get_mask_ious(output['masks'][order], target['masks'])
        else: # use box iou
            ious = box_iou(output['boxes'][order], target['boxes'])
            # ious = torchvision.ops.box_iou(output['boxes'][order], target['boxes'])
        n_pred, n_true = ious.shape[0], ious.shape[1]

        # get matched items, update m_true, m_pred and n_match, order by iou
        pred_idx, true_idx = torch.where(ious >= self.iouv.min())
        ious, order = torch.sort(ious[pred_idx, true_idx], descending=True)
        pred_idx, true_idx, n_match = pred_idx[order], true_idx[order], len(ious)

        self.m_pred.resize_(self.n_match + n_match)
        self.m_pred.narrow(0, self.n_match, n_match).copy_(pred_idx + self.n_pred)
        self.m_true.resize_(self.n_match + n_match)
        self.m_true.narrow(0, self.n_match, n_match).copy_(true_idx + self.n_true)
        self.ious.resize_(self.n_match + n_match)
        self.ious.narrow(0, self.n_match, n_match).copy_(ious)
        self.n_match += n_match

        # update y_true and n_true
        self.y_true.resize_(self.n_true + n_true)
        self.y_true.narrow(0, self.n_true, n_true).copy_(target['labels'])
        self.n_true += n_true

        # update y_pred, n_pred and scores
        self.y_pred.resize_(self.n_pred + n_pred)
        self.y_pred.narrow(0, self.n_pred, n_pred).copy_(o_labels)
        self.scores.resize_(self.n_pred + n_pred)
        self.scores.narrow(0, self.n_pred, n_pred).copy_(o_scores)
        self.n_pred += n_pred

    def ap_per_class(self, iouv=None, ignore=[-100, -1], eps=1e-16):
        """ Compute the average precision, given the recall and precision curves. """
        if iouv is None:
            iouv = self.iouv  # iou vector for mAP@0.5:0.95

        # asign 1 vs 1 match
        matches = torch.stack([self.m_true, self.m_pred, self.ious], -1)
        if ignore:  # remove labels in ignore
            ignored = np.isin(self.y_true[self.m_true], ignore) | np.isin(self.y_pred[self.m_pred], ignore)
            if ignored.any():
                matches = matches[~ignored]
        else:
            ignored = np.array([])
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # unique m_pred
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # unique m_true
        matches = matches[self.y_true[matches[:, 0].long()] == self.y_pred[matches[:, 1].long()]]
        pred_matches = torch.zeros((self.n_pred, len(iouv)), dtype=bool)
        pred_matches[matches[:, 1].long()] = matches[..., -1:] >= iouv

        # When calculate pr, we remove all preds matched with ignored gt from y_pred, scores, y_label
        # In case a predicted tumor match a gt unclassified, include this in pr will reduce performance
        if ignored.any():
            # exclude_true = np.setdiff1d(self.m_true[ignored], matches[:, 0])
            mask = torch.ones(self.n_pred, dtype=bool)
            mask[np.setdiff1d(self.m_pred[ignored], matches[:, 1])] = False
            pred_matches = pred_matches[mask]
            pred_scores = self.scores[mask]
            pred_labels = self.y_pred[mask]
        else:
            pred_matches = pred_matches
            pred_scores = self.scores
            pred_labels = self.y_pred

        # sort by scores
        pred_scores, order = torch.sort(pred_scores, descending=True)
        pred_matches, pred_labels = pred_matches[order], pred_labels[order]

        # Find unique classes, number of classes in y_true
        # classes, counts = np.unique(self.y_true, return_counts=True)
        # unique_classes = {c: n for c, n in zip(classes, counts) if c not in ignore}  # Counter(self.y_true)
        # nc = len(unique_classes)

        # Create Precision-Recall curve and compute AP for each class
        # px, py = np.linspace(0, 1, 1000), []  # for plotting
        # ap, p, r = np.zeros((nc, len(iouv))), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        px, py, counts, labels = np.linspace(0, 1, 1000), [], [], []
        ap, p, r, f1 = [], [], [], []

        # for idx, (c, n_l) in enumerate(unique_classes.items()):
        for c, n_true in zip(*np.unique(self.y_true, return_counts=True)):
            if c in ignore:
                continue
            # names[idx] = self.labels_text[c]
            keep = pred_labels == c
            labels.append(c)
            counts.append(n_true)

            if keep.sum() == 0 or n_true == 0:
                ap.append(np.zeros(len(iouv)))
                r.append(np.zeros(len(px)))
                p.append(np.zeros(len(px)))
                py.append(np.zeros(len(px)))
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (~pred_matches[keep]).cumsum(0)
                tpc = pred_matches[keep].cumsum(0)

                # Recall
                recall = tpc / (n_true + eps)  # recall curve
                r.append(np.interp(-px, -pred_scores[keep], recall[:, 0], left=0))  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p.append(np.interp(-px, -pred_scores[keep], precision[:, 0], left=1))  # p at pr_score

                # AP from recall-precision curve
                ap.append(np.zeros(len(iouv)))
                for j in range(pred_matches.shape[1]):
                    ap[-1][j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                    if j == 0:
                        py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

        stats = {'labels': labels, 'counts': counts, 'px': px, 'py': np.stack(py), 
                 'ap': np.stack(ap), 'p': np.stack(p), 'r': np.stack(r),}

        # Compute F1 (harmonic mean of precision and recall)
        stats['f1'] = 2 * stats['p'] * stats['r'] / (stats['p'] + stats['r'] + eps)

        return stats

    def plot(self, x, filenames=f'./{{}}', names=None):
        names = names or self.labels_text
        names, idx = zip(*[(v, x['labels'].index(k)) for k, v in names.items() if k in x['labels']])
        names, idx = list(names), list(idx)

        # names = [v for k, v in names.items() if k in x['labels']]  # list: only classes that have data
        # names = {i: v for i, v in enumerate(names)}  # to dict
        # print(x['px']), len(x['py']), len(x['f1']), len(x['ap']), len(x['p']), len(x['r']))
        plot_pr_curve(x['px'], x['py'][idx], x['ap'][idx], filenames.format('PR_curve.png'), names)
        plot_mc_curve(x['px'], x['f1'][idx], filenames.format('F1_curve.png'), names, ylabel='F1')
        plot_mc_curve(x['px'], x['p'][idx], filenames.format('P_curve.png'), names, ylabel='Precision')
        plot_mc_curve(x['px'], x['r'][idx], filenames.format('R_curve.png'), names, ylabel='Recall')


def evaluate_detection(target, output, classes, iou_threshold=0.5, iou_type='boxes'):
    # get ious: n_target * n_preds
    if iou_type=='masks' and ('masks' in output and 'masks' in target): # use mask iou
        ious = get_mask_ious(target['masks'], output['masks'])
    else: # use box iou
        ious = torchvision.ops.box_iou(target['boxes'], output['boxes'])
    n_true, n_pred = ious.shape[0], ious.shape[1]
    true_label, pred_label = target['labels'], output['labels']

    # Overall recall: cover gt with prediction
    if n_true > 0 and n_pred > 0:
        matched_ious, matched_idxs = ious.max(1)
        pred_label_r = pred_label[matched_idxs]
        pred_label_r[matched_ious < iou_threshold] = -1
    else:
        matched_ious = torch.zeros_like(true_label) * 1.0
        pred_label_r = -torch.ones_like(true_label)
    recall = {'y_true': true_label, 'y_pred': pred_label_r, 'ious': matched_ious}

    # Overall precision: cover prediction with gt
    if n_true > 0 and n_pred > 0:
        matched_ious, matched_idxs = ious.max(0)
        true_label_p = true_label[matched_idxs]
        true_label_p[matched_ious < iou_threshold] = -1
    else:
        matched_ious = torch.zeros_like(pred_label) * 1.0
        true_label_p = -torch.ones_like(pred_label)
    precision = {'y_true': true_label_p, 'y_pred': pred_label, 'ious': matched_ious}

    ## per_class summary with best bipartite match, includes precision and recall
    stats_per_class = {}
    for c in classes:
        t_idx, o_idx = true_label == c, pred_label == c
        n1, n2 = t_idx.sum().item(), o_idx.sum().item()
        matched_ious, m_iou = [], 0.
        if n1 > 0 and n2 > 0:
            ious_c = ious[t_idx][:,o_idx]
            matched_ious, matched_idxs = ious_c.max(1)
            keep = matched_ious >= iou_threshold
            matched_ious, matched_idxs = matched_ious[keep], matched_idxs[keep]
            if len(matched_idxs.unique()) != len(matched_idxs):
                print(f"check uniqueness: {len(matched_idxs.unique())}, {len(matched_idxs)}")
            m_iou = matched_ious.mean().item() if len(matched_ious) else 0.0
        
        stats_per_class[c] = [len(matched_ious), n1, n2, m_iou]

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
    
    return precision, recall, stats_per_class


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
    cm = pd.DataFrame(cm, index=labels, columns=labels).drop('other', axis=0).drop("missing", axis=0)
    coverage = 1 - cm['missing'].values.sum()/cm.values.sum()
    
    cm_core = cm.drop("unlabeled", axis=0).drop("unlabeled", axis=1)
    K = len(np.diag(cm_core))
    accuracy = np.diag(cm_core.values).sum()/cm_core.values.sum()
    accuracy_c = np.diag(cm_core.values).sum()/cm_core.values[:K, :K].sum()  # ignore missing accuracy
    
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
