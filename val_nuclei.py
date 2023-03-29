# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLO model 
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm

from metayolo import LOGGER
from metayolo.models.metrics import ConfusionMatrix, ap_per_class, summarize_precision_recall, compute_ap, APMeter
from metayolo.models.utils_general import (check_img_size, coco80_to_coco91_class, non_max_suppression, 
                                           scale_coords, xywh2xyxy, xyxy2xywh, box_iou)

from metayolo.engines.callbacks import Callbacks
from metayolo.engines.general import check_requirements, check_yaml, colorstr, increment_path, print_args
from metayolo.engines.plots import output_to_target, plot_images, plot_val_study
from metayolo.engines.torch_utils import select_device, to_device, time_sync

from metayolo.datasets import display_image_and_target, update_size, overlay_detections

import matplotlib.pyplot as plt
from torchvision.models.detection.roi_heads import paste_masks_in_image


def flatten_onehot_objects(x):
    assert x['labels'].dim() == 2, f"labels has shape: {x['labels'].shape}, need an one hot tensor."
    nbox, nc = x['labels'].shape[0], x['labels'].shape[1]
    keep = x['labels'].flatten() > 0.

    res = {k: v for k, v in x.items()}
    res['labels'] = torch.tile(torch.arange(nc, device=res['labels'].device), (nbox,))[keep] # start from 1
    res['labels'][res['labels'] == 0] = -100
    res['boxes'] = torch.repeat_interleave(res['boxes'], nc, 0)[keep]
    if 'scores' in res:
        res['scores'] = res['scores'].flatten()[keep]
    if 'masks' in res:
        res['masks'] = torch.repeat_interleave(res['masks'], nc, 0)[keep]

    return res


def summarize_stats(ap_meter, task_id, plots=False, **kwargs):
    """ Customize the function for own needs.
        Including logging information, printing, images here.
        Return value must be a ductionary and contain 'fitness' as evaluation score. 
    """
    stats = ap_meter.ap_per_class(iouv=torch.linspace(0.5, 0.95, 10), ignore=[-100, -1])
    names = ap_meter.labels_text

    s = ('%10s' * 2 + '%12s' * 5) % (task_id, 'Labels', 'P', 'R', 'F1', 'mAP@.5', 'mAP@.5:.95')
    LOGGER.info(s)

    # max F1 index
    idx = stats['f1'].mean(0).argmax()
    p, r, f1 = stats['p'][:, idx], stats['r'][:, idx], stats['f1'][:, idx]
    ap50, ap = stats['ap'][:, 0], stats['ap'].mean(1)  # AP@0.5, AP@0.5:0.95

#     # precision, recall, f1, map50, map
#     map50, map = ap50.mean(), ap.mean()
#     mp, mr, mf1 = p.mean(), r.mean(), f1.mean()
#     fitness = map50 * 0.1 + map * 0.9

    # ignore class others
    map50, map = ap50[:4].mean(), ap[:4].mean()
    mp, mr, mf1 = p[:4].mean(), r[:4].mean(), f1[:4].mean()
    # map50, map = ap50[:].mean(), ap[:].mean()
    # mp, mr, mf1 = p[:].mean(), r[:].mean(), f1[:].mean()
    fitness = map50 * 0.1 + map * 0.9

    pf = '%10s' + '%10i' + '%12.3g' * 5  # print format
    LOGGER.info(pf % ('all', sum(stats['counts']), mp, mr, mf1, map50, map))
    for i, c in enumerate(stats['labels']):
        # print(names, i, c)
        # print(stats['counts'][i], p[i], r[i], f1[i], ap50[i], ap[i])
        LOGGER.info(pf % (names[c], stats['counts'][i], p[i], r[i], f1[i], ap50[i], ap[i]))

    # Plots
    if plots:
        save_dir = kwargs['save_dir']
        # confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        names = {k: v for k, v in names.items() if k < 4}
        ap_meter.plot(x=stats, filenames=os.path.join(save_dir, f"{task_id}_{{}}"), names=names)
        # callbacks.run('on_val_end')  # only wandb

    return {'mp': mp, 'mr': mr, 'f1': mf1, 'map50': map50, 'map': map, 'fitness': fitness}


# def save_one_txt(predn, shape, file, save_conf=True):
#     # Save one txt result
#     gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
#     for *xyxy, conf, cls in predn.tolist():
#         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#         with open(file, 'a') as f:
#             f.write(('%g ' * len(line)).rstrip() % line + '\n')


@torch.no_grad()
def run(model, dataloader, meta_info, callbacks,
        batch_size=32, half=True,  # use FP16 half-precision inference
        verbose=False, save_txt=False, save_dir=Path(''), plots=True,
        epoch=0,
       ):
    # Initialize/load model and set device
    device = next(model.parameters()).device  # get model device, PyTorch model
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()
    model.eval()

    n_image, ap_meter_dict, confusion_matrix_dict = 0, {}, {}
    for task_id, header in model.headers.items():
        # header.nms_params = {**header.nms_params, 'conf_thres': conf_thres, 'iou_thres': iou_thres,}
        names = meta_info[task_id]['labels_text']
        confusion_matrix_dict[task_id] = ConfusionMatrix(nc=header.nc)
        ap_meter_dict[task_id] = APMeter(names)
    callbacks.run('on_val_start')  # empty

    dt = [0.0, 0.0, 0.0]
    pbar = enumerate(dataloader)
    if verbose:  # desc=s, 
        pbar = tqdm(pbar, total=len(dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    for batch_i, (imgs, targets) in pbar:
        callbacks.run('on_val_batch_start')  # empty
        t1 = time_sync()
        imgs = torch.stack(imgs).to(device, non_blocking=True)
        imgs = imgs.half() if half else imgs.float()
        targets = to_device(targets, device)
        nb, _, h, w = imgs.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        _, outputs = model(imgs, compute_masks=False)
        dt[1] += time_sync() - t2

        # Metrics
        for img_idx, (img, output, target) in enumerate(zip(imgs, outputs, targets)):
        # for output, target in zip(outputs, targets):
            n_image += 1
            for task_id in model.headers:
                labels_color, labels_text = meta_info[task_id]['labels_color'], meta_info[task_id]['labels_text']
                o, t = output[task_id], target['anns'][task_id][0]
                # convert o and t into non-onehot tensors
                if o['labels'].dim() == 2:
                    o = flatten_onehot_objects(o)
                if t['labels'].dim() == 2:
                    t = flatten_onehot_objects(t)
                ap_meter_dict[task_id].add(o, t, iou_type='boxes')
#                 if len(target['boxes']) and plots:
#                     confusion_matrix_dict['task_id'].process_batch(o, t)

                if plots and img_idx == 0:  # don't plot
                    if not os.path.exists(os.path.join(save_dir, f"images_{epoch}")):
                        os.makedirs(os.path.join(save_dir, f"images_{epoch}"))

                    t = {k: v.detach().cpu() for k, v in t.items()}
                    o = {k: v.detach().cpu() for k, v in o.items()}
                    if 'masks' in t:
                        t['masks'] = paste_masks_in_image(t['masks'][:,None,...].float(), t['boxes'].float(), 
                                                          img.shape[-2:], padding=1).squeeze(1)
                        # t['masks'] = torch.nn.functional.interpolate(t['masks'], size=640)
                    else:
                        t['masks'] = None
                    if 'masks' in o:
                        o['masks'] = paste_masks_in_image(o['masks'].float(), o['boxes'].float(), 
                                                          img.shape[-2:], padding=1).squeeze(1)
                    else:
                        o['masks'] = None

                    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
                    # axes[0].imshow(img.permute(1, 2, 0).numpy())
                    axes[0].imshow(np.zeros(img.shape[-2:]))
                    overlay_detections(
                        axes[0], bboxes=t['boxes'], labels=t['labels'], masks=t['masks'],
                        labels_color=labels_color, labels_text=labels_text,
                    )

                    # axes[1].imshow(img.permute(1, 2, 0).numpy())
                    axes[1].imshow(np.zeros(img.shape[-2:]))
                    overlay_detections(
                        axes[1], bboxes=o['boxes'], labels=o['labels'], masks=o['masks'],
                        labels_color=labels_color, labels_text=labels_text,
                    )
                    plt.savefig(os.path.join(save_dir, f"images_{epoch}", f"{batch_i}_{img_idx}.png"))
                    plt.close()

            # Save/log
            # if save_txt:
            #     save_one_txt(predn, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])  # only wandb

#         # Plot images, super slow with plt, don't use for now
#         if plots and batch_i < 3:
#             f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
#             Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
#             f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
#             Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

        callbacks.run('on_val_batch_end')  # empty

    # speeds per image
    t = tuple(_ / n_image * 1E3 for _ in dt)

    val_stats = {task_id: summarize_stats(ap_meter, task_id=task_id, plots=plots, save_dir=save_dir)
                 for task_id, ap_meter in ap_meter_dict.items()}
    fi = sum(stats['fitness'] * 1.0 for task_id, stats in val_stats.items())

    model.float()  # for training

    return fi, val_stats, t

