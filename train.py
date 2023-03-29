# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.

Models and datasets download automatically from the latest YOLOv5 release.
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
    $ python path/to/train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm.auto import tqdm
from collections import OrderedDict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
ROOT = Path('./')

import val_nuclei as val  # for end-of-epoch mAP
from metayolo import LOGGER

from metayolo.common import attempt_load, ModelEMA, de_parallel
from metayolo.datasets import create_dataloader, load_dataset_info, display_image_and_target, update_size
from metayolo.loggers import Loggers

from metayolo.models.yolo import Model
from metayolo.models.utils_torch import EarlyStopping
from metayolo.models.utils_general import check_img_size, labels_to_class_weights, labels_to_image_weights

from metayolo.engines.autoanchor import check_anchors
from metayolo.engines.autobatch import check_train_batch_size
from metayolo.engines.callbacks import Callbacks
from metayolo.engines.downloads import attempt_download
from metayolo.engines.plots import plot_evolve, plot_labels
from metayolo.engines.torch_utils import select_device, torch_distributed_zero_first, collate_fn, to_device
from metayolo.engines.general import (check_file, check_git_status, check_requirements, check_suffix, 
                                      check_yaml, colorstr, get_latest_run, increment_path, init_seeds, 
                                      methods, fitness, one_cycle, print_args, print_mutation, strip_optimizer,
                                      intersect_dicts, convert_yolo_weights, manipulate_header_label_order)
# from models.utils.loggers.wandb.wandb_utils import check_wandb_resume


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def rescale_train_batch(imgs, tgts, size=None):
    imgs = torch.nn.functional.interpolate(imgs, size=size, mode='bilinear', align_corners=False)
    for tgt in tgts:
        tgt['size'] = update_size(tgt['size'], size)
        for k, v in tgt['anns'].items():
            for _ in v:
                _['size'] = update_size(_['size'], size)

    return imgs, tgts


def fitness_fn(stats):
    return stats['map50'] * 0.1 + stats['map'] * 0.9


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')  # empty logger.on_pretrain_routine_start

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            hyp['label_smoothing'] = opt.label_smoothing
            hyp['keep_res'] = opt.keep_res
            hyp['patch_size'] = opt.patch_size
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    if not evolve:
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or load_dataset_info(data)  # check if None
    train_path, val_path, meta_info = data_dict['train'], data_dict['val'], data_dict['meta_info']
    # nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    # names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    # assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    # is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        if isinstance(ckpt, OrderedDict):  # state_dict only
            ckpt = {'model': ckpt, 'ema': None, 'epoch': -1, 'optimizer': None, 'updates': None, 'wandb_id': None, }
            csd = ckpt['model']  # checkpoint state_dict as FP32
            # csd = OrderedDict({k: v.float() for k, v in ckpt['model'].items()})  # checkpoint state_dict as FP32
            model_cfg = cfg
        else:
            model_cfg = cfg or ckpt['model'].yaml
            # only keep (ema) weights, clean up all others if force hard restart
            if opt.restart:
                if 'epoch' in ckpt:
                    ckpt['epoch'] = -1
                if 'optimizer' in ckpt:
                    ckpt['optimizer'] = None
                if 'updates' in ckpt:
                    ckpt['updates'] = None
                if 'wandb_id' in ckpt:
                    ckpt['wandb_id'] = None
                if 'ema' in ckpt and ckpt['ema'] is not None:
                    ckpt['model'] = ckpt['ema']
                    ckpt['ema'] = None
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(model_cfg, hyp=hyp, ch=3, anchors=hyp.get('anchors'))
        exclude = ['anchor'] if (model_cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        try:
            csd = convert_yolo_weights(model, csd)
        except:
            pass
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        ## REMOVE THIS, only in breast_cancer_det convert mask run
        # label_map = [0, 1, 2, 6, 3, 4, 5]
        # model.headers['det'] = manipulate_header_label_order(model.headers['det'], label_map, convert_masks=False)
    else:
        model = Model(cfg, hyp=hyp, ch=3, anchors=hyp.get('anchors'))  # create

    model = model.to(device)
    model = model.freeze(freeze)  # freeze parameters

#     freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
#     for k, v in model.named_parameters():
#         v.requires_grad = True  # train all layers
#         if any(x in k for x in freeze):
#             LOGGER.info(f'freezing {k}')
#             v.requires_grad = False

    # Image size
    gs = [32]
    for k, header in model.headers.items():  # grid size (max stride)
        if hasattr(header, 'stride'):
            gs.append(int(header.stride.max()))
        else:
            gs.append(max([int(buffer.stride) for buffer in header.anchors]))
    gs = max(gs)
    imgsz = hyp['img_size'] = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
    
    param_groups = [
        {'params': g0},
        {'params': g1, 'weight_decay': hyp['weight_decay']},
        {'params': g2},
    ]

    if opt.optimizer == 'Adam':
        optimizer = Adam(param_groups, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(param_groups, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(param_groups, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    # optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, train_dataset = create_dataloader(
        train_path, batch_size // WORLD_SIZE, hyp=hyp, augment=True, cache=True, rank=LOCAL_RANK, 
        workers=workers, image_weights=False, prefix=colorstr('train: '), shuffle=True,
    )

    # mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    # assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader, val_dataset = create_dataloader(
            val_path, batch_size // WORLD_SIZE * 2, hyp=hyp, augment=False, cache=True, 
            rank=-1, workers=workers * 2, prefix=colorstr('val: '), shuffle=False,
        )

        ## display dataset
        if not os.path.exists('display_dataset'):
            os.makedirs('display_dataset')
#         for idx, (image, target) in enumerate(train_dataset, 1):
#             display_image_and_target(image, target, meta_info, plot=f'./display_dataset/train_{idx:04}.png')
        for idx, (image, target) in enumerate(val_dataset, 1):
            display_image_and_target(image, target, meta_info, plot=f'./display_dataset/val_{idx:04}.png')
        
        if not resume:
#             labels = np.concatenate(dataset.labels, 0)
#             if plots:
#                 plot_labels(labels, names, save_dir)
            # Anchors
            if not opt.noautoanchor:
                for task_id, header in model.headers.items():
                    box_sizes = [_['boxes'][:, [2,3]]-_['boxes'][:, [0,1]] 
                                 for idx, _ in enumerate(train_dataset.ann_cache) 
                                 if train_dataset.annotations[idx]['task_id'] == task_id]
                    wh = torch.cat(box_sizes).float()
                    wh = wh[(wh > 10).all(-1)]  # remove box smaller than 10x10
                    # m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
                    # thr = hyp['anchor_t']
                    # check_anchors(wh, m=header, thr=header.det_loss.hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end')  # not used when opt.wandb=False

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)

    # Model attributes
    header_info = {}
    for task_id, header in de_parallel(model).headers.items():
        nc = header.nc  # number of classes
        nl = header.nl  # number of detection layers (to scale hyps)
        header_hyp = header.det_loss.hyp
        header.det_loss.hyp = header.det_loss.get_hyp_params({
            **header_hyp, 
            'box': header_hyp['box'] * 3 / nl,  # scale to layers
            'cls': header_hyp['cls'] * nc / 80 * 3 / nl,  # scale to classes and layers
            'obj': header_hyp['obj'] * (imgsz / 640) ** 2 * 3 / nl,  # scale to image size and layers
        })
        header_info[task_id] = {'nc': nc, 'nl': nl, 'nc_masks': header.nc_masks}

    # model.nc = nc  # attach number of classes to model
    # model.hyp = hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    callbacks.run('on_train_start')  # pass
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    if RANK in (-1, 0):
        best_fitness, stats, _ = val.run(
            model=ema.ema, dataloader=val_loader, 
            meta_info=meta_info, callbacks=callbacks,
            batch_size=batch_size // WORLD_SIZE * 2,
            save_dir=save_dir, plots=plots,
            verbose=opt.verbose,
        )

    # c0 = [a * 12 + k for a in range(3) for k in range(12) if k!=8 and k!=10]
    # input_csd = {k: v.detach().clone() for k, v in model.module.headers.det.m.state_dict().items()}
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')  # empty
        model.train()

        ## Check bn status
#         if RANK in (-1, 0):
#             print(f"==========Epoch: {epoch}==========")
#             print(model.module.backbone[0].bn.weight)
#             # print(model.backbone.0.bn.bias)
#             print(model.module.backbone[0].bn.running_mean)
#             print(f"==================================")

#         ## Check headers.det.m status
#         if RANK in (-1, 0):
#             print(f"==========Epoch: {epoch}==========")
#             tag = True
#             current_csd = model.module.headers.det.m.state_dict()
#             for k, v in input_csd.items():
#                 for idx in c0:
#                     if not torch.allclose(current_csd[k][idx], v[idx], atol=1e-6):
#                         tag = False
#                         print(f"========{idx}=======")
#                         print(f"{current_csd[k][idx]}")
#                         print(f"{v[idx]}")
#             assert tag
#             print(f"==================================")

        # Update image weights (optional, single-GPU only)
#         if opt.image_weights:
#             cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
#             iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
#             print(f'class weights {model.class_weights}, image weights {iw}')
#             dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = {}  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        
        logger_items = [f'{task_id}/{s}' for task_id in header_info for s in ['box', 'obj', 'cls', 'mask']]
        LOGGER.info(('\n' + '%10s%10s' + '%12s' * len(logger_items)) % ('Epoch', 'gpu_mem', *logger_items))
        
        pbar = enumerate(train_loader)
        if RANK in (-1, 0) and opt.verbose:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

        optimizer.zero_grad()
        for i, (imgs, targets) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')  # empty
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = torch.stack(imgs).to(device, non_blocking=True)
            targets = to_device(targets, device)

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale > 0.:
                sz = random.randrange(imgsz * (1-opt.multi_scale), imgsz * (1+opt.multi_scale) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs, targets = rescale_train_batch(imgs, targets, size=ns)

            # Forward
            with amp.autocast(enabled=cuda):
                # DDP doesn't allow compute_masks = 0.
                losses, _ = model(imgs, targets, compute_masks=True)  # forward, loss scaled by batch_size
                loss, loss_items = 0., {}
                for task_id, task_losses in losses.items():
                    loss += task_losses['det_loss']
                    if opt.masks and 'mask_loss' in task_losses:
                        loss += task_losses['mask_loss'] * 1.0
                    for k, v in task_losses['loss_items'].items():
                        loss_items[f'{task_id}/{k}'] = v.item()

                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
#                 if opt.quad:
#                     loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in (-1, 0):
                mloss = {k: (mloss.get(k, 0.) * i + v) / (i + 1) for k, v in loss_items.items()}  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                display_info = ('%10s%10s' + '%12.4g' * len(mloss)) % (f'{epoch + 1}/{epochs}', mem, *list(mloss.values()))
                if opt.verbose:
                    pbar.set_description(display_info)
                callbacks.run('on_train_batch_end', ni, model, imgs.cpu(), to_device(targets, 'cpu'), 
                              meta_info, False, opt.sync_bn)  # saveplots with plt is super slow, fix in future
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = {g: x['lr'] for g, x in zip(['lr/bn', 'lr/weights', 'lr/bias'], optimizer.param_groups)}  # for loggers
        scheduler.step()

        if RANK in (-1, 0):
            if opt.verbose == 0:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                display_info = ('%10s%10s' + '%12.4g' * len(mloss)) % (f'{epoch + 1}/{epochs}', mem, *list(mloss.values()))
                LOGGER.info(display_info)

            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch + 1)  # only used for opt.wandb
            # 'yaml', 'nc', 'stride', 'class_weights' not in attr, it belongs to header
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            log_vals = {**mloss, **lr,}
            if not noval or final_epoch:  # Calculate mAP
                fi, stats, _ = val.run(
                    model=ema.ema, dataloader=val_loader, 
                    meta_info=meta_info, callbacks=callbacks,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    save_dir=save_dir, plots=True,
                    verbose=opt.verbose, epoch=epoch + 1,
                )
                for task_id, scores in stats.items():
                    for k, v in scores.items():
                        log_vals[f'{task_id}/{k}'] = v

                # Update best mAP, {'mp': mp, 'mr': mr, 'f1': f1, 'map50': map50, 'map': map}
                best_fitness, is_best = (fi, True) if fi > best_fitness else (best_fitness, False)
            callbacks.run('on_fit_epoch_end', log_vals, epoch + 1, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch + 1,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'date': datetime.now().isoformat()
                }
                # Save last, best and delete
                torch.save(ckpt, last)
                if is_best:
                    torch.save(ckpt, best)
                if (opt.save_period > 0) and ((epoch+1) % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch + 1}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch + 1, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in (-1, 0):
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                # strip_optimizer(f)  # strip optimizers
                if f is best:
                    ckpt = torch.load(f)
                    LOGGER.info(f'\nValidating {f} (epoch={ckpt["epoch"]}, fitness={ckpt["best_fitness"]})...')
                    fi, stats, _ = val.run(
                        model=ckpt['ema'], dataloader=val_loader, 
                        meta_info=meta_info, callbacks=callbacks,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        save_dir=save_dir, plots=plots,
                        verbose=opt.verbose,
                    )
#                     if is_coco:
#                         callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, plots, epoch + 1, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()

    return results


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
#     parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
#     parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', type=float, default=0.0, help='vary img-size +/- multi-scale%%')
#     parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=str, default=[], help='Freeze layers: `backbone`, `header.det.m`, etc')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    # Extra arguments added by Ruichen
    parser.add_argument('--restart', action='store_true', help='Entirely reinitialize training pretrain.')
    parser.add_argument('--verbose', default=0, help='1: tqdm bar, 0: epoch stats only.')
    parser.add_argument('--keep_res', type=float, default=0.0, 
                        help='Fix resolution by padding/croping without resize when > 0.0.')
    parser.add_argument('--patch_size', type=int, default=0, 
                        help='Patch size random collected from raw image when keep resolution.')
    parser.add_argument('--masks', action='store_true', help='Train mask header.')
    
    return parser


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in (-1, 0):
        print_args(vars(opt))
        check_git_status()
        check_requirements(exclude=['thop'])

    # Resume
    if opt.resume and not opt.evolve:  # resume an interrupted run
    # if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        # assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    parser = argument_parser()
    opt = parser.parse_known_args()[0]

    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    parser = argument_parser()
    opt = parser.parse_args()
    main(opt)
