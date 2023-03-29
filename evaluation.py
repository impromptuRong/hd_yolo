import os
import os
import time
import pickle
import torch
import torchvision
import skimage.io
import numpy as np
import pandas as pd

from collections import OrderedDict
from matplotlib import pyplot as plt
from IPython.display import display, HTML
from torchvision.models.detection.roi_heads import paste_masks_in_image

from metayolo.models.yolo import Model, Deploy, Ensemble
from metayolo.models.utils_general import scale_coords
from metayolo.engines.torch_utils import collate_fn
from metayolo.engines.general import convert_yolo_weights, intersect_dicts, manipulate_header_label_order
from metayolo.datasets import display_image_and_target, update_size, overlay_detections

from val_nuclei import flatten_onehot_objects
from utils_nucls import evaluate_results_new


@torch.no_grad()
def build_model(model_path, ref_model=None, half=True, extra_configs={}):
    # build model, load weights, export models
    ckpt = torch.load(model_path, map_location='cpu')
    if isinstance(ckpt, OrderedDict):  # state_dict only
        csd = ckpt
        assert ref_model is not None, f"model cannot be None if only state_dict is given."
    else:
        ckpt_model = ckpt.get('ema', ckpt['model'])
        ref_model = ref_model or ckpt_model
        csd = ckpt_model.state_dict()
    csd = OrderedDict({k: v for k, v in csd.items() if 'anchor' not in k})

    model = Model(ref_model.cfg, ref_model.hyp, is_scripting=True)
    # csd = convert_yolo_weights(model, csd)
    # csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
    csd['headers.det.mask_indices'] = ref_model.state_dict()['headers.det.mask_indices']
    
#     csd['headers.det.seg_h.maskrcnn_preds.mask_fcn_logits.weight'][0] = csd['headers.det.seg_h.maskrcnn_preds.mask_fcn_logits.weight'][1]
#     csd['headers.det.seg_h.maskrcnn_preds.mask_fcn_logits.bias'][0] = csd['headers.det.seg_h.maskrcnn_preds.mask_fcn_logits.bias'][1]
    model.load_state_dict(csd, strict=False)  # load

    if extra_configs:
        # {'headers': {'det': {'label_map': [0, 1, 2, 6, 3, 4, 5], 'nms_params': nms_params}}}
        if 'headers' in extra_configs:
            for key, cfgs in extra_configs['headers'].items():
                if 'label_map' in cfgs:
                    model.headers[key] = manipulate_header_label_order(model.headers[key], cfgs['label_map'])
                    # print(model.headers[key].nc, model.headers[key].nc_masks, model.headers[key].mask_indices)
                if 'nms_params' in cfgs:
                    model.headers[key].nms_params = model.headers[key].get_nms_params(cfgs['nms_params'])
    
    model.eval()
    if half:
        model.half()

    model_script = torch.jit.script(Deploy(model))

    return model, model_script


@torch.no_grad()
def attempt_load_model(weights_path, ref_model=None, half=True, extra_configs={}):
    if isinstance(weights_path, str):
        return build_model(weights_path, ref_model=ref_model, half=half, extra_configs=extra_configs)
    else:
        model_list, model_script_list = [], []
        for model_path in weights_path:
            model, model_script = build_model(model_path, ref_model=ref_model, half=half, extra_configs=extra_configs)
            model_list.append(model)
            model_script_list.append(model_script)
        return Ensemble(model_list), Ensemble(model_script_list)


@torch.no_grad()
def inference_on_loader_yolov5(model, data_loader, device, 
                               input_size=640, compute_masks=True, 
                               meta_info=None, display=False, 
                               **kwargs):
    if device.type == 'cpu':  # half precision only supported on CUDA
        model.float()
    model.eval()
    model.to(device)

    results = []
    total_time, n_images = 0., 0.
    for images, targets in data_loader:
        # image_sizes = [img.shape[-2:] for img in images]
        images = torch.stack(images).to(device, next(model.parameters()).dtype, non_blocking=True)
        ori_size = images.shape[-2:]

        st = time.time()
        # inputs = torchvision.transforms.functional.resize(images, size=[input_size,])  # keep aspect ratio, min_size = input_size
        inputs = torch.nn.functional.interpolate(images, size=input_size, mode='bilinear', align_corners=False)
        # pred, outputs = model(inputs, compute_masks=False)
        # outputs = model(inputs, compute_masks=False)
        _, outputs = model(inputs, compute_masks=compute_masks)
        total_time += time.time()-st
        n_images += len(images)

        for img, output in zip(images, outputs):
            for task_id in output:
                o = output[task_id]
                o['boxes'] = scale_coords(input_size, o['boxes'], ori_size).round()  # keep aspect ratio
                # convert labels into non-onehot tensors
                if o['labels'].dim() == 2:
                    o = flatten_onehot_objects(o)
                
#                 change_label = ((o['boxes'][:,2] - o['boxes'][:,0]) * (o['boxes'][:,3] - o['boxes'][:,1]) < 400) & (o['labels'] == 3)
#                 o['labels'][change_label] = 6
                
                output[task_id] = {k: v.detach().cpu() for k, v in o.items()}

                if display and meta_info is not None and task_id in meta_info:
                    labels_color, labels_text = meta_info[task_id]['labels_color'], meta_info[task_id]['labels_text']
                    if 'masks' in o:
                        masks = paste_masks_in_image(o['masks'].float(), o['boxes'].float(), 
                                                     img.shape[-2:], padding=1).squeeze(1)
                        masks = masks.float().cpu().numpy()
                    else:
                        masks = None
                    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
                    axes[0].imshow(img.permute(1, 2, 0).float().cpu().numpy())
                    axes[1].imshow(img.permute(1, 2, 0).float().cpu().numpy())
                    # axes[1].imshow(np.zeros(img.shape[-2:]))
                    overlay_detections(
                        axes[1], bboxes=o['boxes'].float().cpu().numpy(), 
                        labels=o['labels'].float().cpu().numpy(), masks=masks,
                        labels_color=labels_color, labels_text=labels_text,
                    )
                    plt.show()
            results.append(output)

#         for o, image_size in zip(outputs, image_sizes):
#             if score_threshold != model.nms_params['conf_thres']:
#                 keep = o['scores'] >= score_threshold
#                 o = {k: v[keep] for k, v in o.items()}
            
#             keep = torchvision.ops.nms(o['boxes'], o['scores'], iou_threshold=iou_threshold)
#             o = {k: v[keep].detach().cpu() for k, v in o.items()}
#             o['boxes'] = detect.scale_coords(images.shape[-2:], o['boxes'], image_size).round()
#             o['labels'] = o['labels'] + 1
            # results.append(o)
    
    return results, total_time/n_images


def run(test_dataset, ref_model=None, run_eval=True, **args):
    exp_name = args['exp_name']
    weights = args['weights']
    output_folder = args['output_folder']
    export_folder = args['export_folder']

    input_size = args.get('inputs_size', 640)
    batch_size = args.get('batch_size', 32)
    half = args.get('half', True)  # state_dict are stored in float16
    iou_threshold = args.get('iou_threshold', 0.5)  # used for matching gt with pred, not nms. Don't change
    # assert iou_threshold == 0.5, f"Manually comment this line if you indeed want to use a different value!"
    device = torch.device(args.get('device', 'cpu'))
    transfer_cfgs = args.get('transfer_cfgs', {})  # put score thres, iou_thresh here
    export_paths = args.get('export_paths', {})
    compute_masks = args.get('compute_masks', True)
    meta_info = args.get('meta_info', None)

    core_labels = args.get('core_labels', )
    label_converter = args.get('label_converter', {})

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    ## stats summary file
    summary_file_name = os.path.join(output_folder, f"{exp_name}_{device.type}.pkl")

    if os.path.exists(summary_file_name):
        stats_summary = pickle.load(open(summary_file_name, 'rb'))
    else:
        stats_summary = {}

    ## result file
    res_file_name = os.path.join(output_folder, f"{exp_name}_{device.type}.pt")

    if os.path.exists(res_file_name):
        res_summary = torch.load(res_file_name)
    else:
        res_summary = {}

    for key, weights_path in weights.items():
        print("==============================")
        print(key)

        if key not in stats_summary:
            if key not in res_summary:
                model, model_script = attempt_load_model(
                    weights_path, ref_model, half=half, 
                    extra_configs=transfer_cfgs, 
                )

                if key in export_paths:
                    torch.save(model.state_dict(), f"{export_paths[key]}.weights.pt")
                    torch.save(model, f"{export_paths[key]}.model.pt")
                    torch.jit.save(model_script, f"{export_paths[key]}.torchscript.pt")

                ## run inference
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, num_workers=16, 
                    collate_fn=collate_fn, shuffle=False,
                )

                res_test, ave_time_test = inference_on_loader_yolov5(
                    model_script, test_loader, device=device, 
                    input_size=input_size, half=half, 
                    compute_masks=compute_masks,
                    meta_info=meta_info,
                    display=True, 
                )
                res_summary[key] = [res_test, ave_time_test]
                torch.save(res_summary, res_file_name)

            if run_eval:
                res_test, ave_time_test = res_summary[key]
                res_test_det = [r['det'] for r in res_test]
                cm_list, stats_list, sum_stats_1, sum_stats_2 = evaluate_results_new(
                    test_dataset, res_test_det, 
                    iou_threshold=iou_threshold,
                    core_labels=core_labels,
                    label_converter=label_converter,
                    meta_info=meta_info.get('det'),
                    display_results=False, # "./tmp_data"
                )
                stats_summary[key] = {"cm": sum_stats_1, "pr": sum_stats_2, "time_per_image": ave_time_test}

                with open(summary_file_name, "wb") as f:
                    pickle.dump(stats_summary, f)

    # Model Summary: 476 layers, 76164816 parameters, 0 gradients, 110.1 GFLOPs

    return res_summary, stats_summary


def organize_stats_summary(stats_summary, core_labels):    
    merge_table_1, merge_table_2 = {}, {}
    for exp_k, v in stats_summary.items():
        sum_stats_1, sum_stats_2, time_per_image = v['cm'], v['pr'], v['time_per_image']
        if sum_stats_1['accuracy'] > 0.5 and sum_stats_1['coverage'] > 0.5:
            print("=============================")
            print(exp_k)

            saved_entry_1 = {
                'coverage': sum_stats_1['coverage'],
                'accuracy_c': sum_stats_1['accuracy_c'],
                'accuracy': sum_stats_1['accuracy'],
                'precision': np.mean([sum_stats_1[('precision', k)] for k in core_labels]),
                'recall': np.mean([sum_stats_1[('recall', k)] for k in core_labels]),
                'f1': np.mean([sum_stats_1[('f1', k)] for k in core_labels]),
                'mcc': sum_stats_1['mcc'],
                'miou': sum_stats_1['miou'],
                'time': time_per_image,
            }
            print("coverage: {:.4f}, accuracy_c: {:.4f}, accuracy: {:.4f}".format(
                saved_entry_1['coverage'], saved_entry_1['accuracy_c'], saved_entry_1['accuracy']))
            print("m_precision: {:.4f}, m_recall: {:.4f}, m_f1: {:.4f}".format(
                saved_entry_1['precision'], saved_entry_1['recall'], saved_entry_1['f1']))
            print("mcc: {:.4f}, miou: {:.4f}, time: {:.4f}".format(
                saved_entry_1['mcc'], saved_entry_1['miou'], time_per_image))

            saved_entry_2 = {'precision': {}, 'recall': {}, 'f1': {}, 'mcc': {}}
            for idx, k in enumerate(core_labels):
                saved_entry_2['precision'][k] = sum_stats_1[('precision', k)]
                saved_entry_2['recall'][k] = sum_stats_1[('recall', k)]
                saved_entry_2['f1'][k] = sum_stats_1[('f1', k)]
                saved_entry_2['mcc'][k] = sum_stats_1[('mcc', k)]
                print("{}: precision={:.4f}, recall={:.4f}, f1={:.4f}, mcc={:.4f}".format(
                    k, sum_stats_1[('precision', k)], sum_stats_1[('recall', k)], 
                    sum_stats_1[('f1', k)], sum_stats_1[('mcc', k)]))
    #         for k in ['tumor', 'stromal', 'sTILs']:
    #             v = sum_stats_2[k]
    #             print("{}: precision={:4f}, recall={:4f}, f1={:4f}, miou={:4f}".format(
    #                 k, v['precision'], v['recall'], v['f1'], v['miou']))
            display(sum_stats_1['cm'])
            display(sum_stats_1['cm_core'])
    #     merge_table_1[exp_k[0].split('/')[-1].split('_')[0]] = saved_entry_1
    #     merge_table_2[exp_k[0].split('/')[-1].split('_')[0]] = saved_entry_2
        merge_table_1[exp_k] = saved_entry_1
        merge_table_2[exp_k] = saved_entry_2

    merge_table_2 = {
        exp_k: {(tag, k): v for tag, kvs in entry.items() for k, v in kvs.items()} 
        for exp_k, entry in merge_table_2.items()
    }
    
    return merge_table_1, merge_table_2

