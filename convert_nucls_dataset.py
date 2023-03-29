import os
import json
import pickle
import argparse
import datetime
from utils_nucls import *

## train val dataset
TRAINVAL_FOLDER = "/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/pathology_image_data/NuCLS/corrected_single_rater"
# TEST_FOLDER = "./data/NuCLS/evaluation_dataset_p_truth"
TEST_IMAGE_FOLDER = "/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/pathology_image_data/NuCLS/evaluation_dataset_p_truth/rgbs"
# TEST_GT_FOLDER = "./danyun_gt/NuCLS/evaluation_p_truth_multi_rater/results/processed_gt/output"

## Don't use this annotation
# BLOOD_CELL_ANN_FILE = "/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/LC25000_classifier/danyun_gt/NuCLS/corrected_single_rater/results/only_red_blood_bbox.csv"
# BLOOD_CELLS = pd.read_csv(BLOOD_CELL_ANN_FILE, index_col=0)
# BLOOD_CELLS = [(x['file'], np.array([x['x0'], x['y0'], x['x1'], x['y1']]), 'blood_cell') for _, x in BLOOD_CELLS.iterrows()]
# meta[['x0', 'y0', 'x1', 'y1']] = meta['bbox'].str.split('\[\s*|\]|\s+', expand=True)[[1,2,3,4]]
# meta = meta.drop('bbox', axis=1)


COLORS = {
    "tumor": [0, 255, 0],
    "stromal": [255, 0, 0],
    "sTILs": [0, 0, 255],
    "other": [0, 148, 225],
    "unlabeled": [148, 148, 148],
}

CLASSES = [
    "tumor", "stromal", "sTILs", "other", 
    # "blood cell", "macrophage nuclei", "dead nuclei", "ductal epithelium", # "unlabeled",
]

CLASSES_TRANSFER_MAP = {
    "apoptotic_body": "unlabeled", "correction_apoptotic_body": "unlabeled", 
    "fibroblast": "stromal", "correction_fibroblast": "stromal", 
    "lymphocyte": "sTILs", "correction_lymphocyte": "sTILs", 
    "macrophage": "stromal", "correction_macrophage": "stromal", 
    "mitotic_figure": "tumor", "correction_mitotic_figure": "tumor", 
    "plasma_cell": "sTILs", "correction_plasma_cell": "sTILs", 
    "tumor": "tumor", "correction_tumor": "tumor", 
    "unlabeled": "unlabeled", "correction_unlabeled": "unlabeled", 

    "ductal_epithelium": "other", 
    "eosinophil": "sTILs", # "other",
    "myoepithelium": "other", # "stromal"
    "neutrophil": "sTILs", # "other", 
    "vascular_endothelium": "stromal",
    "blood_cell": "other", "blood": "other",

    "lymphocyte nuclei": "sTILs", "tumor nuclei": "tumor", "stroma nuclei": "stromal",
    "blood cell": "other", "macrophage nuclei": "stromal", "dead nuclei": "other", "ductal epithelium": "other",
}

TEXT_LABEL_MAP = {
    "apoptotic_body": "dead", "correction_apoptotic_body": "dead", "apoptotic body": "dead",
    "fibroblast": "stromal", "correction_fibroblast": "stromal", 
    "lymphocyte": "immune", "correction_lymphocyte": "immune", 
    "macrophage": "macrophage", "correction_macrophage": "macrophage", 
    "mitotic_figure": "tumor", "correction_mitotic_figure": "tumor", "mitotic figure": "tumor", 
    "plasma_cell": "immune", "correction_plasma_cell": "immune", "plasma cell": "immune",
    "tumor": "tumor", "correction_tumor": "tumor", 
    "unlabeled": "unlabeled", "correction_unlabeled": "unlabeled", 

    "ductal_epithelium": "other", # ignore this one, not in test
    "eosinophil": "immune", 
    "myoepithelium": "stromal", # ignore this one, not in test
    "neutrophil": "immune",
    "vascular_endothelium": "stromal", "vascular endothelium": "stromal",
    "blood_cell": "blood", "blood": "blood", "blood cell": "blood",

    "tumor nuclei": "tumor", 
    "stroma nuclei": "stromal",
    "lymphocyte nuclei": "immune", 
    "macrophage nuclei": "macrophage",
    "dead nuclei": "dead",
}

CLASSES = [
    "tumor", "stromal", "immune", "blood", "macrophage", "dead", "other",
]

## Transfer annotation text into labels (int), pytorch cross_entroy default: ignore_index = -100, set it with "unlabeled"
VAL_TO_LABEL = {k: CLASSES.index(v)+1 if v in CLASSES else -100 for k, v in CLASSES_TRANSFER_MAP.items()}
memo = []  # add some extra slots for danyun's testing gt.
for k, v in {**VAL_TO_LABEL, **{_: i+1 for i, _ in enumerate(CLASSES)}}.items():
    memo.append([' '.join(k.split('_')), v])
    memo.append(['_'.join(k.split(' ')), v])
VAL_TO_LABEL = dict(memo)

## Define color for each label
LABELS_COLOR = dict((class_id+1, np.array(COLORS[name])/255) 
                    for class_id, name in enumerate(CLASSES))
LABELS_COLOR[-100] = np.array(COLORS['unlabeled'])/255

## Define display text for each label
LABELS_TEXT = {**{i+1: _ for i, _ in enumerate(CLASSES)}, -100: "unlabeled"}

## Default dataset config
DEFAULT_DATASET_CONFIG = {
    'labels': CLASSES,
    'image_reader': skimage.io.imread, 
    'output_size': (480, 480),
    'mean': [0., 0., 0.],  # [0.5760583, 0.45498945, 0.57751981], 
    'std': [1., 1., 1.],  # [0.24343736, 0.24815826, 0.20826345],
    'val_to_label': VAL_TO_LABEL, 
    'labels_color': LABELS_COLOR,
    'labels_text': LABELS_TEXT,
    'color_aug': 'dodge',
    'min_area': 1e-4,
    'max_area': 4e-2,
    'min_h': 1e-2,
    'min_w': 1e-2,
}

EXCLUDE_SLIDE_IDS = [
    'TCGA-A1-A0SP-DX1', 'TCGA-A7-A0DA-DX1', 'TCGA-AR-A1AR-DX1',
    'TCGA-C8-A12V-DX1', 'TCGA-E2-A158-DX1',
]


def get_slide_id(image_id, source='trainval'):
    assert source in ['test', 'trainval'], "Source: {} is not a valid option!".format(source)
    if source == 'trainval':
        return image_id.split('_')[0]
    elif source == 'test':
        "TCGA-A1-A0SP-01Z-00-DX1"
        tmp = image_id.split('_')[1].split('-')
        return "-".join([tmp[0], tmp[1], tmp[2], tmp[5]])


def get_trainval_dataset(data_folder, fold, processors, dataset_config, data_filter_fn=None, display=False):
    if not isinstance(processors, dict):
        processors = {'train': processors, 'val': processors}
    
    trainval_image_folder = os.path.join(data_folder, "rgb")
    trainval_gt_folder = os.path.join(data_folder, "csv")
    trainval_split_folder = os.path.join(data_folder, "train_test_splits")
    
    # train/val slides ids
    train_slides = pd.read_csv(os.path.join(trainval_split_folder, "fold_{}_train.csv".format(fold)), index_col=0)
    val_slides = pd.read_csv(os.path.join(trainval_split_folder, "fold_{}_test.csv".format(fold)), index_col=0)
    
    # meta info
    meta_info = pd.read_csv(os.path.join(trainval_gt_folder, "ALL_FOV_LOCATIONS.csv"), index_col=0)
    train_meta = meta_info.loc[meta_info['fovname'].str.split('_').str[0].isin(train_slides['slide_name']),]
    val_meta = meta_info.loc[meta_info['fovname'].str.split('_').str[0].isin(val_slides['slide_name']),]
    
    train_dataset = TorchDataset(trainval_image_folder, trainval_gt_folder, train_meta, 
                                 processor=processors['train'], **dataset_config)
    val_dataset = TorchDataset(trainval_image_folder, trainval_gt_folder, val_meta, 
                               processor=processors['val'], **dataset_config)
    
    if data_filter_fn is not None:
        train_dataset.filter_images(data_filter_fn)
        val_dataset.filter_images(data_filter_fn)
    
    if display is not False:
        train_dataset.display(display)
        val_dataset.display(display)
    
    return train_dataset, val_dataset


def generate_image_info(image_id, file_name, image_size, 
                        date_captured=datetime.datetime.utcnow().isoformat(' '),
                        license_id=1, coco_url="", flickr_url=""):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        # "coco_url": coco_url,
        # "flickr_url": flickr_url
    }


def generate_annotation_info(ann=None, **kwargs):
    """ generate annotation info from bbox, label and mask. 
        {"id" : int, "image_id" : int, "category_id" : int, 
         "segmentation" : RLE or [polygon], "area" : float, 
         "bbox" : [x,y,width,height], "iscrowd" : 0 or 1}
    """
    res = {"id": None, "image_id": None, "category_id": None, 
           "segmentation": None, "area": None, 
           "bbox": None, "iscrowd": None}
    dtypes = {"id": int, "image_id": int, "category_id": int, 
              "area": float, "bbox": list}
    res.update(ann)
    res.update(kwargs)
    ## TODO: Add a type and size check here
    assert res["iscrowd"] in [0, 1]
    if res["iscrowd"] == 1:
        assert isinstance(res["segmentation"], dict)
        assert "counts" in res["segmentation"]
        assert "size" in res["segmentation"]
    else:
        assert isinstance(res["segmentation"], list)
        for x in res["segmentation"]:
            assert len(x) > 4
    for k in dtypes:
        assert isinstance(res[k], dtypes[k])
    
    return res

def coco_header(description, labels, **kwargs):
    dt = datetime.datetime.utcnow()
    default_info = {
        "description": description,
        "version": "0.1.0",
        "year": dt.year,
        "contributor": str(os.getuid()),
        "date_created": dt.isoformat(' ')
    }
    INFO = kwargs.setdefault('info', default_info)

    default_licenses = {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
    LICENSES = kwargs.setdefault('licenses', default_licenses)

    CATEGORIES = kwargs.setdefault('categories', None)
    if CATEGORIES is None:
        CATEGORIES = [{'id': idx, 'name': str(_), 'supercategory': 'default',} for idx, _ in enumerate(labels)]

    ## Basic Coco-Detection format
    res = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }
    return res
    

def convert_to_coco(dataset, description=None, root_path=os.curdir, output_file=None, display=False):
    res = coco_header(description, CLASSES)
    
    object_idx = 1
    for image_idx in range(len(dataset)):
        image, objects, kwargs = dataset.load_image_and_objects(image_idx)
        image_info = dataset.images[image_idx]
        image_id = image_info["image_id"]
        h, w = image.shape[0], image.shape[1]
        
        ## image_notes
        image_path = os.path.relpath(image_info["data"][0], root_path)
        image_path = image_info["data"][0] if root_path is None else root_path.format(image_id=image_id)
        image_notes = generate_image_info(image_idx, image_path, (h, w))
        res["images"].append(image_notes)

        ## annotations_notes
        for (x0, y0, x1, y1), label, mask in zip(objects['boxes'], objects['labels'], objects['masks']):
            # flip and clip bbox
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            x0, x1, y0, y1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
            if x0 >= x1 or y0 >= y1:
                continue
            
            ann = {
                "bbox": [x0, y0, x1-x0, y1-y0], 
                "category_id": int(label-1) if label > 0 else int(label), 
                "iscrowd": 0, 
                "area": float((x1-x0) * (y1-y0)),
            }
            if mask.sum():
                ann["area"] = float((mask > 0).sum())
                if ann["iscrowd"] == 0:
                    ann["segmentation"] = binary_mask_to_polygon(mask > 0, flatten=True, mode='yx')
                else:
                    ann["segmentation"] = binary_mask_to_rle(mask > 0, compress=False)
            else:
                ann["segmentation"] = []

            masks_notes = generate_annotation_info(ann, id=object_idx, image_id=image_idx)
            res["annotations"].append(masks_notes)
            object_idx += 1

#         if display:
#             display_coco_annotations(image_id, output_dir, masks_folder)

    ## Export annotations
    if output_file is not None:
        output_folder = os.path.dirname(output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(output_file, "w") as f:
            json.dump(res, f)
    
    return res


def display_yolo_annotations(image_id, data_folder, masks_folder=None):
    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")
    
    image_file = os.path.join(data_folder, "images", "{}.png".format(image_id))
    ann_file = os.path.join(data_folder, "labels", "{}.txt".format(image_id))
    
    image = skimage.io.imread(image_file)
    h_img, w_img = image.shape[0], image.shape[1]
    
    ann = pd.read_csv(ann_file, header=None, sep=' ').to_numpy()
    bboxes = ann[:, 1:]
    labels = ann[:, 0].astype(int)
    labels = np.where(labels>=0, labels+1, labels)
    bboxes = np.stack([
        (bboxes[:, 0] - bboxes[:, 2]/2) * w_img, 
        (bboxes[:, 1] - bboxes[:, 3]/2) * h_img,
        (bboxes[:, 0] + bboxes[:, 2]/2) * w_img, 
        (bboxes[:, 1] + bboxes[:, 3]/2) * h_img,
    ], axis=-1)
    
    if masks_folder is not None:
        mask_file = os.path.join(masks_folder, "{}.pkl".format(image_id))
        if os.path.exists(mask_file):
            masks = pickle.load(open(mask_file, 'rb'))
        else:
            masks = None
    else:
        masks = None
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 12))
    axes.imshow(image)
    overlay_detections(axes, bboxes, labels, masks, labels_color=LABELS_COLOR, labels_text=LABELS_TEXT)
    plt.show()


def convert_to_yolo(dataset, output_dir, masks_folder=None, display=False):
    images_folder = os.path.join(output_dir, "images")
    labels_folder = os.path.join(output_dir, "labels")
    
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    
    if masks_folder is not None and not os.path.exists(masks_folder):
        os.makedirs(masks_folder)
    
    res = []
    for idx in range(len(dataset)):
        image, objects, kwargs = dataset.load_image_and_objects(idx)
        image_info = dataset.images[idx]
        image_id = image_info['image_id']

        # export image
        dst_path = os.path.join(images_folder, "{}.png".format(image_id))
        skimage.io.imsave(dst_path, img_as('uint8')(image))

        # write annotation
        h, w = image.shape[0], image.shape[1]
        yolo_ann = []
        for (x0, y0, x1, y1), label in zip(objects['boxes'], objects['labels']):
            # h, w = obj['image_size']
            yolo_ann.append([(label-1) if label > 0 else label, (x0+x1)/2/w, (y0+y1)/2/h, (x1-x0)/w, (y1-y0)/h])
        
        ## write annotations
        with open(os.path.join(labels_folder, "{}.txt".format(image_id)), "w") as f:
            contents = ["{} {} {} {} {}".format(*_) for _ in yolo_ann]
            f.write("\n".join(contents))
        
        if masks_folder is not None:
            with open(os.path.join(masks_folder, "{}.pkl".format(image_id)), 'wb') as f:
                pickle.dump(objects['masks'], f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if display:
            display_yolo_annotations(image_id, output_dir, masks_folder)
        
    return None


def convert_to_detectron2(dataset, output_file=None, display=False):
    from detectron2.structures import BoxMode
    from detectron2.utils.visualizer import Visualizer
    
    records = []
    for image_idx in range(len(dataset)):
        image, objects, kwargs = dataset.load_image_and_objects(image_idx)
        image_info = dataset.images[image_idx]
        image_id = image_info["image_id"]
        image_path = image_info["data"][0]
        h, w = image.shape[0], image.shape[1]
        
        record = {
            "file_name": image_path,
            "image_id": image_idx,
            "height": h,
            "width": w,
            "annotations": [],
        }
        
        ## annotations_notes
        for (x0, y0, x1, y1), label, mask in zip(objects['boxes'], objects['labels'], objects['masks']):
            # flip and clip bbox
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            x0, x1, y0, y1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
            if x0 >= x1 or y0 >= y1:
                continue

            ann = {
                "bbox": [x0, y0, x1, y1], 
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(label-1) if label > 0 else int(label), 
                # category_id (int, required): an integer in the range [0, num_categories-1] 
                # representing the category label. The value num_categories is reserved to 
                # represent the “background” category, if applicable.
            }
            if mask.sum():
                ann["segmentation"] = binary_mask_to_polygon(mask > 0, flatten=True, mode='xy')
                # ann["segmentation"] = binary_mask_to_rle(mask > 0, compress=False)
            else:
                ann["segmentation"] = []
            
            record["annotations"].append(ann)
        
        records.append(record)

        if display:
            img = skimage.io.imread(d["file_name"])
            # color is wrong here, just check boxes and masks. refine in future.
            visualizer = Visualizer(img, metadata=LABELS_COLOR, scale=1.0)
            out = visualizer.draw_dataset_dict(d)
            plt.imshow(out.get_image())
            plt.show()

    if output_file is not None:
        output_folder = os.path.dirname(output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(output_file, 'wb') as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Convert and export dataset', add_help=False)
    parser.add_argument('--format', choices=['yolo', 'coco', 'detectron2'], default='yolo', type=str)
    parser.add_argument('--data_dir', default=TRAINVAL_FOLDER, type=str)
    parser.add_argument('--output_dir', default='none', type=str)
    parser.add_argument('--trainval_fold', default=1, type=int, help="trainval split fold id.")
    parser.add_argument('--masks_folder', default='none', help="Folder to save masks for yolo (very large).")

    args = parser.parse_args()

    if args.output_dir == 'none':
        args.output_dir = os.path.join("./data_{}".format(args.format))
    output_folder = os.path.join(args.output_dir, "fold_{}".format(args.trainval_fold))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataset_config = DEFAULT_DATASET_CONFIG
    train_dataset, val_dataset = get_trainval_dataset(
        args.data_dir, args.trainval_fold, None, dataset_config, 
        data_filter_fn=lambda _: get_slide_id(_['image_id']) in EXCLUDE_SLIDE_IDS,
        display=False,
    )

    if args.format == 'yolo':
        if args.masks_folder == 'none':
            args.masks_folder = None
        _ = convert_to_yolo(train_dataset, os.path.join(output_folder, "train"), 
                            masks_folder=args.masks_folder, display=False)
        _ = convert_to_yolo(val_dataset, os.path.join(output_folder, "val"), 
                            masks_folder=args.masks_folder, display=False)
    elif args.format == 'coco':
        _ = convert_to_coco(train_dataset, "nucls_fold_{}_train".format(args.trainval_fold),
                            root_path=TRAINVAL_FOLDER, 
                            output_file=os.path.join(output_folder, "train_annotation.json"), 
                            display=False)
        _ = convert_to_coco(val_dataset, "nucls_fold_{}_val".format(args.trainval_fold),
                            root_path=TRAINVAL_FOLDER, 
                            output_file=os.path.join(output_folder, "val_annotation.json"), 
                            display=False)
    elif args.format == 'detectron2':
        _ = convert_to_detectron2(train_dataset, output_file=os.path.join(output_folder, "train.pkl"), 
                                  display=False)
        _ = convert_to_detectron2(val_dataset, output_file=os.path.join(output_folder, "val.pkl"), 
                                  display=False)
