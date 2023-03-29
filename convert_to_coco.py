import os
import json
import pickle
import argparse
import datetime
from utils_nucls import *

## train val dataset
TRAINVAL_FOLDER = "./data/NuCLS/corrected_single_rater"
TEST_FOLDER = "./data/NuCLS/evaluation_dataset_p_truth"
COCO_OUTPUT_DIR = "./data_coco"

COLORS = {
    "tumor nuclei": [0, 255, 0],
    "stroma nuclei": [255, 0, 0],
    "lymphocyte nuclei": [0, 0, 255],
    "blood cell": [255, 0, 255],
    "macrophage nuclei": [255, 255, 0],
    "dead nuclei": [0, 148, 225],
    "ductal epithelium": [100, 0, 255],
    "unlabeled": [148, 148, 148],
}

CLASSES = [
    "tumor nuclei", "stroma nuclei", "lymphocyte nuclei", 
    "macrophage nuclei", "dead nuclei", "ductal epithelium", "blood cell", # "unlabeled",
]

CLASSES_TRANSFER_MAP = {
    "apoptotic_body": "dead nuclei", "correction_apoptotic_body": "dead nuclei", 
    "fibroblast": "stroma nuclei", "correction_fibroblast": "stroma nuclei", 
    "lymphocyte": "lymphocyte nuclei", "correction_lymphocyte": "lymphocyte nuclei", 
    "macrophage": "macrophage nuclei", "correction_macrophage": "macrophage nuclei", 
    "mitotic_figure": "tumor nuclei", "correction_mitotic_figure": "tumor nuclei", 
    "plasma_cell": "lymphocyte nuclei", "correction_plasma_cell": "lymphocyte nuclei", 
    "tumor": "tumor nuclei", "correction_tumor": "tumor nuclei", 
    "unlabeled": "unlabeled", "correction_unlabeled": "unlabeled", 
    
    "ductal_epithelium": "ductal epithelium", 
    "eosinophil": "lymphocyte nuclei", 
    "myoepithelium": "stroma nuclei", 
    "neutrophil": "lymphocyte nuclei",
    "vascular_endothelium": "stroma nuclei",
}

# pytorch ignore_index = -100
VAL_TO_LABEL = {k: CLASSES.index(v)+1 if v in CLASSES else -100 for k, v in CLASSES_TRANSFER_MAP.items()}
LABELS_COLOR = dict((class_id+1, np.array(COLORS[name])/255) 
                    for class_id, name in enumerate(CLASSES))
LABELS_COLOR[-100] = np.array(COLORS['unlabeled'])/255
LABELS_TEXT = {**{i+1: _ for i, _ in enumerate(CLASSES)}, -100: "unlabeled"}

## classes and colors
DEFAULT_DATASET_CONFIG = {
    'labels': CLASSES,
    'image_reader': skimage.io.imread, 
    'output_size': (512, 512),
    'mean': [0., 0., 0.], 
    'std': [1., 1., 1.], 
    'val_to_label': VAL_TO_LABEL, 
    'labels_color': LABELS_COLOR,
    'labels_text': LABELS_TEXT,
    'color_aug': 'dodge',
    'min_area': 1e-4,
    'max_area': 4e-2,
    'min_h': 1e-2,
    'min_w': 1e-2,
}


def get_trainval_dataset(data_folder, fold, processors, dataset_config, display=False):
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
    

def export_to_coco(dataset, output_file, description=None, root_path=os.curdir, display=False):
    output_folder = os.path.dirname(output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

        ## write annotations
    with open(output_file, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('convert to coco', add_help=False)
    parser.add_argument('--data_dir', default=TRAINVAL_FOLDER, type=str)
    parser.add_argument('--output_dir', default=COCO_OUTPUT_DIR, type=str)
    parser.add_argument('--trainval_fold', default=1, type=int, help="trainval split fold id.")
    
    args = parser.parse_args()
    coco_output_folder = os.path.join(args.output_dir, "fold_{}".format(args.trainval_fold))
    if not os.path.exists(coco_output_folder):
        os.makedirs(coco_output_folder)
    
    dataset_config = DEFAULT_DATASET_CONFIG
    train_dataset, val_dataset = get_trainval_dataset(
        args.data_dir, args.trainval_fold, None, dataset_config, display=False,
    )
    
    export_to_coco(train_dataset, os.path.join(coco_output_folder, "train_annotation.json"), 
                   description="nucls_fold_{}_train".format(args.trainval_fold),
                   root_path=TRAINVAL_FOLDER, display=False)
    export_to_coco(val_dataset, os.path.join(coco_output_folder, "val_annotation.json"), 
                   description="nucls_fold_{}_val".format(args.trainval_fold),
                   root_path=TRAINVAL_FOLDER, display=False)

