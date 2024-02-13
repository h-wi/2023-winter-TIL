import os.path as osp

import mmcv
import random

from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress

import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re

def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info

def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))

    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin

    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann

def convert_facemask_to_coco(ann_paths, out_file):
    annotations = []
    images = []
    categories = []

    obj_count = 0 #bounding box id

    # ready for object labeling
    labels_str = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    labels_ids = [0, 1, 2]
    label2id = dict(zip(labels_str, labels_ids))

    for path in tqdm(ann_paths):
        # Read annotation xml
        ann_tree = ET.parse(path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=True)
        img_id = img_info['id']
        images.append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': obj_count})
            annotations.append(ann)
            obj_count = obj_count + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        categories.append(category_info)

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories
    )
    dump(coco_format_json, out_file)

def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '') -> List[str]:

    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split(" ")

    ann_ids.pop()

    ann_ids = [i+ext_with_dot for i in ann_ids]
    ann_paths = [os.path.join(ann_dir_path, aid) for aid in ann_ids]
    return ann_paths

if __name__ == '__main__':
    train_size = 753
    val_size = 100

    split = random.sample(range(0, train_size + val_size), train_size + val_size)

    train_ids = split[:train_size]
    val_ids = split[train_size:]

    with open("train_ann_ids.txt", "w") as f:
        for i in train_ids:
            f.write("maksssksksss" + str(i) + " ")

    with open("val_ann_ids.txt", "w") as f:
        for i in val_ids:
            f.write("maksssksksss" + str(i) + " ")

    train_paths = get_annpaths(ann_dir_path='./annotations', ann_ids_path='train_ann_ids.txt', ext='xml')
    val_paths = get_annpaths(ann_dir_path='./annotations', ann_ids_path='val_ann_ids.txt', ext='xml')

    convert_facemask_to_coco(ann_paths=train_paths,
                                out_file='./annotations/train/annotation_coco_mask.json')

    convert_facemask_to_coco(ann_paths=val_paths,
                                out_file=f'./annotations/val/annotation_coco_mask.json')

