# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/atincuzun/Desktop/nn-dataset/ab/nn/loader/coco_detection.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 2024-12-20 08:58:21 UTC (1734685101)

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import os
import requests
from torchvision.datasets.utils import download_and_extract_archive
from torch.nn.utils.rnn import pad_sequence

coco_ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
coco_img_url = 'http://images.cocodataset.org/zips/{}2017.zip'
MIN_CLASS_LIST = list(range(91))
MIN_CLASS_N = len(MIN_CLASS_LIST)

def class_n():
    return MIN_CLASS_N

def class_list():
    return MIN_CLASS_LIST

def collate_fn_test(batch):
    images = []
    boxes = []
    labels = []
    max_objects = 0
    for img, target in batch:
        images.append(img)
        max_objects = max(max_objects, len(target['boxes']))
    for img, target in batch:
        num_objects = len(target['boxes'])
        padded_boxes = torch.zeros((max_objects, 4))
        padded_labels = torch.zeros(max_objects)
        if num_objects > 0:
            padded_boxes[:num_objects] = target['boxes']
            padded_labels[:num_objects] = target['labels']
        boxes.append(padded_boxes)
        labels.append(padded_labels)
    images = torch.stack(images)
    boxes = torch.stack(boxes)
    labels = torch.stack(labels)
    targets = torch.cat([boxes, labels.unsqueeze(-1)], dim=2)
    return images, targets

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, 0)
    return images, targets

class COCODetectionDataset(Dataset):
    def __init__(self, root, split='train', transform=None, class_list=None):
        """
        Initialize COCO detection dataset
        
        Parameters:
        -----------
        root : str
            Path to COCO dataset root directory
        split : str
            'train' or 'val'
        transform : callable, optional
            Transform to apply to images and targets
        class_list : list, optional
            List of class IDs to use (for subset of classes)
        """
        valid_splits = ['train', 'val']
        if split not in valid_splits:
            raise ValueError(f'Invalid split: {split}')
        self.root = root
        self.transform = transform
        self.class_list = class_list or MIN_CLASS_LIST
        ann_file = os.path.join(root, 'annotations', f'instances_{split}2017.json')
        if not os.path.exists(os.path.join(root, 'annotations')):
            print('Annotation file doesn\'t exist! Downloading')
            os.makedirs(root, exist_ok=True)
            download_and_extract_archive(coco_ann_url, root, filename='annotations_trainval2017.zip')
            print('Annotation file preparation complete')
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_dir = os.path.join(root, f'{split}2017')
        first_image_info = self.coco.loadImgs(self.ids[0])[0]
        first_file_path = os.path.join(self.img_dir, first_image_info['file_name'])
        if not os.path.exists(first_file_path):
            print(f'Image dataset doesn\'t exist! Downloading {split} split...')
            download_and_extract_archive(coco_img_url.format(split), root, filename=f'{split}2017.zip')
            print('Image dataset preparation complete')
        self.collate_fn = self.collate_batch

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        file_path = os.path.join(self.img_dir, img_info['file_name'])
        try:
            image = Image.open(file_path).convert('RGB')
        except:
            if not hasattr(self, 'no_missing_img'):
                print('Failed to read image(s). Download will be performed as needed.')
                self.no_missing_img = True
            response = requests.get(img_info['coco_url'])
            if response.status_code == 200:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                image = Image.open(file_path).convert('RGB')
            else:
                raise RuntimeError(f"Failed to download image: {img_info['file_name']}")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            cat_id = ann['category_id']
            if cat_id not in self.class_list:
                continue
            cat_id = self.class_list.index(cat_id)
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(cat_id)
            areas.append(ann['area'])
            iscrowd.append(0)
        target = {}
        if boxes:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['area'] = torch.as_tensor(areas, dtype=torch.float32)
            target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)
            if self.transform is not None:
                image, target = self.transform(image, target)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
            if self.transform is not None:
                image = self.transform(image)
        target['image_id'] = torch.tensor([img_id])
        target['orig_size'] = torch.as_tensor([img_info['height'], img_info['width']])
        return image, target

    @staticmethod
    def collate_batch(batch):
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        images = torch.stack(images, 0)
        return images, targets

    def __len__(self):
        return len(self.ids)

def loader(path='./data/cocos', transform=None, class_list=None, **kwargs):
    """
    Main entry point following repository pattern.
    Returns train and validation datasets for COCO object detection.
    
    Parameters:
    -----------
    path : str
        Path to COCO dataset root directory
    transform : callable, optional
        Transform to apply to images and targets
    class_list : list, optional
        List of class IDs to use (for subset of classes)
    **kwargs : dict
        Additional arguments passed to dataset
    
    Returns:
    --------
    tuple: (train_dataset, val_dataset)
    """
    train_dataset = COCODetectionDataset(root=path, split='train', transform=transform, class_list=class_list)
    val_dataset = COCODetectionDataset(root=path, split='val', transform=transform, class_list=class_list)
    return (1, train_dataset, val_dataset)
