import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import os
import requests
import tqdm
from torchvision.datasets.utils import download_and_extract_archive

from torch.nn.utils.rnn import pad_sequence
# Use the same URLs as cocos.py
coco_ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
coco_img_url = "http://images.cocodataset.org/zips/{}2017.zip"

def collate_fn(batch):
    images = []
    boxes = []
    labels = []
    max_objects = 0
    
    # First pass to get max objects for padding
    for img, target in batch:
        images.append(img)
        max_objects = max(max_objects, len(target['boxes']))
    
    # Second pass to pad boxes and labels
    for img, target in batch:
        num_objects = len(target['boxes'])
        padded_boxes = torch.zeros((max_objects, 4))
        padded_labels = torch.zeros(max_objects)
        
        if num_objects > 0:
            padded_boxes[:num_objects] = target['boxes']
            padded_labels[:num_objects] = target['labels']
            
        boxes.append(padded_boxes)
        labels.append(padded_labels)
    
    # Stack everything into batches
    images = torch.stack(images)  # (batch_size, C, H, W)
    boxes = torch.stack(boxes)    # (batch_size, max_objects, 4)
    labels = torch.stack(labels)  # (batch_size, max_objects)
    
    # Combine boxes and labels into single tensor
    targets = torch.cat([boxes, labels.unsqueeze(-1)], dim=2)  # (batch_size, max_objects, 5)
    
    return images, targets


class COCODetectionDataset(Dataset):
    def __init__(self, root, split="train", transform=None, class_list=None):
        """Initialize COCO detection dataset
        
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
        valid_splits = ["train", "val"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}")
        
        self.root = root
        self.transform = transform
        self.class_list = class_list
        
        # Setup annotation file path
        ann_file = os.path.join(root, "annotations", f"instances_{split}2017.json")
        
        # Download annotations if they don't exist
        if not os.path.exists(os.path.join(root, "annotations")):
            print("Annotation file doesn't exist! Downloading")
            os.makedirs(root, exist_ok=True)
            download_and_extract_archive(coco_ann_url, root, filename="annotations_trainval2017.zip")
            print("Annotation file preparation complete")
        
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Set image directory and check/download images
        self.img_dir = os.path.join(root, f"{split}2017")
        # Test first image to see if dataset exists
        first_image_info = self.coco.loadImgs(self.ids[0])[0]
        first_file_path = os.path.join(self.img_dir, first_image_info['file_name'])
        if not os.path.exists(first_file_path):
            print(f"Image dataset doesn't exist! Downloading {split} split...")
            download_and_extract_archive(
                coco_img_url.format(split), 
                root, 
                filename=f"{split}2017.zip"
            )
            print("Image dataset preparation complete")
        self.collate_fn = collate_fn

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        file_path = os.path.join(self.img_dir, img_info['file_name'])
        try:
            image = Image.open(file_path).convert('RGB')
        except:
            if not hasattr(self, 'no_missing_img'):
                print("Failed to read image(s). Download will be performed as needed.")
                self.no_missing_img = True
            response = requests.get(img_info["coco_url"])
            if response.status_code == 200:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                image = Image.open(file_path).convert('RGB')
            else:
                raise RuntimeError(f"Failed to download image: {img_info['file_name']}")
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Prepare target
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue  # Skip invalid boxes

            boxes.append([x, y, x + w, y + h])
            
            cat_id = ann['category_id']
            if self.class_list is not None:
                if cat_id not in self.class_list:
                    continue
                cat_id = self.class_list.index(cat_id)
            labels.append(cat_id)
            
            areas.append(ann['area'])
            iscrowd.append(0)

        # Convert to tensors with proper handling of empty annotations
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,))
        areas = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,))
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,))

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([img_info['height'], img_info['width']])
        }

        if self.transform is not None:
            image, target = self.transform(image, target)
        # Post-transform validation
        if len(target['boxes']) > 0:  # Check if there are any boxes
            # Get valid box mask
            boxes = target['boxes']
            valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            
            # Only keep valid boxes and corresponding labels/areas
            target['boxes'] = boxes[valid_boxes]
            target['labels'] = target['labels'][valid_boxes]
            target['area'] = target['area'][valid_boxes]
            target['iscrowd'] = target['iscrowd'][valid_boxes]

        return image, target

    def __len__(self):
        return len(self.ids)

def loader(path="./data/cocos", transform=None, class_list=None, **kwargs):
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
    train_dataset = COCODetectionDataset(
        root=path,
        split="train",
        transform=transform,
        class_list=class_list
    )
    
    val_dataset = COCODetectionDataset(
        root=path,
        split="val",
        transform=transform,
        class_list=class_list
    )



    return train_dataset, val_dataset
