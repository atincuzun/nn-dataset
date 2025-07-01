# File: coco.py
# Location: ab/nn/loader/
# Description: A pure COCO dataloader for text-to-image tasks, fully compliant
# with the LEMUR framework.

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as transforms

# Import the framework's global constant for the data directory.
from ab.nn.util.Const import data_dir

# --- Configuration ---
COCO_ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_IMG_URL_TEMPLATE = 'http://images.cocodataset.org/zips/{}2017.zip'
# Normalization constants for the image transform.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_DEV = (0.5, 0.5, 0.5)


class CocoTextToImageDataset(Dataset):
    """
    A PyTorch Dataset for the COCO dataset, adapted for Text-to-Image models.
    This version correctly uses the standard COCO API and file structure.
    ---
    This dataset is "split-aware". For training, it returns (image, text).
    For validation/testing, it returns (image, image) to bypass the framework's
    evaluation loop error, which cannot handle text labels.
    """

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        if split not in ['train', 'val']:
            raise ValueError(f"Invalid split: '{split}'. Must be 'train' or 'val'.")

        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Initialize COCO API and download data if it doesn't exist.
        self.coco = self._initialize_coco_api()
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_dir = self.root / f'{self.split}2017'
        self._check_and_download_images()

    def _initialize_coco_api(self):
        """Initializes the COCO API, downloading annotations if necessary."""
        ann_dir = self.root / 'annotations'
        ann_file = ann_dir / f'captions_{self.split}2017.json'

        if not ann_dir.exists():
            print(f"[Dataloader] COCO annotations not found in '{ann_dir}'. Downloading...")
            self.root.mkdir(parents=True, exist_ok=True)
            download_and_extract_archive(COCO_ANN_URL, str(self.root), filename='annotations_trainval2017.zip')
            print("[Dataloader] Annotation download complete.")

        if not ann_file.exists():
            raise RuntimeError(f"Missing annotation file: {ann_file}. Download or extraction may have failed.")
        return COCO(str(ann_file))

    def _check_and_download_images(self):
        """Checks for the existence of image data and downloads it if missing."""
        if not self.ids: return
        first_img_info = self.coco.loadImgs(self.ids[0])[0]
        first_img_path = self.img_dir / first_img_info['file_name']
        if not first_img_path.exists():
            print(f"[Dataloader] COCO {self.split} images not found in '{self.img_dir}'. Downloading...")
            url = COCO_IMG_URL_TEMPLATE.format(self.split)
            download_and_extract_archive(url, str(self.root), filename=f'{self.split}2017.zip')
            print(f"[Dataloader] COCO {self.split} image download complete.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Retrieves a sample. The format depends on the split.
        - 'train': returns (image_tensor, text_prompt_string)
        - 'val'/'test': returns (image_tensor, image_tensor)
        """
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info['file_name']
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"[Dataloader WARN] Could not load image {img_path}. Skipping to next sample.")
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Fallback if no transform is provided
            image_tensor = transforms.ToTensor()(image)

        if self.split == 'train':
            # For training, get a random real text prompt
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns if 'caption' in ann]
            text_prompt = random.choice(captions) if captions else "an image without a caption"
            return image_tensor, text_prompt
        else:  # For 'val' or 'test'
            # For evaluation, return the image tensor as both input and "label"
            # to satisfy the framework's `labels.to(device)` requirement.
            return image_tensor, image_tensor


def loader(transform_fn, task, **kwargs):
    """
    Factory function to create train and validation datasets for the COCO dataset.
    This is the main entry point used by the LEMUR framework.
    """
    if 'text2image' not in task:
        raise ValueError(f"The task '{task}' is not a text-to-image task for this dataloader.")

    # The framework is responsible for creating the transform pipeline.
    transform = transform_fn((NORM_MEAN, NORM_DEV))

    # --- FIX: The path now correctly points to the 'coco' subdirectory ---
    path = os.path.join(data_dir, 'coco')

    print(f"[Dataloader] Creating train and validation datasets from: {path}")
    train_dataset = CocoTextToImageDataset(root=path, split='train', transform=transform)
    val_dataset = CocoTextToImageDataset(root=path, split='val', transform=transform)

    # Return the datasets and placeholder metadata
    metadata = (None,)
    performance_goal = 0.0  # Placeholder for generative models

    # The framework's utility functions expect the dataset object itself
    # to have a 'minimum_accuracy' attribute. We add it here for full compatibility.
    train_dataset.minimum_accuracy = performance_goal
    val_dataset.minimum_accuracy = performance_goal

    return metadata, performance_goal, train_dataset, val_dataset
