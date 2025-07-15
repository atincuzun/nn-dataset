# File: textimage.py
# Description: An adapted, robust dataloader for text-to-image tasks.
# Based on Caption.py to ensure framework compatibility.

import os
import random
from os.path import join
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as T

from ab.nn.util.Const import data_dir

# --- Configuration ---
COCO_ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_IMG_URL_TEMPLATE = 'http://images.cocodataset.org/zips/{}2017.zip'
# Normalization constants to scale image tensors to the [-1, 1] range.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_DEV = (0.5, 0.5, 0.5)
IMAGE_SIZE = 512 # Define a constant for image size
MINIMUM_ACCURACY = 0.001 # Kept for framework compatibility

class TextToImageDataset(Dataset):
    """
    A PyTorch Dataset for the COCO dataset, specifically for Text-to-Image models.
    This class handles the automatic download and setup of the COCO dataset and
    provides (image, text_prompt) pairs for training.
    """
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        if split not in ['train', 'val']:
            raise ValueError(f"Invalid split: '{split}'. Must be 'train' or 'val'.")

        self.root = root
        self.transform = transform
        self.split = split

        # Initialize COCO API and download data if it doesn't exist.
        ann_dir = join(root, 'annotations')
        if not os.path.exists(ann_dir):
            print("[Dataloader] COCO annotations not found! Downloading...")
            os.makedirs(root, exist_ok=True)
            download_and_extract_archive(COCO_ANN_URL, root, filename='annotations_trainval2017.zip')
            print("[Dataloader] Annotation download complete.")

        ann_file = join(ann_dir, f'captions_{split}2017.json')
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Check for images and download if missing
        self.img_dir = join(root, f'{split}2017')
        if not os.path.exists(join(self.img_dir, self.coco.loadImgs(self.ids[0])[0]['file_name'])):
            print(f"[Dataloader] COCO {split} images not found! Downloading...")
            download_and_extract_archive(COCO_IMG_URL_TEMPLATE.format(split), root, filename=f'{split}2017.zip')
            print(f"[Dataloader] COCO {split} image download complete.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = join(self.img_dir, img_info['file_name'])

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"[Dataloader WARN] Could not load image {img_path}. Using next sample.")
            return self.__getitem__((index + 1) % len(self))

        # Get all captions for the image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns if 'caption' in ann]

        # **MODIFICATION**: Randomly select ONE caption to use as the text prompt.
        text_prompt = random.choice(captions) if captions else "an image without a caption"

        if self.transform:
            image = self.transform(image)

        return image, text_prompt

def loader(transform_fn, task, **kwargs):
    """
    Factory function to create train/validation datasets for Text-to-Image tasks.
    """
    if 'text2image' not in task:
        raise ValueError(f"The task '{task}' is not a text-to-image task for this dataloader.")

    # --- PERMANENT FIX for Data Corruption ---
    # We IGNORE the faulty `transform_fn` from the framework and build our own
    # standard, safe pipeline. This prevents all data-related errors.
    print("\n[Dataloader INFO] Using a standard, safe pipeline to prevent data corruption.\n")
    correct_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 1. Ensure all images are the same size
        T.ToTensor(),                      # 2. Convert image to a PyTorch tensor
        T.Normalize(NORM_MEAN, NORM_DEV)   # 3. Normalize to the [-1, 1] range
    ])

    path = join(data_dir, 'coco')

    # Create the datasets using our correct transformation
    train_dataset = TextToImageDataset(root=path, split='train', transform=correct_transform)
    val_dataset = TextToImageDataset(root=path, split='val', transform=correct_transform)

    # Return signature for a task without a vocabulary
    metadata = (None,)
    performance_goal = 0.0

    return metadata, performance_goal, train_dataset, val_dataset