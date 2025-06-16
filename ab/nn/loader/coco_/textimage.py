# File: coco.py
# Description: Dataloader for the COCO dataset, specifically refactored for
# text-to-image generation tasks within the LEMUR framework.

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np

# Albumentations is a powerful library for transformations.
# Ensure it is installed: pip install albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the framework's global constant for the data directory.
from ab.nn.util.Const import data_dir

# --- Configuration ---
# Constants are defined at the top for clarity and easy modification.
COCO_ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_IMG_URL_TEMPLATE = 'http://images.cocodataset.org/zips/{}2017.zip'
# Normalization constants to scale image tensors to the [-1, 1] range.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_DEV = (0.5, 0.5, 0.5)


class CocoTextToImageDataset(Dataset):
    """
    A PyTorch Dataset for the COCO dataset, adapted for Text-to-Image models.

    This class handles the automatic download and setup of the COCO dataset.
    For each image, it randomly selects one of its five official captions to
    serve as the text prompt, providing diverse (image, text) pairs for training.
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
        Retrieves an image and one of its corresponding captions, chosen randomly.
        """
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info['file_name']

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"[Dataloader WARN] Could not load image {img_path}. Skipping to next sample.")
            return self.__getitem__((index + 1) % len(self))

        # Get all captions for the image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns if 'caption' in ann]

        # Randomly select one caption to use as the text prompt
        text_prompt = random.choice(captions) if captions else "an image without a caption"

        if self.transform:
            # Albumentations pipelines expect a NumPy array as input.
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented['image']

        return image, text_prompt


def loader(transform_fn, task, **kwargs):
    """
    Factory function to create train and validation datasets for the COCO dataset.
    This is the main entry point used by the LEMUR framework.

    Args:
        transform_fn (function): A function passed by the framework that returns a
                                 composed transform pipeline.
        task (str): The task name from the collection config (e.g., 'text2image').
        **kwargs: Additional arguments from the framework's config.

    Returns:
        tuple: A tuple containing (metadata, performance_goal, train_dataset, val_dataset).
    """
    if 'text2image' not in task:
        raise ValueError(f"The task '{task}' is not a text-to-image task for this dataloader.")

    # The framework is responsible for creating the transform pipeline. We just
    # call the function it provides with the normalization stats our model expects.
    transform = transform_fn((NORM_MEAN, NORM_DEV))

    # Use the framework's global `data_dir` constant to locate the dataset
    path = os.path.join(data_dir, 'coco')

    print(f"[Dataloader] Creating train and validation datasets from: {path}")
    train_dataset = CocoTextToImageDataset(root=path, split='train', transform=transform)
    val_dataset = CocoTextToImageDataset(root=path, split='val', transform=transform)

    # The LEMUR framework expects this specific return signature.
    # For a text-to-image task, vocab_size and a performance goal are not needed.
    metadata = (None,)
    performance_goal = 0.0

    return metadata, performance_goal, train_dataset, val_dataset
