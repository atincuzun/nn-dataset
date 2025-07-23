# File: celeba_hq_loader.py
# Description: A memory-efficient dataloader for the Captioned CelebA-HQ dataset,
#              fully compatible with the Lemur framework.

import os
import json
from os.path import join
from PIL import Image

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as T

from ab.nn.util.Const import data_dir

# --- Configuration ---
# Public URL for a captioned version of the CelebA-HQ dataset
DATASET_URL = 'https://github.com/huggingface/datasets/raw/main/data/celeb_a_hq/celeb_a_hq.zip'
CAPTIONS_URL = 'https://raw.githubusercontent.com/Fair-Implicit-Generative-Models-for-Text-to-Image-Synthesis/Fair-Implicit-Generative-Models-for-Text-to-Image-Synthesis.github.io/main/celeba_caption.json'

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_DEV = (0.5, 0.5, 0.5)
IMAGE_SIZE = 256  # CelebA-HQ images are high quality


class CelebAHQTextToImageDataset(Dataset):
    """
    A PyTorch Dataset for the Captioned CelebA-HQ dataset.
    """

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_dir = join(self.root, 'images')
        self.caption_file = join(self.root, 'captions.json')

        # --- Download and Setup ---
        if not os.path.exists(self.image_dir):
            print("[Dataloader] CelebA-HQ images not found! Downloading (~1.3 GB)...")
            download_and_extract_archive(DATASET_URL, self.root, filename='celeb_a_hq.zip')
            os.rename(join(self.root, 'CelebA-HQ-img'), self.image_dir)  # Rename for consistency
            print("[Dataloader] Image download complete.")

        if not os.path.exists(self.caption_file):
            print("[Dataloader] CelebA-HQ captions not found! Downloading...")
            # Note: torchvision's download utils don't work well for single files,
            # so we use a simple alternative.
            import urllib.request
            urllib.request.urlretrieve(CAPTIONS_URL, self.caption_file)
            print("[Dataloader] Caption download complete.")

        # --- Load Captions ---
        with open(self.caption_file, 'r') as f:
            caption_data = json.load(f)

        # Create a flat list of (image_filename, caption) pairs
        self.samples = []
        for item in caption_data:
            image_filename = item['img_name']
            # Use only the first caption for simplicity
            caption = item['captions'][0]
            if os.path.exists(join(self.image_dir, image_filename)):
                self.samples.append((image_filename, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_filename, caption = self.samples[index]
        image_path = join(self.image_dir, image_filename)

        try:
            image = Image.open(image_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"[Dataloader WARN] Could not load image {image_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, caption


def loader(transform_fn, task, **kwargs):
    """
    Factory function for the Lemur framework.
    """
    if 'txt-image' not in task.strip().lower():
        raise ValueError(f"The task '{task}' is not a text-to-image task for this dataloader.")

    transform = transform_fn((NORM_MEAN, NORM_DEV))

    # Define the root path for the new dataset
    path = join(data_dir, 'celeba-hq')
    os.makedirs(path, exist_ok=True)

    # Create one full dataset instance
    full_dataset = CelebAHQTextToImageDataset(root=path, transform=transform)

    # Split into training and validation sets (90% / 10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(
        f"[Dataloader] Captioned CelebA-HQ loaded. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    metadata = (None,)
    performance_goal = 0.0
    return metadata, performance_goal, train_dataset, val_dataset