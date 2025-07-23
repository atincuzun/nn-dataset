# File: CelebAText2Image.py
# Description: A "split-aware" dataloader for the CelebA dataset, designed
# to be a drop-in replacement for the COCO loader.

import os
import random
from os.path import join
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_file_from_google_drive
import torchvision.transforms as T

from ab.nn.util.Const import data_dir

# --- Configuration ---
# Note: The CelebA dataset is often hosted on Google Drive.
# The following are standard IDs for the aligned images and attribute files.
CELEBA_IMG_GDRIVE_ID = '1_ee_0u7vcNLOfNLegJRHmJekpWOn7O_4'
CELEBA_ATTR_GDRIVE_ID = '1_ee_0u7vcNLOfNLegJRHmJekpWOn7O_4'  # Often bundled with images

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_DEV = (0.5, 0.5, 0.5)


class CelebAText2Image(Dataset):
    """
    A PyTorch Dataset for the CelebA dataset, adapted for Text-to-Image models.

    This dataset is "split-aware".
    - For training, it returns (image, text_prompt_from_attributes).
    - For validation/testing, it returns (image, image) to work with
      evaluation loops that expect tensor labels.
    """
    base_folder = "celeba"
    img_folder = "img_align_celeba"
    file_list = {
        'img': ('img_align_celeba.zip', '00d2c5bc6d35e252742224ab0c1e8fcb'),
        'attr': ('list_attr_celeba.txt', '75e246fa4810816ffd6f842da2d478b6'),
        'partition': ('list_eval_partition.txt', 'd32c9cbf5e040fd4025c592c306e6668'),
    }

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.root = join(root, self.base_folder)
        self.transform = transform

        # PyTorch convention: 'train', 'valid', 'test'
        # CelebA partition file: 0=train, 1=valid, 2=test
        split_map = {'train': 0, 'val': 1, 'valid': 1, 'test': 2}
        if split not in split_map:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(split_map.keys())}")
        self.split_idx = split_map[split]
        self.split = split

        # --- Download and Verify Data ---
        self._download()

        # --- Load Metadata ---
        self._load_metadata()

    def _download(self):
        """Downloads and extracts the dataset if not present."""
        if os.path.isdir(self.root):
            # A simple check to see if the main image folder exists
            if os.path.exists(join(self.root, self.img_folder)):
                print('Files already downloaded and verified.')
                return

        # Download and extract the main archive (contains images and annotations)
        # Using a reliable source from a third-party that bundled them
        # NOTE: This is a large file (~1.4 GB)
        print("CelebA dataset not found. Downloading...")
        download_file_from_google_drive(
            '1_ee_0u7vcNLOfNLegJRHmJekpWOn7O_4',
            self.root,
            filename="celeba-dataset.zip",
            md5="00d2c5bc6d35e252742224ab0c1e8fcb"  # Verifies integrity
        )

        # Since torchvision doesn't have an unpack for .zip, we use our own utility
        from zipfile import ZipFile
        with ZipFile(join(self.root, "celeba-dataset.zip"), 'r') as zip_ref:
            zip_ref.extractall(self.root)

    def _load_metadata(self):
        """Loads partitions and attributes using pandas for efficiency."""
        partition_path = join(self.root, self.file_list['partition'][0])
        attr_path = join(self.root, self.file_list['attr'][0])

        # Load partition data
        partition_df = pd.read_csv(partition_path, delim_whitespace=True, header=None, index_col=0)

        # Load attribute data
        self.attr_df = pd.read_csv(attr_path, delim_whitespace=True, header=1)
        self.attr_names = list(self.attr_df.columns)

        # Filter filenames for the current split
        target_filenames = partition_df[partition_df[1] == self.split_idx].index
        self.filenames = [fn for fn in target_filenames if fn in self.attr_df.index]
        self.img_dir = join(self.root, self.img_folder)

    def _create_prompt(self, filename):
        """Generates a descriptive text prompt from positive attributes."""
        attributes = self.attr_df.loc[filename]
        positive_attrs = attributes[attributes == 1].index.tolist()

        if not positive_attrs:
            return "A photo of a person."

        # Make attributes more human-readable
        formatted_attrs = [attr.replace('_', ' ').lower() for attr in positive_attrs]
        random.shuffle(formatted_attrs)

        if len(formatted_attrs) == 1:
            prompt = f"A photo of a person with {formatted_attrs[0]}."
        elif len(formatted_attrs) == 2:
            prompt = f"A photo of a person who is {formatted_attrs[0]} and has {formatted_attrs[1]}."
        else:
            last_attr = formatted_attrs.pop()
            prompt = f"A photo of a person with {', '.join(formatted_attrs)}, and {last_attr}."

        return prompt

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Retrieves a sample. The format depends on the split.
        - 'train': returns (image_tensor, text_prompt_string)
        - 'val'/'test': returns (image_tensor, image_tensor)
        """
        filename = self.filenames[index]
        img_path = join(self.img_dir, filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"Warning: Could not load image {img_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        # --- SPLIT-AWARE LOGIC ---
        if self.split == 'train':
            # For training, generate and return the text prompt
            text_prompt = self._create_prompt(filename)
            return image_tensor, text_prompt
        else:  # For 'val' or 'test'
            # For evaluation, return the image tensor as both input and "label"
            return image_tensor, image_tensor


def loader(transform_fn, task, **kwargs):
    """
    Factory function to create train and validation datasets for CelebA.
    This is the main entry point used by the LEMUR framework.
    """
    if 'txt-image' not in task.strip().lower():
        raise ValueError(f"The task '{task}' is not a text-to-image task for this dataloader.")

    transform = transform_fn((NORM_MEAN, NORM_DEV))

    # The root directory where 'celeba' folder will be created
    path = data_dir

    train_dataset = CelebAText2Image(root=path, split='train', transform=transform)
    # Using 'val' as per the CelebA partition standard
    val_dataset = CelebAText2Image(root=path, split='val', transform=transform)

    metadata = (None,)
    performance_goal = 0.0
    return metadata, performance_goal, train_dataset, val_dataset