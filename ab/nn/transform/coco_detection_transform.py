import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image, target):
        # Original size for scale calculation
        orig_size = image.size[::-1]  # PIL size is (width, height)
        image = F.resize(image, self.size)
        
        if target is None:
            return image, target
            
        # Rescale bounding boxes
        h_ratio = self.size[0] / orig_size[0]
        w_ratio = self.size[1] / orig_size[1]
        
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes.clone()
            scaled_boxes[:, [0, 2]] *= w_ratio  # rescale x
            scaled_boxes[:, [1, 3]] *= h_ratio  # rescale y
            target["boxes"] = scaled_boxes
        
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

def transform(**kwargs):
    """
    Returns transform for object detection:
    - Resizes image to 320x320 (SSDLite's expected input size)
    - Converts to tensor
    - Normalizes with ImageNet stats
    """
    return Compose([
        Resize((320, 320)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
