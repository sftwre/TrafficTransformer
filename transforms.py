import torch
from torchvision import transforms


class ToTensor(object):
    def __call__(self, frames):
        # Convert from (T, H, W, C) to (T, C, H, W)
        frames = frames.transpose(0, 3, 1, 2)
        # Convert to tensor and normalize to [0, 1]
        return torch.from_numpy(frames).float() / 255.0


basic_transforms = transforms.Compose(
    [
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        ToTensor(),
    ]
)
