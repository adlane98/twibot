from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

valid_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5)
])

train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=[0.25, 1.75], contrast=[0.75, 2], saturation=[0.5, 1.5]),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(20, fill=0.2)
])


class AlexNetDataset(Dataset):
    def __init__(self, root_dir: Path, mode="train", transform=True):
        self.root_dir = root_dir
        self.mode = mode
        if transform:
            if mode == "train":
                self.transform = train_transforms
            elif mode == "val":
                self.transform = valid_transforms
            else:
                raise AttributeError("Invalid mode (train or val)")
        else:
            self.transform = None

        self.classes = list(range(1, 121))
        self.labels = list(range(120))

        self.path_file = self.root_dir / f"{mode}.txt"
        with self.path_file.open("r") as f:
            self.image_pathes = [Path(p[:-1]) for p in f.readlines()]
            self.image_classes = [int(p.parent.stem) for p in self.image_pathes]
            self.image_labels = [c - 1 for c in self.image_classes]

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, item):
        image = cv2.imread(str(self.image_pathes[item]))
        label = torch.tensor(self.image_labels[item])

        image = cv2.resize(image, (128, 128))
        image = np.transpose(image, (2, 0, 1))
        image = torch.Tensor(image) / 255
        if self.transform:
            image = self.transform(image)

        return image.cuda(), label.cuda()


def build_dataloader(root_dir, batch_size=32, mode="train", transform=True, shuffle=True, num_workers=8):
    dataset = AlexNetDataset(root_dir, mode, transform)
    return dataset, DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)
