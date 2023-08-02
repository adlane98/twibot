import itertools
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def normalize_img(image):
    return cv2.resize(image, (128, 128)) / 255


class SiameseDataset(Dataset):
    def __init__(self, training_dir, mode):
        self.nb_cards = 80
        self.mode = mode
        self.training_dir = training_dir

        self.transforms = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            T.RandomAutocontrast()
        ]) if self.mode == "train" else T.Compose([T.ToTensor()])

        self.image_pairs = list(itertools.combinations(range(1, self.nb_cards+1), 2))

    def __getitem__(self, index):
        if index % 2 == 0:
            new_index = index // 2
            image_pair = self.image_pairs[new_index]
        else:
            image_pair = random.randint(1, 80)
            image_pair = (image_pair, image_pair)

        x1_folder = Path(r"E:\twibot\twinit-dataset\cards") / str(image_pair[0])
        x2_folder = Path(r"E:\twibot\twinit-dataset\cards") / str(image_pair[1])

        x1_path, x2_path = "", ""
        while x1_path == x2_path:
            x1_path = random.choice(list(x1_folder.glob("*.jpg")))
            x2_path = random.choice(list(x2_folder.glob("*.jpg")))

        x1 = cv2.imread(str(x1_path))
        x2 = cv2.imread(str(x2_path))
        sim = index % 2

        x1 = normalize_img(x1)
        x2 = normalize_img(x2)

        x1_t = self.transforms(x1).float().cuda()
        x2_t = self.transforms(x2).float().cuda()
        sim_t = torch.as_tensor(sim).float().cuda()

        return x1_t, x2_t, sim_t

    def __len__(self):
        return len(self.image_pairs) * 2
