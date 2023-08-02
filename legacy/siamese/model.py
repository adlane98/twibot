# create a siamese network
import torch
from torch import nn


class FeaturesGeneration(nn.Module):
    def __init__(self):
        super(FeaturesGeneration, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )
        # Defining the fully connected layers

        self.fc1 = nn.Sequential(
            nn.Linear(115200, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, img1, img2):
        features1 = self.forward_once(img1)
        features2 = self.forward_once(img2)
        return features1, features2


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.features_generation = FeaturesGeneration()
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),

            nn.Linear(512, 128),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),

            nn.Linear(128, 16),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),

            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        f1, f2 = self.features_generation(img1, img2)
        concat = torch.cat((f1, f2), dim=1)
        output = self.fc2(concat)
        return output


if __name__ == '__main__':
    m = SiameseNetwork()
    x1 = torch.rand((1, 3, 128, 128))
    x2 = torch.rand((1, 3, 128, 128))
    m(x1, x2)