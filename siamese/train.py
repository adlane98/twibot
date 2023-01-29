from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import BCELoss
from tqdm import tqdm

from siamese.dataset import SiameseDataset
from siamese.loss import ContrastiveLoss
from siamese.model import SiameseNetwork

if __name__ == '__main__':

    wandb.init(project="twibot-siamese")

    wandb.config = {
        "learning_rate": 0.1,
        "epochs": 5000,
        "batch_size": 16
    }

    net = SiameseNetwork().cuda()
    criterion = BCELoss()
    # criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    train_dataset = SiameseDataset(
        Path(r"E:\twibot\twinit-dataset\video-4"), "train"
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )

    #train the model
    def train():
        epoch_losses = []

        for epoch in range(1, 5000):
            for i, data in tqdm(enumerate(train_dataloader)):
                x1, x2, gt = data

                optimizer.zero_grad()

                pred = net(x1, x2)

                loss = criterion(pred[:, 0], gt)
                loss.backward()

                optimizer.step()

                print(
                    f"Iteration {i} "
                    f"- Loss: {loss.item()} "
                )
            wandb.log({
                "loss": loss.item(),
            })
            print(f"\nEpoch {epoch} - Current loss {loss.item()}")
            epoch_losses.append(loss.item())

            torch.save(net.state_dict(), f"experiments/epoch_{epoch}.pth")
        return net

    model = train()
    torch.save(model.state_dict(), "model.pt")
    print("Model Saved Successfully")
