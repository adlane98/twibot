from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import BCELoss
from tqdm import tqdm
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import wandb

from classification.dataset import AlexNetDataset
from classification.model import AlexNet
from dataset import build_dataloader


def get_all_embeddings(dataset, model, device):
    tester = testers.BaseTester(data_device=device)
    return tester.get_all_embeddings(dataset, model)


def get_accuracy(val_dataset, train_dataset, model, device):
    # Please make sure val dataset is passed as first argument
    # Other wise false accuracy might be reported
    # due to brodcasting with a small number of samples

    with torch.no_grad():
        val_embeddings, val_labels = get_all_embeddings(val_dataset, model,
                                                        device=device)
        train_embeddings, train_labels = get_all_embeddings(train_dataset,
                                                            model,
                                                            device=device)
        # for each val embedding, find distance with all embeddings in train embeddings
        dist = torch.cdist(val_embeddings, train_embeddings)

    query_labels = np.array(val_dataset.labels)
    # Find index of closesest matching embedding
    matched_idx = torch.argmin(dist, axis=1).cpu().numpy()
    matched_labels = np.array(train_dataset.labels)[matched_idx]

    accuracy = (query_labels == matched_labels).mean()
    return accuracy


if __name__ == '__main__':
    wandb.init(project="twibot-alexnet")

    model_path = Path("experiments") / datetime.now().strftime("%Y%m%d-%H%m%S")
    model_path.mkdir()

    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 256,
        "num_workers": 0,
        "embedding": 120,
        "pretrained": True
    }

    best_accuracy = 0

    train_dataset, train_dataloader = build_dataloader(
        root_dir=Path(r"E:\twibot\twinit-dataset\cards\dataset"),
        batch_size=wandb.config["batch_size"],
        mode="train",
        transform=True,
        shuffle=True,
        num_workers=wandb.config["num_workers"]
    )

    val_dataset, val_dataloader = build_dataloader(
        root_dir=Path(r"E:\twibot\twinit-dataset\cards\dataset"),
        batch_size=wandb.config["batch_size"],
        mode="val",
        transform=True,
        shuffle=True,
        num_workers=wandb.config["num_workers"]
    )

    net = AlexNet(
        embedding_size=wandb.config["embedding"],
        pretrained=wandb.config["pretrained"]
    ).cuda()
    optimizer = optim.Adam(net.parameters(), lr=wandb.config["learning_rate"])

    # Define triplet loss utility functions
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    # loss_func = losses.TripletMarginLoss(
    #     margin=0.2, distance=distance, reducer=reducer
    # )
    loss_func = torch.nn.CrossEntropyLoss()
    # mining_func = miners.TripletMarginMiner(
    #     margin=0.2, distance=distance, type_of_triplets="semihard"
    # )

    net.train()
    for epoch in range(wandb.config["epochs"]):

        running_loss = 0
        for i, data in tqdm(enumerate(train_dataloader)):
            img, label = data

            optimizer.zero_grad()

            embeddings = net(img)

            # indices_tuple = mining_func(embeddings, label)
            loss = loss_func(embeddings, label)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

            if i % 5 == 4:
                msg = (
                    f"Epoch [{epoch+1}/{wandb.config['epochs']}] "
                    f"Iter [{i+1}/{len(train_dataloader)}] - "
                    f"Loss: {loss.item()},"
                    # f"Triplets: {mining_func.num_triplets}"
                )
                print(msg)
            wandb.log({
                "train/batch/epoch": (i + 1 + len(train_dataloader) * epoch)/len(train_dataloader),
                "train/batch/loss": loss.item(),
            })
        wandb.log({
            "train/epoch/epoch": epoch + 1,
            "train/epoch/loss": running_loss / len(train_dataloader),
        })

        if epoch % 5 == 4:
            val_running_loss = 0
            all_labels = []
            all_preds = []
            net.eval()
            for i, data in tqdm(enumerate(val_dataloader)):
                img, label = data
                all_labels.extend(label.tolist())

                with torch.no_grad():
                    probs = net(img)

                preds = torch.argmax(probs, axis=1)
                all_preds.extend(preds.tolist())
                loss = loss_func(probs, label)

            accuracy = sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds)

            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                torch.save(net.state_dict(), model_path / "model_best.pth")

            wandb.log({
                "val/epoch/epoch": epoch + 1,
                "val/epoch/loss": val_running_loss / len(val_dataloader),
                "val/epoch/accuracy": accuracy,
                "val/epoch/precision": precision.mean(),
                "val/epoch/recall": recall.mean(),
                "val/epoch/f1_score": f1.mean(),
            })

        torch.save(net.state_dict(), model_path / f"epoch_{epoch}.pth")
