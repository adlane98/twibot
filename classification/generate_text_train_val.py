from pathlib import Path
import random

if __name__ == '__main__':
    cards_folder = Path(r"E:\twibot\twinit-dataset\cards")
    train_pathes = []
    val_pathes = []

    for card_index in range(1, 121):
        card_pathes = list((cards_folder / str(card_index)).glob("*.jpg"))
        random.shuffle(card_pathes)
        val_number = int(0.1 * len(card_pathes))
        val_pathes.extend(card_pathes[:val_number])
        train_pathes.extend(card_pathes[val_number:])

    random.shuffle(train_pathes)
    random.shuffle(val_pathes)

    train_pathes = [f"{p}\n" for p in train_pathes]
    val_pathes = [f"{p}\n" for p in val_pathes]

    with (cards_folder / "dataset" / "train.txt").open("w") as f:
        f.writelines(train_pathes)

    with (cards_folder / "dataset" / "val.txt").open("w") as f:
        f.writelines(val_pathes)
