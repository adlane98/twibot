from pathlib import Path
import random


def make_train_val_set(size, cards, dict_to_check):
    dataset = []
    prime_dict = {}

    count = 0
    while count < size:
        print(count)
        (anchor, negative) = random.sample(cards, 2)

        k = f"{anchor}_{negative}"
        try:
            x = prime_dict[k]
            for dc in dict_to_check:
                x = dc[k]
                x = dc[k]
            continue
        except KeyError:
            prime_dict[k] = 1
            count += 1

        anchor_folder = cards_path / str(anchor)
        anchor_images = list(anchor_folder.glob("*.jpg"))
        anchor_image, positive_image = random.sample(anchor_images, k=2)

        negative_folder = cards_path / str(negative)
        negative_images = list(negative_folder.glob("*.jpg"))
        negative_image = random.choice(negative_images)

        string_triplet = f"{anchor_image};{positive_image};{negative_image}\n"
        dataset.append(string_triplet)

    return dataset, prime_dict


if __name__ == '__main__':
    cards_path = Path(r"E:\twibot\twinit-dataset\video-4\cards")
    cards_number = list(range(1, 38))
    only_validation_cards = [17, 24, 32, 34, 35]
    train_cards = list(set(cards_number) - set(only_validation_cards))

    train_set, train_dict = make_train_val_set(900, train_cards, [])
    val_set, val_dict = make_train_val_set(20, only_validation_cards, [])
    val_set.extend(make_train_val_set(80, cards_number, [train_dict, val_dict])[0])

    with (cards_path.parent / "train.txt").open("w") as f:
        f.writelines(train_set)

    with (cards_path.parent / "val.txt").open("w") as f:
        f.writelines(val_set)
