import itertools
from pathlib import Path
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from game import Game
from image import TwinitImage, compute_card_diff
from utils import read_image, show_image


def test_main():
    results_path = Path(r"E:\twibot\twinit-dataset\small\first_results")
    base_path = Path(r"E:\twibot\twinit-dataset\video")

    game = Game(Path(r"E:\twibot\twinit-dataset\twinit-ref.png"))

    start_time = time.time()
    # for image_path in base_path.glob("*.jpg"):
    image_path = base_path / "325.jpg"
    image = TwinitImage(image_path, game, results_path)
    image.segment_cards()

    print("Twinit image = ", time.time() - start_time)
    time00 = time.time()
    all_cards = []
    for i, card in enumerate(image.cards):
        time0 = time.time()
        card.crop_card()
        print("Crop card = ", time.time() - time0)

        time1 = time.time()
        card.straighten(i)
        print("Straighten card = ", time.time() - time1)

        time2 = time.time()
        card.recolor()
        print("Recolor = ", time.time() - time2)
        # cv2.imwrite(str(
        #         card.image_belonged.results_path /
        #         f"{card.image_belonged.image_path.stem}-{i}.png"
        #     ), cv2.cvtColor(card.recolored_card, cv2.COLOR_RGB2BGR)
        # )
        card.compute_rotations()
        all_cards.append(card)

    print("Récupération des cartes = ", time.time() - time00)

    indexes_to_keep = []
    for i in range(len(all_cards)):
        for j in range(i+1, len(all_cards)):
            d = compute_card_diff(all_cards[i], all_cards[j])
            if any(np.array(d) > 0.64):
                indexes_to_keep.append(i)
                indexes_to_keep.append(j)
            print(f"({i}, {j}) = {compute_card_diff(all_cards[i], all_cards[j])}")

    print("Récupération des similarités = ", time.time() - start_time)

    image_detection = image.draw_card_countours(list(set(indexes_to_keep)))

    cv2.imwrite(str(
            results_path /
            f"{image_path.stem}-res.png"
        ), cv2.cvtColor(image_detection, cv2.COLOR_RGB2BGR)
    )
    print(time.time() - start_time)
    # whatever

        # card.get_hist()

        # box = keep_contours[0]
        #
        # maxl = np.max(box[:, 1])
        # minl = np.min(box[:, 1])
        # maxc = np.max(box[:, 0])
        # minc = np.min(box[:, 0])
        #
        # print(box)
        # print(maxl)
        # print(minl)
        # print(maxc)
        # print(minc)
        #
        # bbox = image[minl:maxl, minc:maxc, :]
        #
        # cv2.imwrite(str(results_path / "bbox.png"), bbox)

        # return keep_contours


if __name__ == '__main__':
    test_main()

