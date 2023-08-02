from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_match_score(matches):
    score = 0
    for m, n in matches:
        score += int(m.distance < 0.7 * n.distance)
    return score


def get_all_scores(descriptors, tested_card, sift, flann):
    scores = []
    for des in descriptors:
        _, tested_des = sift.detectAndCompute(tested_card, None)
        try:
            matches = flann.knnMatch(des, tested_des, k=2)
        except cv2.error:
            scores.append(0)
        else:
            scores.append(compute_match_score(matches))
    return scores


def mean_scores():

    a_dess = []
    b_dess = []

    sift = cv2.SIFT_create()


    for card_index in range(120):
        card_index_path = target_path / str(card_index + 1)

        a_card = cv2.imread(str(card_index_path / "a.jpg"),
                            cv2.IMREAD_GRAYSCALE)
        _, a_des = sift.detectAndCompute(a_card, None)
        a_dess.append(a_des)

        b_card = cv2.imread(str(card_index_path / "b.jpg"),
                            cv2.IMREAD_GRAYSCALE)
        _, b_des = sift.detectAndCompute(b_card, None)
        b_dess.append(b_des)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    all_mean_scores = []
    card_names = []
    for card_path in tqdm(base_path.glob("*.jpg")):
        card = cv2.imread(str(card_path), cv2.IMREAD_GRAYSCALE)
        card_name = card_path.stem

        a_scores = get_all_scores(a_dess, card, sift, flann)
        b_scores = get_all_scores(b_dess, card, sift, flann)

        mean_scores = (np.array(a_scores) + np.array(b_scores)) / 2

        all_mean_scores.append(mean_scores)
        card_names.append(card_name)

    all_mean_scores = pd.DataFrame(all_mean_scores, columns=list(range(1, 121)),
                                   index=card_names)
    all_mean_scores.to_csv(base_path / "mean_scores.csv", sep=";")


if __name__ == '__main__':

    base_path = Path(r"E:\twibot\twinit-dataset\video-9\cards")
    target_path = Path(r"E:\twibot\twinit-dataset\cards")

    mean_scores()

    mean_matrix_path = Path(r"E:\twibot\twinit-dataset\video-9\cards\mean_scores.csv")
    mean_matrix = pd.read_csv(mean_matrix_path, sep=";", index_col=0)

    nothing_indexes = mean_matrix.index[mean_matrix.max(axis="columns") < 30]
    for zeros_index in nothing_indexes:
        card_path = base_path / f"{zeros_index}.jpg"
        shutil.copy(card_path, target_path / "nothing" / card_path.name)

    for card_name, card_index in mean_matrix.idxmax(axis="columns").iteritems():
        if card_name not in nothing_indexes:
            new_card_path = target_path / card_index / "infered"
            new_card_path.mkdir(exist_ok=True)
            new_card_path = new_card_path / f"{card_name}.jpg"

            card_path = base_path / f"{card_name}.jpg"

            shutil.copy(card_path, new_card_path)