from datetime import datetime
from itertools import product
import json
from pathlib import Path

import cv2
import numpy as np
import plotly.express as px

if __name__ == '__main__':
    nb_cards = 120
    cards_folder = Path(r"E:\twibot\twinit-dataset\cards")
    sim_matrix_folder = Path(r"E:\twibot\twinit-dataset\debug\sim_matrix")

    similarity_matrix = np.zeros((nb_cards, nb_cards)).astype(int)

    parameters = {
        "flann_index_kdtree": [1, 2, 3, 4, 5],
        "trees": list(range(1, 16)),
        "search_params": [50],
        "gray": [True, False],
        "k": [2],
        "coeff_dist": list(np.array(list(range(1, 11))) / 10)
    }

    keys, values = zip(*parameters.items())
    result = [dict(zip(keys, p)) for p in product(*values)]

    for index_comb, param in enumerate(result):
        computation_times = []

        sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = param["flann_index_kdtree"]
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=param["trees"])
        search_params = dict(checks=param["search_params"])
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        for card_index_i in range(nb_cards):
            i_pathes = list((cards_folder / str(card_index_i + 1)).glob("*.jpg"))
            img1_path = cards_folder / str(card_index_i + 1) / "a.jpg"
            img2_path = cards_folder / str(card_index_i + 1) / "b.jpg"

            for card_index_j in range(card_index_i, nb_cards):
                if card_index_j > card_index_i:
                    img2_path = cards_folder / str(card_index_j + 1) / "b.jpg"

                start = datetime.now()
                g = cv2.IMREAD_GRAYSCALE if param["gray"] else cv2.IMREAD_UNCHANGED
                img1 = cv2.imread(str(img1_path), g)
                img2 = cv2.imread(str(img2_path), g)

                _, des1 = sift.detectAndCompute(img1, None)
                _, des2 = sift.detectAndCompute(img2, None)

                matches = flann.knnMatch(des1, des2, k=param["k"])

                good = []
                for m, n in matches:
                    if m.distance < param["coeff_dist"] * n.distance:
                        good.append(m)

                computation_times.append((datetime.now() - start).microseconds)

                similarity_matrix[card_index_i, card_index_j] = len(good)
                similarity_matrix[card_index_j, card_index_i] = len(good)

        score = 2 * np.trace(similarity_matrix) - np.sum(similarity_matrix)

        mean_computation_time = sum(computation_times) / len(computation_times)

        combinaison = {
            **param,
            "score": str(score),
            "time_exec": str(mean_computation_time)
        }

        comb_folder = sim_matrix_folder / f"comb_{index_comb}"
        comb_folder.mkdir(exist_ok=True)

        with (comb_folder / "combinaison.json").open("w") as f:
            json.dump(combinaison, f, indent=4)

        np.savetxt(
            str(comb_folder / "similarity_matrix.csv"),
            similarity_matrix,
            fmt="%i",
            delimiter=";"
        )

        # similarity_matrix[similarity_matrix > 49] = 25

        fig = px.imshow(similarity_matrix, color_continuous_scale='gray')
        fig.write_image(str(comb_folder / "similarity_matrix.jpg"))

