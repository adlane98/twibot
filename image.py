from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from game import Game
from utils import read_image, show_image


class TwinitImage:
    def __init__(self, image_path: Path, game: Game, results_path: Path = None):
        self.image_path = image_path
        self.results_path = results_path

        self.closing_path = None
        self.contrast_path = None

        self.contours = None
        self.image_with_contours = None
        self.cards = []

        self.game = game

        if self.results_path:
            self.closing_path = results_path / "closing"
            self.contrast_path = results_path / "contrast"

            results_path.mkdir(exist_ok=True)
            self.closing_path.mkdir(exist_ok=True)
            self.contrast_path.mkdir(exist_ok=True)

        self.image = read_image(image_path, convert_to_rgb=True)

    def get_contours(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # threshold
        _, thresholded = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # closing
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(
            thresholded, cv2.MORPH_CLOSE, kernel, iterations=5
        )

        self.contours = cv2.findContours(closing, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        self.contours = self.contours[0] if len(self.contours) == 2 \
            else self.contours[1]

        if self.results_path:
            cv2.imwrite(str(self.closing_path / self.image_path.name), closing)

        return self.contours

    def get_cards(self, draw=False):
        if draw:
            self.image_with_contours = self.image.copy()

        for c in self.contours:
            if cv2.contourArea(c) > 10000:
                rot_rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rot_rect)
                box = box.astype(int)
                self.cards.append(Card(self, box, self.game))

                if draw:
                    cv2.drawContours(
                        self.image_with_contours, [box], 0, (0, 255, 0), 3
                    )

        if draw and self.results_path:
            cv2.imwrite(
                str(self.results_path / self.image_path.name),
                self.image_with_contours
            )

        return self.cards

    def draw_card_countours(self, indexes: list):
        res_image = self.image.copy()
        for i in indexes:
            card = self.cards[i]
            cv2.drawContours(
                res_image, [card.coords], 0, (0, 255, 0), 3
            )
        return res_image

    def segment_cards(self):
        self.get_contours()
        self.get_cards()


class Card:
    def __init__(self, image_belonged, coords, game):
        self.image_belonged = image_belonged
        self.coords = coords

        card_weight_size = np.max(coords[:, 0]) - np.min(coords[:, 0])
        card_height_size = np.max(coords[:, 1]) - np.min(coords[:, 1])
        self.yolo_bbox = [
            (np.min(coords[:, 0]) + card_weight_size/2) / image_belonged.image.shape[1],
            (np.min(coords[:, 1]) + card_height_size/2) / image_belonged.image.shape[0],
            card_weight_size / image_belonged.image.shape[1],
            card_height_size / image_belonged.image.shape[0]
        ]

        self.cropped = None
        self.straighten_card = None
        self.recolored_card = None
        self.rot = [None, None, None, None]
        self.labels = None
        self.rot_labels = [None, None, None, None]

        self.game = game

    def crop_card(self):
        mask = np.zeros_like(self.image_belonged.image[:, :, 0])
        cv2.drawContours(mask, [self.coords], 0, 255, -1)

        # Extract out the object and place into output image
        cropped = np.zeros_like(self.image_belonged.image)
        cropped[mask == 255] = self.image_belonged.image[mask == 255]

        # Now crop
        (y, x) = np.where(mask == 255)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        self.cropped = cropped[topy:bottomy + 1, topx:bottomx + 1, :]

        return self.cropped

    def _compute_angle(self):
        corner0 = self.coords[0]
        corner1 = self.coords[1]

        angle = np.arctan(abs(corner0[0] - corner1[0]) / abs(corner0[1] - corner1[1]))

        return angle * 180 / np.pi

    def straighten(self, index=0):
        angle = self._compute_angle()
        image_center = tuple(np.array(self.cropped.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            self.cropped,
            rot_mat,
            self.cropped.shape[1::-1],
            flags=cv2.INTER_LINEAR
        )

        min_l, max_l, min_c, max_c = get_min_max_coord(result)
        result = cv2.resize(result[min_l+1:max_l, min_c+1:max_c, :], dsize=(128, 128))

        self.straighten_card = result
        return self.straighten_card

    def get_hist(self):
        hsv = cv2.cvtColor(self.cropped, cv2.COLOR_BGR2HSV)
        hist_hsv = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_hsv, hist_hsv, alpha=0, beta=0, norm_type=cv2.NORM_MINMAX)

        plt.plot(hist_hsv)
        plt.show()

    def recolor(self):
        recolored_card = np.zeros_like(self.straighten_card)

        sample_image = np.reshape(
            self.straighten_card,
            (np.prod(self.straighten_card.shape[:-1]), 3)
        )
        labels = self.game.kmeans.predict(sample_image)
        self.labels = np.reshape(labels, self.straighten_card.shape[:-1])

        for i in range(self.game.mean_colors.shape[0]):
            recolored_card[self.labels == i] = self.game.mean_colors[i, ...]

        self.recolored_card = recolored_card
        return self.recolored_card

    def compute_rotations(self):
        self.rot[0] = self.recolored_card
        self.rot_labels[0] = self.labels

        for i in range(1, 4):
            self.rot[i] = cv2.rotate(self.rot[i-1], cv2.ROTATE_90_CLOCKWISE)
            self.rot_labels[i] = cv2.rotate(
                self.rot_labels[i-1], cv2.ROTATE_90_CLOCKWISE
            )


def compute_card_diff(card1: Card, card2: Card):
    return [
        np.equal(card1.labels, card2.rot_labels[i]).sum() / (128*128)
        for i in range(4)
    ]


def get_min_max_coord(img):
    min_l, max_l, min_c, max_c = 0, img.shape[0], 0, img.shape[1]

    lines_to_remove = []
    for i in range(img.shape[0]):
        if img[i, :, :].sum() < 10000:
            lines_to_remove.append(i)

    cols_to_remove = []
    for i in range(img.shape[1]):
        if img[:, i, :].sum() < 10000:
            cols_to_remove.append(i)

    last_id = 0
    for i in range(1, len(lines_to_remove)):
        current_id = lines_to_remove[i]
        if last_id + 1 != current_id:
            min_l = lines_to_remove[i-1]
            max_l = lines_to_remove[i]
            break
        last_id = current_id

    last_id = 0
    for i in range(1, len(cols_to_remove)):
        current_id = cols_to_remove[i]
        if last_id + 1 != current_id:
            min_c = cols_to_remove[i-1]
            max_c = cols_to_remove[i]
            break
        last_id = current_id

    return min_l, max_l, min_c, max_c