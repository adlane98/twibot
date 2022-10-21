from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def read_image(image_path: Path, convert_to_rgb=True):
    image = cv2.imread(str(image_path))
    if convert_to_rgb:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_image(img, gray=""):
    plt.imshow(img, cmap="gray")
    plt.show()
