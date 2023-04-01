from pathlib import Path

import torch
from sklearn.cluster import KMeans

from io_image import read_image
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device


class Game:
    def __init__(
            self,
            ref_image_path: Path,
            model_path: Path,
            imgsz: int = 640
    ):
        # model loading
        self.device = select_device("0")
        self.model = attempt_load(model_path, map_location=self.device)
        res = self.model(
            torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(
                next(self.model.parameters())
            )
        )

        # k means color
        self.ref_image_path = ref_image_path
        self.ref_image = read_image(ref_image_path, convert_to_rgb=True)
        self._launch_kmeans()

    def _launch_kmeans(self):
        sample_image = self.ref_image.reshape(
            self.ref_image.shape[0] * self.ref_image.shape[1], 3
        )
        self.kmeans = KMeans(n_clusters=5, random_state=0).fit(sample_image)
        self.labels = self.kmeans.predict(sample_image)
        self.mean_colors = self.kmeans.cluster_centers_.astype(int)


