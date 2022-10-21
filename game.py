from pathlib import Path

from sklearn.cluster import KMeans

from utils import read_image


class Game:
    def __init__(self, ref_image_path: Path):
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


