from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.base import BaseEstimator, TransformerMixin


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        lbp_points: int = 24,
        lbp_radius: int = 3,
        lbp_method: str = "uniform",
    ) -> None:
        self.width = width
        self.height = height
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.lbp_method = lbp_method

    def fit(
        self, X: Sequence[str], y: Sequence[object] | None = None
    ) -> ImageFeatureExtractor:
        del X, y
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        features: list[list[float]] = []

        for image_path in X:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_features: list[float] = []
            for channel in range(3):
                histogram = cv2.calcHist(
                    [image_rgb], [channel], None, [256], [0, 256]
                )
                color_features.extend(
                    cv2.normalize(histogram, histogram).flatten().tolist()
                )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.width, self.height))
            lbp = local_binary_pattern(
                gray,
                self.lbp_points,
                self.lbp_radius,
                method=self.lbp_method,
            )
            bin_count = self.lbp_points + 2
            lbp_histogram, _ = np.histogram(
                lbp.ravel(),
                bins=bin_count,
                range=(0, bin_count),
                density=True,
            )

            gray_small = (gray // 4).astype(np.uint8)
            levels = 64
            glcm = np.zeros((levels, levels), dtype=np.float64)
            glcm[
                gray_small[:, :-1].ravel(),
                gray_small[:, 1:].ravel(),
            ] += 1
            glcm += glcm.T
            glcm /= glcm.sum()

            i, j = np.mgrid[0:levels, 0:levels]
            mu_i = (i * glcm).sum()
            mu_j = (j * glcm).sum()
            denominator = np.sqrt((glcm * (i - mu_i) ** 2).sum()) * np.sqrt(
                (glcm * (j - mu_j) ** 2).sum()
            )
            texture_features = [
                *lbp_histogram.tolist(),
                float((glcm * (i - j) ** 2).sum()),
                float((glcm**2).sum()),
                float((glcm / (1 + (i - j) ** 2)).sum()),
                float(
                    (glcm * (i - mu_i) * (j - mu_j)).sum()
                    / (denominator + 1e-10)
                ),
            ]
            features.append(color_features + texture_features)

        return np.asarray(features, dtype=np.float64)
