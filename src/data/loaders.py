"""Image loading: read a local image path into an RGB numpy array.

Albumentations expects ``HxWxC`` uint8 RGB arrays, so we read with OpenCV and
convert BGR->RGB here. URL/remote loaders can be added later as alternative
implementations behind the same ``ImageLoader`` surface.
"""

from __future__ import annotations

import cv2
import numpy as np


class ImageLoader:
    """Loads local image files as RGB uint8 numpy arrays."""

    def load(self, path: str) -> np.ndarray:
        """Read ``path`` into an ``HxWx3`` RGB uint8 array.

        Parameters:
            path (str): Filesystem path to the image.

        Returns:
            np.ndarray: RGB image array.

        Raises:
            FileNotFoundError: If the file cannot be read/decoded.
        """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
