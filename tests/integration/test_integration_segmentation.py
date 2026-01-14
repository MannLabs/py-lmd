from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lmd.lib import SegmentationLoader


@pytest.fixture
def test_image(tmp_path: Path) -> Path:
    """Create a segmentation mask image with test shapes and return its path"""
    # Create grayscale segmentation mask (not RGB) with integer labels
    image = np.zeros(shape=(1500, 1500), dtype=np.uint32)

    # Add some labeled regions (class 1 and class 2)
    image[10:20, 10:20] = 1
    image[100:150, 100:150] = 2

    # Save to temp directory
    image_path = tmp_path / "test_segmentation.tif"
    Image.fromarray(image).save(image_path)

    return image_path


def test_segmentation_loader(test_image: Path) -> None:
    im = Image.open(test_image)
    segmentation = np.array(im).astype(np.uint32)

    all_classes = np.unique(segmentation)

    cell_sets = [{"classes": all_classes, "well": "A1"}]

    calibration_points = np.array([[0, 0], [0, 1000], [1000, 1000]])

    loader_config = {"orientation_transform": np.array([[0, -1], [1, 0]])}

    sl = SegmentationLoader(config=loader_config)
    sl(segmentation, cell_sets, calibration_points)
