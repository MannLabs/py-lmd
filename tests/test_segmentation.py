import os
import pathlib

import geopandas as gpd
import numpy as np
import pytest
import shapely
from lxml import etree as ET
from PIL import Image

from lmd import tools
from lmd._utils import _download_segmentation_example_file
from lmd.lib import Collection, SegmentationLoader, Shape



def test_segmentation_loader():
    test_segmentation_path = _download_segmentation_example_file()

    im = Image.open(test_segmentation_path)
    segmentation = np.array(im).astype(np.uint32)

    all_classes = np.unique(segmentation)

    cell_sets = [{"classes": all_classes, "well": "A1"}]

    calibration_points = np.array([[0, 0], [0, 1000], [1000, 1000]])

    loader_config = {"orientation_transform": np.array([[0, -1], [1, 0]])}

    sl = SegmentationLoader(config=loader_config)
    sl(segmentation, cell_sets, calibration_points)
