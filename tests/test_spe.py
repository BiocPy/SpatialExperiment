from random import random

import genomicranges
import numpy as np
import pandas as pd
from biocframe import BiocFrame

from spatialexperiment import SpatialExperiment, SpatialImage

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


nrows = 200
ncols = 6
counts = np.random.rand(nrows, ncols)
row_data = BiocFrame(
    {
        "seqnames": [
            "chr1",
            "chr2",
            "chr2",
            "chr2",
            "chr1",
            "chr1",
            "chr3",
            "chr3",
            "chr3",
            "chr3",
        ]
        * 20,
        "starts": range(100, 300),
        "ends": range(110, 310),
        "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"] * 20,
        "score": range(0, 200),
        "GC": [random() for _ in range(10)] * 20,
    }
)

gr = genomicranges.GenomicRanges.from_pandas(row_data.to_pandas())

col_data = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)

x_coords = np.random.uniform(low=0.0, high=100.0, size=nrows)
y_coords = np.random.uniform(low=0.0, high=100.0, size=nrows)

spatial_coords = pd.DataFrame({
    'x': x_coords,
    'y': y_coords
})

img_data = BiocFrame(
    {
        "sample_id": ["sample_1", "sample_1", "sample_2"],
        "image_id": ["aurora", "dice", "desert"],
        "data": [SpatialImage("images/sample_image1.jpg"), SpatialImage("images/sample_image2.png"), SpatialImage("images/sample_image3.jpg")],
        "scale_factor": [1, 1, 1]
    }
)

def test_SPE_empty_constructor():
    spe = SpatialExperiment()

    assert spe is not None
    assert isinstance(spe, SpatialExperiment)
    assert isinstance(spe.img_data, BiocFrame)
    assert len(spe.img_data) == 0
    assert len(spe.spatial_coords_names) == 0
    assert isinstance(spe.spatial_coords, BiocFrame)
    assert spe.spatial_coords.shape == (spe.shape[1], 0)
