from copy import deepcopy
from random import random

import numpy as np
from biocframe import BiocFrame

from spatialexperiment import SpatialExperiment, SpatialImage

nrows = 200
ncols = 500
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
        * int(nrows / 10),
        "starts": range(100, 100 + nrows),
        "ends": range(110, 110 + nrows),
        "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"] * int(nrows / 10),
        "score": range(0, nrows),
        "GC": [random() for _ in range(10)] * int(nrows / 10),
    }
)

col_data = BiocFrame(
    {
        "n_genes": [50, 200] * int(ncols / 2),
        "condition": ["healthy", "tumor"] * int(ncols / 2),
        "cell_id": ["spot_1", "spot_2"] * int(ncols / 2),
        "sample_id": ["sample_1", "sample_2"] * int(ncols / 2)
    }
)

x_coords = np.random.uniform(low=0.0, high=100.0, size=ncols)
y_coords = np.random.uniform(low=0.0, high=100.0, size=ncols)

spatial_coords = BiocFrame({
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

spe = SpatialExperiment(
    assays={"counts": counts},
    row_data=row_data,
    column_data=col_data,
    spatial_coords=spatial_coords,
    img_data=img_data
)

def test_set_col_data_to_none():
    tmp = deepcopy(spe)
    tmp.column_data = None 

    assert tmp.col_data.columns.as_list() == ["sample_id"]
    assert tmp.column_data["sample_id"] == spe.column_data["sample_id"]

def test_valid_col_data_without_sample_id():
    tmp = deepcopy(spe)

    new_col_data = BiocFrame(
        {
            "barcode": list(range(ncols))
        }
    )

    tmp.column_data = new_col_data

    assert spe.column_data["sample_id"] == tmp.column_data["sample_id"]
