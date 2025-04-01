import os
import pytest
import numpy as np
from biocframe import BiocFrame
from spatialexperiment import SpatialExperiment, construct_spatial_image_class
from random import random


@pytest.fixture
def spe():
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
            "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"]
            * int(nrows / 10),
            "score": range(0, nrows),
            "GC": [random() for _ in range(10)] * int(nrows / 10),
        }
    )

    col_data = BiocFrame(
        {
            "n_genes": [50, 200] * int(ncols / 2),
            "condition": ["healthy", "tumor"] * int(ncols / 2),
            "cell_id": ["spot_1", "spot_2"] * int(ncols / 2),
            "sample_id": ["sample_1"] * int(ncols / 2) + ["sample_2"] * int(ncols / 2),
        }
    )

    row_names = BiocFrame({"row_names": range(nrows)})

    column_names = BiocFrame({"column_names": range(ncols)})

    x_coords = np.random.uniform(low=0.0, high=100.0, size=ncols)
    y_coords = np.random.uniform(low=0.0, high=100.0, size=ncols)

    spatial_coords = BiocFrame({"x": x_coords, "y": y_coords})

    img_data = BiocFrame(
        data={
            "sample_id": ["sample_1", "sample_1", "sample_2"],
            "image_id": ["aurora", "dice", "desert"],
            "data": [
                construct_spatial_image_class("tests/images/sample_image1.jpg"),
                construct_spatial_image_class("tests/images/sample_image2.png"),
                construct_spatial_image_class("tests/images/sample_image3.jpg"),
            ],
            "scale_factor": [1, 1, 1],
        },
        row_names=[0, 1, 2]
    )

    spe_instance = SpatialExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        row_names=row_names,
        column_names=column_names,
        spatial_coords=spatial_coords,
        img_data=img_data,
    )

    return spe_instance

@pytest.fixture
def dir():
    return "tests/10xVisium"

@pytest.fixture
def sample_ids():
    return ["section1", "section2"]

@pytest.fixture
def samples(dir, sample_ids):
    return [os.path.join(dir, sample_id, "outs") for sample_id in sample_ids]
