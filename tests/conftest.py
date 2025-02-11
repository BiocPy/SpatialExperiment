import pytest
from random import random
import numpy as np
from biocframe import BiocFrame
import anndata as ad
import spatialdata as sd
from spatialexperiment import SpatialExperiment, construct_spatial_image_class
from spatialdata.models import Image2DModel, PointsModel


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

    x_coords = np.random.uniform(low=0.0, high=100.0, size=ncols)
    y_coords = np.random.uniform(low=0.0, high=100.0, size=ncols)

    spatial_coords = BiocFrame({"x": x_coords, "y": y_coords})

    img_data = BiocFrame(
        {
            "sample_id": ["sample_1", "sample_1", "sample_2"],
            "image_id": ["aurora", "dice", "desert"],
            "data": [
                construct_spatial_image_class("tests/images/sample_image1.jpg"),
                construct_spatial_image_class("tests/images/sample_image2.png"),
                construct_spatial_image_class("tests/images/sample_image3.jpg"),
            ],
            "scale_factor": [1, 1, 1],
        }
    )

    spe_instance = SpatialExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        spatial_coords=spatial_coords,
        img_data=img_data,
    )

    return spe_instance


@pytest.fixture
def sdata():
    img = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    img = Image2DModel.parse(data=img)
    img.name = "image01"
    img.attrs['scale_factor'] = 1

    num_cols = 25
    x_coords = np.random.uniform(low=0.0, high=100.0, size=num_cols)
    y_coords = np.random.uniform(low=0.0, high=100.0, size=num_cols)
    stacked_coords = np.column_stack((x_coords, y_coords))
    points = PointsModel.parse(stacked_coords)

    n_vars = 10
    X = np.random.random((num_cols, n_vars))
    adata = ad.AnnData(X=X)

    sdata = sd.SpatialData(
        images={"sample01": img},
        points={"coords": points},
        tables=adata
    )

    return sdata


@pytest.fixture
def sdata_tree():
    img_1 = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    img_1 = Image2DModel.parse(data=img_1)
    img_1.name = "image01"
    
    img_2 = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    img_2 = Image2DModel.parse(data=img_2)
    img_2.name = "image02"
