from biocframe import BiocFrame
from spatialexperiment import SpatialExperiment
import numpy as np

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_SPE_empty_constructor():
    tspe = SpatialExperiment()

    assert isinstance(tspe, SpatialExperiment)

    assert isinstance(tspe.img_data, BiocFrame)
    assert tspe.img_data.shape[0] == 0

    assert len(tspe.spatial_coords_names) == 0
    assert isinstance(tspe.spatial_coords, BiocFrame)
    assert tspe.spatial_coords.shape == (tspe.shape[1], 0)

    assert "sample_id" in tspe.column_data.columns.as_list()
    assert tspe.column_data.shape == (tspe.shape[1], 1)


def test_spe_basic():
    nrows = 200
    ncols = 500
    counts = np.random.rand(nrows, ncols)
    tspe = SpatialExperiment(assays={"spots": counts})

    assert isinstance(tspe, SpatialExperiment)
