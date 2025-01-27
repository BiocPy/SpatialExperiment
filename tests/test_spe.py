from biocframe import BiocFrame
from spatialexperiment import SpatialExperiment

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_SPE_empty_constructor():
    spe = SpatialExperiment()

    assert isinstance(spe, SpatialExperiment)

    assert isinstance(spe.img_data, BiocFrame)
    assert spe.img_data.shape[0] == 0

    assert len(spe.spatial_coords_names) == 0
    assert isinstance(spe.spatial_coords, BiocFrame)
    assert spe.spatial_coords.shape == (spe.shape[1], 0)

    assert "sample_id" in spe.column_data.columns.as_list()
    assert spe.column_data.shape == (spe.shape[1], 1)
