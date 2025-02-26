from copy import deepcopy

import pytest
import numpy as np
import biocutils as ut
from spatialexperiment import SpatialExperiment

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


nrows = 200
ncols = 500
counts = np.random.rand(nrows, ncols)

x_coords = np.random.uniform(low=0.0, high=100.0, size=ncols)
y_coords = np.random.uniform(low=0.0, high=100.0, size=ncols)
spatial_coords = np.column_stack((x_coords, y_coords))


def test_spatial_coords_numpy():
    tspe = SpatialExperiment(assays={"counts": counts}, spatial_coords=spatial_coords)

    assert isinstance(tspe, SpatialExperiment)
    assert isinstance(tspe.spatial_coords, np.ndarray)
    assert len(tspe.spatial_coords_names) == 0

    with pytest.warns(UserWarning):
        tspe.spatial_coordinates_names = ["x", "y"]


def test_set_spatial_coords_numpy(spe):
    tspe = deepcopy(spe)

    tspe.spatial_coords = spatial_coords

    assert np.array_equal(tspe.spatial_coords, spatial_coords)
    assert isinstance(tspe.spatial_coords, np.ndarray)
    assert len(tspe.spatial_coords_names) == 0


def test_spatial_coords_names(spe):
    assert spe.spatial_coords_names == spe.spatial_coords.columns.as_list()


def test_set_spatial_coords_names(spe):
    tspe = deepcopy(spe)

    new_spatial_coords_names = list(map(str, range(len(spe.spatial_coords_names))))

    tspe.spatial_coords_names = new_spatial_coords_names

    assert tspe.spatial_coords_names == new_spatial_coords_names
    assert tspe.spatial_coords_names == tspe.spatial_coords.columns.as_list()


def test_get_scale_factors(spe):
    sfs = spe.get_scale_factors(sample_id=True, image_id=True)

    assert ut.is_list_of_type(sfs, float) or ut.is_list_of_type(sfs, int)
    assert len(sfs) == spe.img_data.shape[0]
    assert sfs == spe.img_data["scale_factor"]
