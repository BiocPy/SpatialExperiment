from copy import deepcopy
import biocutils as ut

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


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
