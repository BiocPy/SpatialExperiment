from copy import deepcopy

import pytest
import biocutils as ut

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_combine_columns(spe):
    spe1 = deepcopy(spe)
    spe2 = deepcopy(spe)

    # TODO: this is a temporary fix until https://github.com/BiocPy/SpatialExperiment/issues/25 is finished
    spe1.column_data["sample_id"] = [f"{sample_id}_A" for sample_id in spe1.column_data["sample_id"]]
    spe2.column_data["sample_id"] = [f"{sample_id}_B" for sample_id in spe2.column_data["sample_id"]]
    spe1.img_data["sample_id"] = [f"{sample_id}_A" for sample_id in spe1.img_data["sample_id"]]
    spe2.img_data["sample_id"] = [f"{sample_id}_B" for sample_id in spe2.img_data["sample_id"]]

    combined = ut.combine_columns(spe1, spe2)

    # img_data checks
    assert combined.img_data.shape[0] == 2 * spe.img_data.shape[0]
    assert set(combined.column_data["sample_id"]) == set(combined.img_data["sample_id"])
    assert set(combined.column_data["sample_id"]) == set(spe1.column_data["sample_id"] + spe2.column_data["sample_id"])

    idx1 = range(spe1.img_data.shape[0])
    idx2 = range(spe1.img_data.shape[0], spe1.img_data.shape[0] + spe2.img_data.shape[0])
    img_data1 = combined.img_data[idx1, :]
    img_data2 = combined.img_data[idx2, :]

    assert img_data1["sample_id"] == spe1.img_data["sample_id"]
    assert img_data1["image_id"] == spe1.img_data["image_id"]
    assert img_data1["data"] == spe1.img_data["data"]
    assert img_data1["scale_factor"] == spe1.img_data["scale_factor"]

    assert img_data2["sample_id"] == spe2.img_data["sample_id"]
    assert img_data2["image_id"] == spe2.img_data["image_id"]
    assert img_data2["data"] == spe2.img_data["data"]
    assert img_data2["scale_factor"] == spe2.img_data["scale_factor"]

    # spatial_coords checks
    idx1 = range(spe1.spatial_coords.shape[0])
    idx2 = range(spe1.spatial_coords.shape[0], spe1.spatial_coords.shape[0] + spe2.spatial_coords.shape[0])
    spatial_coords1 = combined.spatial_coords[idx1, :]
    spatial_coords2 = combined.spatial_coords[idx2, :]

    assert (spatial_coords1.to_pandas() == spe1.spatial_coords.to_pandas()).all().all()
    assert (spatial_coords2.to_pandas() == spe2.spatial_coords.to_pandas()).all().all()


def test_duplicate_sample_ids(spe):
    with pytest.warns(UserWarning):
        combined = ut.combine_columns(spe, spe)

    assert len(set(combined.column_data["sample_id"])) == 2 * len(
        set(spe.column_data["sample_id"])
    )
    assert combined.shape[0] == spe.shape[0]
    assert combined.shape[1] == 2 * spe.shape[1]
    assert combined.rownames == spe.rownames
    assert set(combined.colnames.as_list()) == set(spe.colnames.as_list())
