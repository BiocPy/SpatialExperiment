from copy import deepcopy

import pytest
import biocutils as ut

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


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


def test_img_data_combined_correctly(spe):
    spe1 = deepcopy(spe)
    spe2 = deepcopy(spe)

    # TODO: this is a temporary fix until https://github.com/BiocPy/SpatialExperiment/issues/25 is finished
    spe1.column_data["sample_id"] = [f"{sample_id}_A" for sample_id in spe1.column_data["sample_id"]]
    spe2.column_data["sample_id"] = [f"{sample_id}_B" for sample_id in spe2.column_data["sample_id"]]
    spe1.img_data["sample_id"] = [f"{sample_id}_A" for sample_id in spe1.img_data["sample_id"]]
    spe2.img_data["sample_id"] = [f"{sample_id}_B" for sample_id in spe2.img_data["sample_id"]]

    with pytest.warns(None):
        combined = ut.combine_columns(spe1, spe2)

    assert combined.img_data.shape[0] == 2 * spe.img_data.shape[0]
    assert set(combined.column_data["sample_id"]) == set(combined.img_data["sample_id"])
    assert set(combined.column_data["sample_id"]) == set(spe1.column_data["sample_id"] + spe2.column_data["sample_id"])

    one = range(len(spe1.img_data))
    two = range(len(spe1.img_data), len(spe1.img_data) + len(spe2.img_data))

    # TODO: .all().all() doesn't work for BiocFrames
    assert (spe3.img_data[one, :] == spe1.img_data).all().all()
    assert (spe3.img_data[two, :] == spe2.img_data).all().all()
