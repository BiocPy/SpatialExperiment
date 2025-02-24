from copy import deepcopy

import pytest
import numpy as np
import biocutils as ut
from spatialexperiment import SpatialExperiment

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
