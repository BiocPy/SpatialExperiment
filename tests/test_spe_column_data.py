import pytest
from copy import deepcopy
from biocframe import BiocFrame

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_set_col_data_to_none(spe):
    tspe = deepcopy(spe)
    tspe.column_data = None

    assert tspe.col_data.columns.as_list() == ["sample_id"]
    assert tspe.column_data["sample_id"] == spe.column_data["sample_id"]


def test_valid_col_data_without_sample_id(spe):
    tspe = deepcopy(spe)

    new_col_data = BiocFrame({"barcode": list(range(spe.column_data.shape[0]))})

    tspe.column_data = new_col_data

    assert spe.column_data["sample_id"] == tspe.column_data["sample_id"]


def test_valid_sample_id(spe):
    tspe = deepcopy(spe)

    new_col_data = BiocFrame(
        {
            "n_genes": [50, 200] * int(tspe.column_data.shape[0] / 2),
            "condition": ["healthy", "tumor"] * int(tspe.column_data.shape[0] / 2),
            "cell_id": ["spot_1", "spot_2"] * int(tspe.column_data.shape[0] / 2),
            "passed_qc": [True, False] * int(tspe.column_data.shape[0] / 2),
            "sample_id": ["sample_1", "sample_2"] * int(tspe.column_data.shape[0] / 2),
        }
    )

    tspe.column_data = new_col_data


def test_invalid_sample_id(spe):
    tspe = deepcopy(spe)

    new_col_data = BiocFrame(
        {
            "n_genes": [50, 200] * int(tspe.column_data.shape[0] / 2),
            "condition": ["healthy", "tumor"] * int(tspe.column_data.shape[0] / 2),
            "cell_id": ["spot_1", "spot_2"] * int(tspe.column_data.shape[0] / 2),
            "sample_id": ["foo"] * tspe.column_data.shape[0],
        }
    )

    with pytest.raises(ValueError):
        tspe.column_data = new_col_data
