from spatialexperiment import SpatialExperiment

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_drop_all_samples(spe):
    tspe = spe[:, []]

    assert tspe.shape[1] == 0
    assert tspe.shape[0] == spe.shape[0]

    assert tspe.img_data.shape == (0, 4)


# bug for empty row slicing in SCE
# https://github.com/BiocPy/SingleCellExperiment/issues/59
# def test_drop_all_features(spe):
#     tspe = spe[[], :]

#     assert tspe.shape == (0, spe.shape[1])
#     assert tspe.img_data == spe.img_data


def test_spe_slice_removes_sample(spe):
    mask = ["sample_1" == sample_id for sample_id in spe.column_data["sample_id"]]
    tspe_slice = spe[:, mask]

    assert tspe_slice is not None
    assert isinstance(tspe_slice, SpatialExperiment)

    assert set(tspe_slice.column_data["sample_id"]) == {"sample_1"}
    assert set(tspe_slice.img_data["sample_id"]) == {"sample_1"}
