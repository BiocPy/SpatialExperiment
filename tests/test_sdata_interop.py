import pytest
from spatialexperiment import SpatialExperiment

def test_from_sdata(sdata):
    spe = SpatialExperiment.from_spatialdata(sdata)

    assert isinstance(spe, SpatialExperiment)

    table = sdata['table']
    assert spe.shape == (table.shape[1], table.shape[0])

    sdata_points = next(iter(sdata.points.values()))
    assert spe.spatial_coords.shape == (len(sdata_points), sdata_points.shape[1])
    assert sorted(spe.spatial_coords.columns.as_list()) == sorted(['x','y'])

    assert spe.img_data.shape == (1, 4)
    assert spe.img_data["sample_id"] == ["sample01"]
    assert spe.img_data["image_id"] == ["image01"]
    assert spe.img_data["scale_factor"] == [1]


def test_invalid_input():
    with pytest.raises(TypeError):
        SpatialExperiment.from_spatialdata("Not a SpatialData object!")