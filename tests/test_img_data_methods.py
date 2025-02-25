import pytest
from copy import deepcopy
from spatialexperiment.SpatialImage import VirtualSpatialImage

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_get_img_without_img_data(spe):
    tspe = deepcopy(spe)

    tspe.img_data = None
    assert not tspe.get_img()


def test_get_img_no_matches(spe):
    images = spe.get_img(sample_id="foo", image_id="foo")
    assert not images


def test_get_img_both_null(spe):
    res = spe.get_img(sample_id=None, image_id=None)
    image = spe.img_data["data"][0]

    assert isinstance(res, VirtualSpatialImage)
    assert res == image


def test_get_img_both_true(spe):
    res = spe.get_img(sample_id=True, image_id=True)
    images = spe.img_data["data"]

    assert isinstance(res, list)
    assert res == images


def test_get_img_specific_sample(spe):
    res = spe.get_img(sample_id="sample_1", image_id=True)
    images = spe.img_data["data"][:2]

    assert isinstance(res, list)
    assert res == images


def test_get_img_specific_image(spe):
    res = spe.get_img(sample_id=True, image_id="desert")
    images = spe.img_data["data"][2]

    assert isinstance(res, VirtualSpatialImage)
    assert res == images


def test_add_img(spe):
    tspe = spe.add_img(
        image_source="tests/images/sample_image4.png",
        scale_factor=1,
        sample_id="sample_2",
        image_id="unsplash",
    )

    assert tspe.img_data.shape[0] == spe.img_data.shape[0] + 1


def test_add_img_already_exists(spe):
    img_data = spe.img_data
    with pytest.raises(ValueError):
        spe.add_img(
            image_source="tests/images/sample_image4.png",
            scale_factor=1,
            sample_id=img_data["sample_id"][0],
            image_id=img_data["image_id"][0],
        )
