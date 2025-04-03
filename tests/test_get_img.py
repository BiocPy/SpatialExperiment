import pytest
from copy import deepcopy
import numpy as np

from spatialexperiment.SpatialImage import VirtualSpatialImage

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_get_img_no_img_data(spe):
    tspe = deepcopy(spe)

    tspe.img_data = None
    assert not tspe.get_img()


def test_get_img_no_matches(spe):
    with pytest.raises(ValueError):
        images = spe.get_img(sample_id="foo", image_id="foo")


def test_get_img_both_str(spe):
    res = spe.get_img(sample_id="sample_1", image_id="dice")
    images = spe.img_data[np.array(spe.img_data["sample_id"]) == "sample_1",]
    images = images[np.array(images["image_id"]) == "dice",]["data"][0]

    assert isinstance(res, VirtualSpatialImage)
    assert res == images


def test_get_img_both_true(spe):
    res = spe.get_img(sample_id=True, image_id=True)
    images = spe.img_data["data"]

    assert isinstance(res, list)
    assert res == images


def test_get_img_both_none(spe):
    res = spe.get_img(sample_id=None, image_id=None)
    image = spe.img_data[0,]["data"][0]

    assert isinstance(res, VirtualSpatialImage)
    assert res == image


def test_get_img_sample_str_image_true(spe):
    res = spe.get_img(sample_id="sample_1", image_id=True)
    images = spe.img_data[np.array(spe.img_data["sample_id"]) == "sample_1",]["data"]

    assert isinstance(res, list)
    assert res == images


def test_get_img_sample_true_image_str(spe):
    res = spe.get_img(sample_id=True, image_id="desert")
    images = spe.img_data[np.array(spe.img_data["image_id"]) == "desert",]["data"][0]

    assert isinstance(res, VirtualSpatialImage)
    assert res == images


def test_get_img_sample_str_image_none(spe):
    res = spe.get_img(sample_id="sample_1", image_id=None)
    images = spe.img_data[np.array(spe.img_data["sample_id"]) == "sample_1",]["data"][0]

    assert isinstance(res, VirtualSpatialImage)
    assert res == images


def test_get_img_sample_none_image_str(spe):
    res = spe.get_img(sample_id=None, image_id="aurora")
    images = spe.img_data[np.array(spe.img_data["image_id"]) == "aurora",]["data"][0]

    assert isinstance(res, VirtualSpatialImage)
    assert res == images


def test_get_img_sample_true_image_none(spe):
    res = spe.get_img(sample_id=True, image_id=None)
    idxs = [spe.img_data["sample_id"].index(x) for x in set(spe.img_data["sample_id"])]
    images = spe.img_data[idxs,]["data"]

    assert isinstance(res, list) and all(isinstance(item, VirtualSpatialImage) for item in res)
    assert set(res) == set(images)


def test_get_img_sample_none_image_true(spe):
    res = spe.get_img(sample_id=None, image_id=True)
    first_sample_id = spe.img_data["sample_id"][0]
    images = spe.img_data[np.array(spe.img_data["sample_id"]) == first_sample_id,]["data"]

    assert isinstance(res, list) and all(isinstance(item, VirtualSpatialImage) for item in res)
    assert set(res) == set(images)
