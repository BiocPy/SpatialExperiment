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


def test_remove_img_no_img_data(spe):
    tspe = deepcopy(spe)
    tspe.img_data = None
    with pytest.raises(AttributeError):
        tspe.remove_img()


def test_remove_img_no_matches(spe):
    with pytest.raises(ValueError):
        spe.remove_img(sample_id="foo", image_id="foo")


def test_remove_img_both_str(spe):
    tspe = deepcopy(spe)
    original_shape = tspe.img_data.shape[0]
    
    result = tspe.remove_img(sample_id="sample_1", image_id="dice")
    
    # Check if not in-place
    assert id(result) != id(tspe)
    assert tspe.img_data.shape[0] == original_shape
    assert result.img_data.shape[0] == original_shape - 1
    
    # Check the image was actually removed
    with pytest.raises(ValueError):
        result.get_img(sample_id="sample_1", image_id="dice")


def test_remove_img_in_place(spe):
    tspe = deepcopy(spe)
    original_shape = tspe.img_data.shape[0]
    
    result = tspe.remove_img(sample_id="sample_1", image_id="dice", in_place=True)
    
    # Check if in-place
    assert id(result) == id(tspe)
    assert tspe.img_data.shape[0] == original_shape - 1
    
    # Check the image was actually removed
    with pytest.raises(ValueError):
        tspe.get_img(sample_id="sample_1", image_id="dice")


def test_remove_img_both_true(spe):
    tspe = deepcopy(spe)
    original_shape = tspe.img_data.shape[0]
    
    result = tspe.remove_img(sample_id=True, image_id=True)
    
    # Check if not in-place
    assert id(result) != id(tspe)
    assert tspe.img_data.shape[0] == original_shape
    assert result.img_data.shape[0] == 0


def test_remove_img_both_none(spe):
    tspe = deepcopy(spe)
    original_shape = tspe.img_data.shape[0]
    first_sample = tspe.img_data["sample_id"][0]
    first_image = tspe.img_data["image_id"][0]
    
    result = tspe.remove_img(sample_id=None, image_id=None)
    
    # Check if not in-place
    assert id(result) != id(tspe)
    assert tspe.img_data.shape[0] == original_shape
    assert result.img_data.shape[0] == original_shape - 1
    
    # Check first image was removed
    with pytest.raises(ValueError):
        result.get_img(sample_id=first_sample, image_id=first_image)


def test_remove_img_sample_str_image_true(spe):
    tspe = deepcopy(spe)
    original_shape = tspe.img_data.shape[0]
    sample_images = len(tspe.get_img(sample_id="sample_1", image_id=True))
    
    result = tspe.remove_img(sample_id="sample_1", image_id=True)
    
    # Check correct number of images removed
    assert id(result) != id(tspe)
    assert tspe.img_data.shape[0] == original_shape
    assert result.img_data.shape[0] == original_shape - sample_images
    
    # Check no images remain for sample_1
    with pytest.raises(ValueError):
        result.get_img(sample_id="sample_1", image_id=True)


def test_remove_img_sample_true_image_str(spe):
    tspe = deepcopy(spe)
    original_shape = tspe.img_data.shape[0]
    image_count = sum(1 for img_id in tspe.img_data["image_id"] if img_id == "desert")
    
    result = tspe.remove_img(sample_id=True, image_id="desert")
    
    # Check correct number of images removed
    assert result.img_data.shape[0] == original_shape - image_count
    
    # Check no images remain with image_id "desert"
    with pytest.raises(ValueError):
        result.get_img(sample_id=True, image_id="desert")


def test_remove_img_sample_str_image_none(spe):
    tspe = deepcopy(spe)
    original_shape = tspe.img_data.shape[0]
    
    result = tspe.remove_img(sample_id="sample_1", image_id=None)
    
    # Check one image was removed
    assert result.img_data.shape[0] == original_shape - 1
    
    # Verify first image of sample_1 was removed but others might remain
    sample_1_images_before = len([i for i, s in enumerate(tspe.img_data["sample_id"]) if s == "sample_1"])
    sample_1_images_after = len([i for i, s in enumerate(result.img_data["sample_id"]) if s == "sample_1"])
    assert sample_1_images_after == sample_1_images_before - 1


def test_remove_img_sample_none_image_str(spe):
    tspe = deepcopy(spe)
    original_shape = tspe.img_data.shape[0]
    
    result = tspe.remove_img(sample_id=None, image_id="aurora")
    
    # Check one image was removed
    assert result.img_data.shape[0] == original_shape - 1
    
    # Check the specific image was removed
    with pytest.raises(ValueError):
        result.get_img(sample_id=None, image_id="aurora")