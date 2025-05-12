import pytest
from copy import deepcopy
from PIL import Image
import numpy as np
from spatialexperiment import (
    LoadedSpatialImage,
    StoredSpatialImage,
    RemoteSpatialImage,
    construct_spatial_image_class
)

def test_loaded_spatial_image_img_raster():
    image = Image.open("tests/images/sample_image2.png")
    spi_loaded = construct_spatial_image_class(image, is_url=False)
    raster = spi_loaded.img_raster()

    assert isinstance(spi_loaded, LoadedSpatialImage)
    assert isinstance(raster, Image.Image)

    np_image = np.zeros((100, 100, 3), dtype=np.uint8)
    spi_loaded_np = construct_spatial_image_class(np_image)
    raster_np = spi_loaded_np.img_raster()

    assert isinstance(spi_loaded_np, LoadedSpatialImage)
    assert isinstance(raster_np, Image.Image)


def test_stored_spatial_image_img_raster():
    image_path = "tests/images/sample_image1.jpg"
    spi_stored = construct_spatial_image_class(image_path, is_url=False)
    raster = spi_stored.img_raster()

    assert isinstance(spi_stored, StoredSpatialImage)
    assert isinstance(raster, Image.Image)


def test_remote_spatial_image_img_raster(monkeypatch):
    image_url = "https://example.com/test_image.jpg"
    spi_remote = construct_spatial_image_class(image_url, is_url=True)

    # Mock the _get_cached_path method to return an image
    mock_path = "tests/images/sample_image1.jpg"
    monkeypatch.setattr(spi_remote, "_get_cached_path", lambda: mock_path)

    raster = spi_remote.img_raster()

    assert isinstance(spi_remote, RemoteSpatialImage)
    assert isinstance(raster, Image.Image)

    # Test LRU cache works as expected
    num_calls = 0

    def mock_download():
        num_calls += 1
        return mock_path

    monkeypatch.setattr(spi_remote, "_get_cached_path", mock_download)

    raster2 = spi_remote.img_raster()
    assert num_calls == 0
    assert raster2 is raster


def test_img_raster_no_img_data(spe):
    tspe = deepcopy(spe)
    tspe.img_data = None
    assert not tspe.img_raster()


def test_img_raster_no_matches(spe):
    with pytest.raises(ValueError):
        res = spe.img_raster(sample_id="foo", image_id="foo")


def test_img_raster_both_str(spe):
    res = spe.img_raster(sample_id="sample_1", image_id="dice")
    expected_raster = spe.get_img(sample_id="sample_1", image_id="dice").img_raster()

    assert isinstance(res, Image.Image)
    assert res == expected_raster


def test_img_raster_both_true(spe):
    res = spe.img_raster(sample_id=True, image_id=True)
    images = spe.get_img(sample_id=True, image_id=True)
    expected_rasters = [img.img_raster() for img in images]

    assert isinstance(res, list)
    assert res == expected_rasters


def test_img_raster_both_none(spe):
    res = spe.img_raster(sample_id=None, image_id=None)
    expected_raster = spe.get_img(sample_id=None, image_id=None).img_raster()

    assert isinstance(res, Image.Image)
    assert res == expected_raster


def test_img_raster_sample_str_image_true(spe):
    res = spe.img_raster(sample_id="sample_1", image_id=True)
    images = spe.get_img(sample_id="sample_1", image_id=True)
    expected_rasters = [img.img_raster() for img in images]

    assert isinstance(res, list)
    assert res == expected_rasters


def test_img_raster_sample_true_image_str(spe):
    res = spe.img_raster(sample_id=True, image_id="desert")
    expected_raster = spe.get_img(sample_id=True, image_id="desert").img_raster()

    assert isinstance(res, Image.Image)
    assert res == expected_raster


def test_img_raster_sample_str_image_none(spe):
    res = spe.img_raster(sample_id="sample_1", image_id=None)
    expected_raster = spe.get_img(sample_id="sample_1", image_id=None).img_raster()

    assert isinstance(res, Image.Image)
    assert res == expected_raster


def test_img_raster_sample_none_image_str(spe):
    res = spe.img_raster(sample_id=None, image_id="aurora")
    expected_raster = spe.get_img(sample_id=None, image_id="aurora").img_raster()

    assert isinstance(res, Image.Image)
    assert res == expected_raster


def test_img_raster_sample_true_image_none(spe):
    res = spe.img_raster(sample_id=True, image_id=None)
    images = spe.get_img(sample_id=True, image_id=None)
    expected_rasters = [img.img_raster() for img in images]

    assert isinstance(res, list)
    assert res == expected_rasters


def test_img_raster_sample_none_image_true(spe):
    res = spe.img_raster(sample_id=None, image_id=True)
    images = spe.get_img(sample_id=None, image_id=True)
    expected_rasters = [img.img_raster() for img in images]

    assert isinstance(res, list)
    assert res == expected_rasters
