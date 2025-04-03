import pytest
from copy import deepcopy
from pathlib import Path
from PIL import Image
import numpy as np
from spatialexperiment import construct_spatial_image_class
from spatialexperiment.SpatialImage import (
    LoadedSpatialImage,
    StoredSpatialImage,
    RemoteSpatialImage
)

def test_loaded_spatial_image_img_source():
    image = Image.open("tests/images/sample_image2.png")
    spi_loaded = construct_spatial_image_class(image, is_url=False)

    assert isinstance(spi_loaded, LoadedSpatialImage)
    assert spi_loaded.img_source() is None
    assert spi_loaded.img_source(as_path=True) is None

    np_image = np.zeros((100, 100, 3), dtype=np.uint8)
    spi_loaded_np = construct_spatial_image_class(np_image)

    assert isinstance(spi_loaded_np, LoadedSpatialImage)
    assert spi_loaded_np.img_source() is None
    assert spi_loaded_np.img_source(as_path=True) is None


def test_stored_spatial_image_img_source():
    image_path = "tests/images/sample_image1.jpg"
    spi_stored = construct_spatial_image_class(image_path, is_url=False)

    assert isinstance(spi_stored, StoredSpatialImage)

    source_path = spi_stored.img_source()
    assert isinstance(source_path, Path)
    assert image_path in str(source_path)

    source_str = spi_stored.img_source(as_path=True)
    assert isinstance(source_str, str)
    assert image_path in source_str

    assert str(source_path) == str(spi_stored.path)


def test_remote_spatial_image_img_source():
    image_url = "https://example.com/test_image.jpg"
    spi_remote = construct_spatial_image_class(image_url, is_url=True)

    assert isinstance(spi_remote, RemoteSpatialImage)

    source = spi_remote.img_source()
    assert isinstance(source, str)
    assert source == image_url


def test_remote_spatial_image_img_source_with_mock(monkeypatch):
    image_url = "https://example.com/test_image.jpg"
    spi_remote = construct_spatial_image_class(image_url, is_url=True)

    assert isinstance(spi_remote, RemoteSpatialImage)

    # Mock the _download_image method to return a fixed path
    mock_path = Path("/tmp/image.jpg")
    monkeypatch.setattr(spi_remote, "_download_image", lambda: mock_path)

    # Test with as_path=True (returns the cached path)
    source_path = spi_remote.img_source(as_path=True)
    assert source_path == str(mock_path)

    # Test default behavior returns URL
    assert spi_remote.img_source() == image_url


def test_img_source_no_img_data(spe):
    tspe = deepcopy(spe)
    tspe.img_data = None
    assert not tspe.img_source()


def test_img_source_no_matches(spe):
    with pytest.raises(ValueError):
        sources = spe.img_source(sample_id="foo", image_id="foo")


def test_img_source_both_str(spe):
    res = spe.img_source(sample_id="sample_1", image_id="dice")
    expected_source = spe.get_img(sample_id="sample_1", image_id="dice").img_source()

    assert isinstance(res, Path)
    assert res == expected_source


def test_img_source_both_str_path(spe):
    res = spe.img_source(sample_id="sample_1", image_id="dice", path=True)
    expected_source = spe.get_img(sample_id="sample_1", image_id="dice").img_source(as_path=True)

    assert isinstance(res, str)
    assert res == expected_source


def test_img_source_both_true(spe):
    res = spe.img_source(sample_id=True, image_id=True)
    images = spe.get_img(sample_id=True, image_id=True)
    expected_sources = [img.img_source() for img in images]

    assert isinstance(res, list)
    assert res == expected_sources


def test_img_source_both_true_path(spe):
    res = spe.img_source(sample_id=True, image_id=True, path=True)
    images = spe.get_img(sample_id=True, image_id=True)
    expected_sources = [img.img_source(as_path=True) for img in images]

    assert isinstance(res, list)
    assert res == expected_sources


def test_img_source_both_none(spe):
    res = spe.img_source(sample_id=None, image_id=None)
    expected_source = spe.get_img(sample_id=None, image_id=None).img_source()

    assert isinstance(res, Path)
    assert res == expected_source


def test_img_source_sample_str_image_true(spe):
    res = spe.img_source(sample_id="sample_1", image_id=True)
    images = spe.get_img(sample_id="sample_1", image_id=True)
    expected_sources = [img.img_source() for img in images]

    assert isinstance(res, list)
    assert res == expected_sources


def test_img_source_sample_true_image_str(spe):
    res = spe.img_source(sample_id=True, image_id="desert")
    expected_source = spe.get_img(sample_id=True, image_id="desert").img_source()

    assert isinstance(res, Path)
    assert res == expected_source


def test_img_source_sample_str_image_none(spe):
    res = spe.img_source(sample_id="sample_1", image_id=None)
    expected_source = spe.get_img(sample_id="sample_1", image_id=None).img_source()

    assert isinstance(res, Path)
    assert res == expected_source


def test_img_source_sample_none_image_str(spe):
    res = spe.img_source(sample_id=None, image_id="aurora")
    expected_source = spe.get_img(sample_id=None, image_id="aurora").img_source()

    assert isinstance(res, Path)
    assert res == expected_source


def test_img_source_sample_true_image_none(spe):
    res = spe.img_source(sample_id=True, image_id=None)
    images = spe.get_img(sample_id=True, image_id=None)
    expected_sources = [img.img_source() for img in images]

    assert isinstance(res, list)
    assert res == expected_sources


def test_img_source_sample_none_image_true(spe):
    res = spe.img_source(sample_id=None, image_id=True)
    images = spe.get_img(sample_id=None, image_id=True)
    expected_sources = [img.img_source() for img in images]

    assert isinstance(res, list)
    assert res == expected_sources
