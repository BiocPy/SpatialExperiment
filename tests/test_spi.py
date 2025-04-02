import pytest
from PIL import Image
from spatialexperiment import construct_spatial_image_class
from spatialexperiment.SpatialImage import (
    VirtualSpatialImage,
    StoredSpatialImage,
    LoadedSpatialImage,
    RemoteSpatialImage,
)

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_spi_constructor_path():
    spi = construct_spatial_image_class("tests/images/sample_image1.jpg", is_url=False)

    assert issubclass(type(spi), VirtualSpatialImage)
    assert isinstance(spi, StoredSpatialImage)

    assert "tests/images/sample_image1.jpg" in str(spi.path)


def test_spi_constructor_spi():
    spi_1 = construct_spatial_image_class(
        "tests/images/sample_image1.jpg", is_url=False
    )
    spi_2 = construct_spatial_image_class(spi_1, is_url=False)

    assert issubclass(type(spi_2), VirtualSpatialImage)
    assert isinstance(spi_2, StoredSpatialImage)

    assert str(spi_1.path) == str(spi_2.path)


def test_spi_constructor_image():
    image = Image.open("tests/images/sample_image2.png")
    spi = construct_spatial_image_class(image, is_url=False)

    assert issubclass(type(spi), VirtualSpatialImage)
    assert isinstance(spi, LoadedSpatialImage)

    assert spi.image == image


def test_spi_constructor_url():
    image_url = "https://example.com/test_image.jpg"
    spi_remote = construct_spatial_image_class(image_url, is_url=True)
    assert issubclass(type(spi_remote), VirtualSpatialImage)
    assert isinstance(spi_remote, RemoteSpatialImage)
    assert spi_remote.url == image_url


def test_auto_detect_url():
    url = "https://example.com/image.jpg"
    img = construct_spatial_image_class(url)
    assert isinstance(img, RemoteSpatialImage)
    assert img.img_source() == url


def test_spi_equality():
    spi_path_1 = construct_spatial_image_class(
        "tests/images/sample_image1.jpg", is_url=False
    )
    spi_path_2 = construct_spatial_image_class(
        "tests/images/sample_image1.jpg", is_url=False
    )

    assert spi_path_1 == spi_path_2

    image_url = "https://example.com/test_image.jpg"
    spi_url_1 = construct_spatial_image_class(image_url, is_url=True)
    spi_url_2 = construct_spatial_image_class(image_url, is_url=True)

    assert spi_url_1 == spi_url_2

    image = Image.open("tests/images/sample_image2.png")
    spi_image_1 = construct_spatial_image_class(image, is_url=False)
    spi_image_2 = construct_spatial_image_class(image, is_url=False)

    assert spi_image_1 == spi_image_2

    assert spi_path_1 != spi_url_1
    assert spi_path_1 != spi_image_1
    assert spi_url_1 != spi_image_1


def test_invalid_input():
    with pytest.raises(Exception):
        construct_spatial_image_class(5, is_url=False)
