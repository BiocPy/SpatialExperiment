import pytest
from PIL import Image
from spatialexperiment import construct_spatial_image_class
from spatialexperiment.SpatialImage import VirtualSpatialImage, StoredSpatialImage, LoadedSpatialImage, RemoteSpatialImage

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_si_constructor_path():
    si = construct_spatial_image_class("tests/images/sample_image1.jpg", is_url=False)

    assert issubclass(type(si), VirtualSpatialImage)
    assert isinstance(si, StoredSpatialImage)

    assert "tests/images/sample_image1.jpg" in str(si.path)
    assert si.path is not None


def test_si_constructor_si():
    si_1 = construct_spatial_image_class("tests/images/sample_image1.jpg", is_url=False)
    si_2 = construct_spatial_image_class(si_1, is_url=False)

    assert issubclass(type(si_2), VirtualSpatialImage)
    assert isinstance(si_2, StoredSpatialImage)

    assert str(si_1.path) == str(si_2.path)


def test_si_constructor_image():
    image = Image.open("tests/images/sample_image2.png")
    si = construct_spatial_image_class(image, is_url=False)

    assert issubclass(type(si), VirtualSpatialImage)
    assert isinstance(si, LoadedSpatialImage)

    assert si.image == image


def test_invalid_input():
    si_remote = construct_spatial_image_class("https://i.redd.it/3pw5uah7xo041.jpg", is_url=True)
    assert issubclass(type(si_remote), VirtualSpatialImage)
    assert isinstance(si_remote, RemoteSpatialImage)
    assert si_remote.url == "https://i.redd.it/3pw5uah7xo041.jpg"

    with pytest.raises(Exception):
        construct_spatial_image_class(5, is_url=False)
