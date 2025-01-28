import pytest
from PIL import Image
from spatialexperiment import SpatialImage

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_si_constructor_path():
    si = SpatialImage("images/sample_image1.jpg")

    assert isinstance(si, SpatialImage)
    assert si.path == "images/sample_image1.jpg"
    assert si.image is None


def test_si_constructor_si():
    si_1 = SpatialImage("images/sample_image1.jpg")
    si_2 = SpatialImage(si_1)

    assert isinstance(si_2, SpatialImage)
    assert si_1.image == si_2.image
    assert si_1.path == si_2.path


def test_si_constructor_image():
    image = Image.open("tests/images/sample_image2.png")
    si = SpatialImage(image)

    assert isinstance(si, SpatialImage)
    assert si.path is None
    assert si.image == image


def test_invalid_input():
    with pytest.raises(ValueError):
        SpatialImage("https://i.redd.it/3pw5uah7xo041.jpg")

    with pytest.raises(ValueError):
        SpatialImage(5)
