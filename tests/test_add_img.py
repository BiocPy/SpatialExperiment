import pytest

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


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
