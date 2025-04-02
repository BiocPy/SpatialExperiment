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

    added_img = tspe.get_img(sample_id="sample_2", image_id="unsplash")
    assert added_img is not None


def test_add_img_already_exists(spe):
    img_data = spe.img_data
    with pytest.raises(ValueError):
        spe.add_img(
            image_source="tests/images/sample_image4.png",
            scale_factor=1,
            sample_id=img_data["sample_id"][0],
            image_id=img_data["image_id"][0],
        )


def test_add_img_in_place(spe):
    original_count = spe.img_data.shape[0]
    original_id = id(spe)

    result = spe.add_img(
        image_source="tests/images/sample_image4.png",
        scale_factor=1.5,
        sample_id="sample_3",
        image_id="new_image",
        in_place=True
    )

    # Check if in-place
    assert id(result) == original_id
    assert spe.img_data.shape[0] == original_count + 1


def test_add_img_not_in_place(spe):
    result = spe.add_img(
        image_source="tests/images/sample_image4.png",
        scale_factor=2,
        sample_id="sample_4",
        image_id="another_image"
    )

    # Check if not in-place
    assert id(result) != id(spe)
    assert spe.img_data.shape[0] == spe.img_data.shape[0]
    assert result.img_data.shape[0] == spe.img_data.shape[0] + 1


def test_add_img_invalid_source(spe):
    with pytest.raises(FileNotFoundError):
        spe.add_img(
            image_source="non_existent_image.png",
            scale_factor=1,
            sample_id="sample_6",
            image_id="invalid_source"
        )
