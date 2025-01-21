import biocframe

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def _validate_id(id):
    is_valid = isinstance(id, str) or id is True or id is None
    if not is_valid:
        raise ValueError(f"{id} must be one of [str, True, None]")


def _validate_sample_image_ids(img_data: biocframe.BiocFrame, new_sample_id: str, new_image_id: str):
    for row in img_data:
        data = row[1]
        if data["sample_id"] == new_sample_id and data["image_id"] == new_image_id:
            raise ValueError(f"Image with Sample ID: {new_sample_id} and Image ID: {new_image_id} already exists")


def _validate_spatial_coords(spatial_coords, shape):
    if spatial_coords is None:
        return
    
    if not isinstance(spatial_coords, biocframe.BiocFrame):
        raise TypeError("'spatial_coords' is not a BiocFrame object.")
    
    if shape[1] != spatial_coords.shape[0]:
        raise ValueError(f"Spatial coordinates do not contain coordinates for all cells.")


def _validate_img_data(img_data):
    if img_data is None:
        return

    if not isinstance(img_data, biocframe.BiocFrame):
        raise TypeError("'img_data' is not a BiocFrame object.")
