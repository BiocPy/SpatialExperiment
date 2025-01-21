import biocframe
from summarizedexperiment._frameutils import _sanitize_frame

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def _validate_column_data(column_data):
    error_message = "'column_data' must have a column named 'sample_id'."

    if column_data is None:
        raise ValueError(error_message)
    
    column_data_sanitized = _sanitize_frame(column_data)
    if not column_data_sanitized.has_column("sample_id"):
        raise ValueError(error_message)

def _validate_id(id):
    is_valid = isinstance(id, str) or id is True or id is None
    if not is_valid:
        raise ValueError(f"{id} must be one of [str, True, None]")


def _validate_sample_image_ids(img_data, new_sample_id, new_image_id):
    if img_data is None:
        return

    if not isinstance(img_data, biocframe.BiocFrame):
        raise TypeError("`img_data` is not a BiocFrame object.")

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
