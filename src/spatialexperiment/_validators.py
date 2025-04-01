import warnings

from biocframe import BiocFrame
import biocutils as ut

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def _validate_spatial_coords_names(spatial_coords_names, spatial_coords):
    if not ut.is_list_of_type(spatial_coords_names, str):
        raise TypeError("'spatial_coords_names' is not a list of strings")

    if len(spatial_coords_names) != spatial_coords.shape[1]:
        raise ValueError(f"Expected {spatial_coords.shape[1]} names. Got {len(spatial_coords_names)} names.")


def _validate_column_data(column_data):
    if column_data is None:
        raise ValueError("'column_data' must have a column named 'sample_id'.")

    if not isinstance(column_data, BiocFrame):
        raise TypeError("'column_data' must be a BiocFrame object.")

    if "sample_id" not in column_data.columns:
        raise ValueError("'column_data' must have a column named 'sample_id'.")


def _validate_id(id):
    is_valid = isinstance(id, str) or id is True or id is None
    if not is_valid:
        raise ValueError(f"{id} must be one of [str, True, None]")


def _validate_sample_image_ids(img_data, new_sample_id, new_image_id):
    if img_data is None:
        return

    if not isinstance(img_data, BiocFrame):
        raise TypeError("`img_data` is not a BiocFrame object.")

    for row in img_data:
        data = row[1]
        if data["sample_id"] == new_sample_id and data["image_id"] == new_image_id:
            raise ValueError(f"Image with Sample ID: {new_sample_id} and Image ID: {new_image_id} already exists")

    # TODO: check if 'new_sample_id' is present in column_data['sample_id']


def _validate_spatial_coords(spatial_coords, column_data):
    if spatial_coords is None:
        return

    if not hasattr(spatial_coords, "shape"):
        raise TypeError("Spatial coordinates must be a dataframe-like object." "Does not contain a `shape` property.")

    if column_data.shape[0] != spatial_coords.shape[0]:
        raise ValueError("'spatial_coords' do not contain coordinates for all cells.")


def _validate_img_data(img_data):
    if img_data is None:
        return

    if not isinstance(img_data, BiocFrame):
        raise TypeError("'img_data' must be a BiocFrame object.")

    if img_data.shape[0] == 0:
        return

    required_columns = ["sample_id", "image_id", "data", "scale_factor"]
    if not all(column in img_data.columns for column in required_columns):
        missing = list(set(required_columns) - set(img_data.columns))
        raise ValueError(f"'img_data' is missing required columns: {missing}")


def _validate_sample_ids(column_data, img_data):
    """Ensure consistency of sample_id between img_data and column_data."""
    if img_data is None or img_data.shape[0] == 0:
        return

    img_data_sample_ids = set(img_data["sample_id"])
    column_data_sample_ids = set(column_data["sample_id"])

    if not img_data_sample_ids <= column_data_sample_ids:
        raise ValueError("All 'sample_id's in 'img_data' must be present in 'column_data['sample_id']")

    if img_data_sample_ids != column_data_sample_ids:
        warnings.warn(
            "Not all 'sample_id's in 'column_data' correspond to an entry in 'img_data'",
            UserWarning,
        )
