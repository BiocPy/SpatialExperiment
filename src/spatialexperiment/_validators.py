import biocframe
import pandas as pd
from type_checks import is_list_of_type
from summarizedexperiment._frameutils import _sanitize_frame

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def _validate_spatial_coords_names(spatial_coords_names, spatial_coords):
    if not is_list_of_type(spatial_coords_names, str):
        raise TypeError("'spatial_coords_names' is not a list of strings")

    if len(spatial_coords_names) != spatial_coords.shape[1]:
        raise ValueError(
            f"Expected {spatial_coords.shape[1]} names. Got {len(spatial_coords_names)} names."
        )


def _validate_column_data(column_data, img_data):
    error_message = "'column_data' must have a column named 'sample_id'."

    if column_data is None:
        raise ValueError(error_message)

    if not isinstance(column_data, (pd.DataFrame, biocframe.BiocFrame)):
        raise TypeError("'column_data' must be a DataFrame or BiocFrame object.")
    
    if "sample_id" not in column_data.columns:
        raise ValueError(error_message)

    num_unique_sample_ids = len(img_data["sample_id"].unique())
    num_unique_sample_ids_provided = len(column_data["sample_id"].unique())

    if num_unique_sample_ids != num_unique_sample_ids_provided:
        raise ValueError(f"Number of unique 'sample_id's is {num_unique_sample_ids}, but {num_unique_sample_ids_provided} were provided.")


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
            raise ValueError(
                f"Image with Sample ID: {new_sample_id} and Image ID: {new_image_id} already exists"
            )


def _validate_spatial_coords(spatial_coords, column_data):
    if spatial_coords is None:
        return

    if not isinstance(spatial_coords, (pd.DataFrame, biocframe.BiocFrame)):
        raise TypeError("'spatial_coords' must be a DataFrame or BiocFrame object.")

    if column_data.shape[0] != spatial_coords.shape[0]:
        raise ValueError("'spatial_coords' do not contain coordinates for all cells.")


def _validate_img_data(img_data):
    if img_data is None:
        return

    if not isinstance(img_data, (pd.DataFrame, biocframe.BiocFrame)):
        raise TypeError("'img_data' must be a DataFrame or BiocFrame object.")

    required_columns = ["sample_id", "image_id", "data", "scale_factor"]
    if not all(column in img_data.columns for column in required_columns):
        missing = list(set(required_columns) - set(img_data.columns))
        raise ValueError(f"'img_data' is missing required columns: {missing}")
