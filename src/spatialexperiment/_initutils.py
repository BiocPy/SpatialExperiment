from copy import deepcopy
from typing import List, Tuple

from biocframe import BiocFrame
from PIL import Image
from .SpatialImage import construct_spatial_image_class
from summarizedexperiment._frameutils import _sanitize_frame

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def construct_spatial_coords_from_names(
    spatial_coords_names: List[str], column_data: BiocFrame
) -> Tuple[BiocFrame, BiocFrame]:
    """Construct the `spatial_coords` dataframe from names.

    Args:
        spatial_coords_names:
            A list of strings of column names from `column_data` containing spatial coordinates.

        column_data:
            The sample data.

    Returns:
        A tuple containing two `BiocFrame` objects:
        - The first `BiocFrame` contains columns of spatial coordinates.
        - The second `BiocFrame` is a subset of the original `column_data`, with the spatial coordinate columns removed.
    """
    if spatial_coords_names is None:
        raise ValueError("No spatial coordinate names were provided.")

    current_column_data = _sanitize_frame(column_data, num_rows=column_data.shape[1])

    missing_names = [name for name in spatial_coords_names if name not in current_column_data.column_names]
    if missing_names:
        raise ValueError(
            f"The following names in `spatial_coords_names` are missing from `column_data`: {missing_names}"
        )

    spatial_coords = deepcopy(current_column_data[:, spatial_coords_names])

    column_data_subset = deepcopy(
        current_column_data[
            :,
            [col for col in current_column_data.column_names if col not in spatial_coords_names],
        ]
    )

    return spatial_coords, column_data_subset


def construct_img_data(
    sample_id: str,
    image_id: str,
    image_sources: List[str],
    scale_factors: List[float],
    load_image: bool = False,
) -> BiocFrame:
    """Construct the image data for a `SpatialExperiment`.

    Args:
        sample_id:
            The sample id.

        image_id:
            The image id.

        image_sources:
            The file paths to the images. Must be the same length as `scale_factors`.

        scale_factors:
            The scaling factors associated with the images. Must be the same length as
            `image_sources`.

        load_image:
            Whether to load the images into memory. Defaults to False.

    Returns:
        A `BiocFrame` representing the image data for a `SpatialExperiment`.
    """
    if not len(image_id) == len(image_sources) == len(scale_factors):
        raise ValueError("'image_id', 'image_sources' and 'scale_factors' are not the same length.")

    spis = []
    for image_source in image_sources:
        result = Image.open(image_source) if load_image else image_source
        spi = construct_spatial_image_class(result)
        spis.append(spi)

    img_data = {
        "sample_id": sample_id,
        "image_id": image_id,
        "data": spis,
        "scale_factor": scale_factors,
    }

    return BiocFrame(img_data)
