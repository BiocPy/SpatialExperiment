from __future__ import annotations
from typing import List, Tuple

from warnings import warn
from copy import deepcopy
import itertools

from biocframe import BiocFrame
import biocutils as ut
from spatialexperiment import SpatialExperiment


def _append_indices_to_samples(bframes: List[BiocFrame]) -> List[BiocFrame]:
    """Append indices to sample IDs for a list of `BiocFrames`.

    For each `BiocFrame`, appends an index to all sample IDs to ensure uniqueness
    across multiple frames.

    Args:
        bframes: List of `BiocFrame` objects containing sample IDs.

    Returns:
        List of `BiocFrame`s with modified sample IDs.
    """
    modified_bframes = []
    for i, bframe in enumerate(bframes, start=1):
        bframe_copy = deepcopy(bframe)
        bframe_copy["sample_id"] = [f"{sample_id}_{i}" for sample_id in bframe_copy["sample_id"]]
        modified_bframes.append(bframe_copy)
    return modified_bframes


def merge_spatial_frames(x: List[SpatialExperiment], relaxed: bool = False) -> Tuple[BiocFrame, BiocFrame]:
    """Merge column data and image data from multiple ``SpatialExperiment`` objects.

    If duplicate sample IDs exist across objects, appends indices to make them unique.
    Sample IDs in column data determine the uniqueness check as they are the superset
    of IDs in image data.

    Args:
        x: List of ``SpatialExperiment`` objects
        relaxed: If `True`, allows frames with different columns to be combined.
            Absent columns in any frame are filled with appropriate placeholder values.
            Defaults to `False`.

    Returns:
        A tuple with the merged column data and image data.
    """
    cols = [y._cols for y in x]
    img_datas = [y._img_data for y in x]

    expected_unique = sum([len(set(_cols["sample_id"])) for _cols in cols])
    all_sample_ids = list(itertools.chain.from_iterable(_cols["sample_id"] for _cols in cols))

    if len(set(all_sample_ids)) < expected_unique:
        warn(
            "'sample_id's are duplicated across 'SpatialExperiment' objects to 'combine_columns'; appending sample indices."
        )
        modified_columns = _append_indices_to_samples(cols)
        modified_img_data = _append_indices_to_samples(img_datas)
    else:
        modified_columns = cols
        modified_img_data = img_datas

    if relaxed:
        _new_cols = ut.relaxed_combine_rows(*modified_columns)
        _new_img_data = ut.relaxed_combine_rows(*modified_img_data)
    else:
        _new_cols = ut.combine_rows(*modified_columns)
        _new_img_data = ut.combine_rows(*modified_img_data)
    return _new_cols, _new_img_data


def merge_spatial_coordinates(spatial_coords: List[BiocFrame], relaxed: bool = False) -> BiocFrame:
    """Merge spatial coordinates from multiple frames.

    Args:
        spatial_coords: List of `BiocFrame`s containing spatial coordinates.
        relaxed: If `True`, allows frames with different columns to be combined.
            Absent columns in any frame are filled with appropriate placeholder values.
            Defaults to `False`.

    Returns:
        A merged BiocFrame containing all spatial coordinates.

    Raises:
        ValueError: If spatial coordinates have different numbers of columns.

    Warns:
        If dimension names are not consistent across all `BiocFrame`s.
    """
    first_shape = spatial_coords[0].shape[1]
    if not all(coords.shape[1] == first_shape for coords in spatial_coords):
        raise ValueError("Not all 'spatial_coords' have the same number of columns.")

    first_columns = spatial_coords[0].columns
    if not all(coords.columns == first_columns for coords in spatial_coords):
        warn("Not all 'spatial_coords' have the same dimension names.")

    if relaxed:
        _new_spatial_coords = ut.relaxed_combine_rows(*spatial_coords)
    else:
        _new_spatial_coords = ut.combine_rows(*spatial_coords)
    return _new_spatial_coords
