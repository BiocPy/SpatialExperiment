from warnings import warn
from copy import deepcopy
import itertools
import biocutils as ut


def merge_spe_cols(cols):
    num_unique = sum([len(set(_cols["sample_id"])) for _cols in cols])

    sample_ids = list(itertools.chain.from_iterable(_cols["sample_id"] for _cols in cols))

    if len(set(sample_ids)) < num_unique:
        warn(
            "'sample_id's are duplicated across 'SpatialExperiment' objects to 'combine_columns'; appending sample indices."
        )
        _all_cols = []
        for i, _cols in enumerate(cols, start=1):
            _cols_copy = deepcopy(_cols)
            _cols_copy["sample_id"] = [f"{sample_id}_{i}" for sample_id in _cols_copy["sample_id"]]
            _all_cols.append(_cols_copy)
    else:
        _all_cols = cols

    _new_cols = ut.combine_rows(*_all_cols) 
    return _new_cols


def merge_spe_spatial_coords(spatial_coords):
    first_shape = spatial_coords[0].shape[1]
    if not all(coords.shape[1] == first_shape for coords in spatial_coords):
        raise ValueError("Not all 'spatial_coords' have the same number of columns.")

    first_columns = spatial_coords[0].columns
    if not all(coords.columns == first_columns for coords in spatial_coords):
        warn(
            "Not all 'spatial_coords' have the same dimension names."
        )

    _new_spatial_coords = ut.combine_rows(*spatial_coords)
    return _new_spatial_coords
