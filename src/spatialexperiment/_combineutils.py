from warnings import warn
from copy import deepcopy
import itertools
import biocutils as ut


def merge_spe_cols(cols):
    sample_ids = list(itertools.chain.from_iterable(_cols["sample_id"] for _cols in cols))

    if len(set(sample_ids)) != len(sample_ids):
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
