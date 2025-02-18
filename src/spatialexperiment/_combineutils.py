from warnings import warn
import itertools
import biocutils as ut


def merge_spe_cols(cols):
    sample_ids = list(itertools.chain.from_iterable(_cols["sample_id"] for _cols in cols))

    if len(set(sample_ids)) != len(sample_ids):
        warn(
            "'sample_id's are duplicated across 'SpatialExperiment' objects to 'combine_columns'; appending sample indices."
        )
        _all_cols = []
        for i, _cols in enumerate(cols):
            _cols_copy = _cols.copy()
            _cols_copy["sample_id"] = _cols_copy["sample_id"] + f".{i}"
            _all_cols.append(_cols_copy)
    else:
        _all_cols = cols

    _new_cols = ut.combine_rows(*_all_cols) 
    return _new_cols
