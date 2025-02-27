import os
import json
import numpy as np
import pandas as pd
from spatialexperiment import read_tenx_visium, SpatialExperiment, VirtualSpatialImage


def test_read_tenx_visium():
    dir = "tests/10xVisium"
    sample_ids = ["section1", "section2"]
    samples = [os.path.join(dir, sample_id, "outs") for sample_id in sample_ids]

    spe = read_tenx_visium(
        samples=samples,
        sample_ids=sample_ids,
        type="sparse",
        data="raw",
        images="lowres",
        load=False,
    )

    assert isinstance(spe, SpatialExperiment)
    assert all(
        [
            isinstance(img, VirtualSpatialImage)
            for img in spe.get_img(sample_id=True, image_id=True)
        ]
    )

    column_names = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]

    tissue_positions_paths = [
        os.path.join(sample, "spatial", "tissue_positions_list.csv") for sample in samples
    ]
    tissue_positions_list = [
        pd.read_csv(tissue_positions_path, header=None, index_col=0, names=column_names)
        for tissue_positions_path in tissue_positions_paths
    ]

    sample_ids = np.repeat(
        sample_ids,
        [len(tissue_positions_df) for tissue_positions_df in tissue_positions_list],
    )

    tissue_positions = pd.concat(tissue_positions_list, axis=0)
    tissue_positions["sample_id"] = sample_ids
    tissue_positions["in_tissue"] = tissue_positions["in_tissue"].astype(bool)

    assert np.array_equal(
        pd.crosstab(spe.column_data["sample_id"], spe.column_data["in_tissue"]).values,
        pd.crosstab(
            tissue_positions["sample_id"], tissue_positions["in_tissue"]
        ).values,
    )

    scale_factor_paths = [
        os.path.join(sample, "spatial", "scalefactors_json.json") for sample in samples
    ]
    scale_factors = []
    for scale_factor_path in scale_factor_paths:
        with open(scale_factor_path) as f:
            scale_factors.append(json.load(f)["tissue_lowres_scalef"])
    scale_factors = np.array(scale_factors)

    assert np.array_equal(spe.img_data["scale_factor"], scale_factors)
