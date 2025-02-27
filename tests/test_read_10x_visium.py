import os
from spatialexperiment import read_tenx_visium


def test_read_10x_visium():
    dir = "tests/10xVisium"
    sample_ids = ["section1", "section2"]
    samples = [os.path.join(dir, sample_id, "outs") for sample_id in sample_ids]

    spe = read_tenx_visium(
        samples=samples,
        sample_ids=sample_ids,
        type="sparse",
        data="raw",
        images="lowres",
        load=False
    )