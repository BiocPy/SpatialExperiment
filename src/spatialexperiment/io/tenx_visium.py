"""Creates a ``SpatialExperiment`` from the Space Ranger output directories for 10x Genomics Visium spatial gene expression data"""

from typing import List, Union, Optional
from warnings import warn
import os
import re
import json

from biocframe import BiocFrame
import biocutils as ut
from singlecellexperiment import read_tenx_mtx
from ..SpatialExperiment import SpatialExperiment
from .._imgutils import construct_img_data
from .._initutils import construct_spatial_coords_from_names


def read_tissue_positions(tissue_positions_path) -> "pd.DataFrame":
    """Read and parse tissue position file.

    Args:
        tissue_positions_path: The path to tissue positions file from Space Ranger output.
            Can be either 'tissue_positions.csv' or 'tissue_positions_list.csv'.

    Returns:
        A DataFrame with the tissue positions.
    """
    import pandas as pd

    column_names = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]

    has_header = "list" not in os.path.basename(tissue_positions_path)

    tissue_positions = pd.read_csv(tissue_positions_path, header=0 if has_header else None, names=column_names)
    tissue_positions = tissue_positions.set_index("barcode")
    tissue_positions["in_tissue"] = tissue_positions["in_tissue"].astype(bool)

    return tissue_positions


def read_img_data(
    path: str = ".",
    sample_ids: Optional[List[str]] = None,
    image_sources: Optional[List[str]] = None,
    scale_factors: str = None,
    load: bool = True,
) -> BiocFrame:
    """Read in images and scale factors for 10x Genomics Visium data, and return as a valid `img_data` object.

    Args:
        path:
            A path where to find one or more images.

        sample_ids:
            The `sample_id`'s for the ``SpatialExperiment`` object.

        image_sources:
            The source path(s) to the image(s).

        scale_factors:
            The .json file where to find the scale factors.

        load:
            A boolean specifying whether the image(s) should be loaded into memory? If False, will store the path/URL instead.
            Defaults to `True`.
    """
    # get sample identifiers
    if sample_ids is None:
        raise ValueError("`sample_id` mustn't be NULL.")

    if not isinstance(sample_ids, list) or not all(isinstance(s, str) for s in sample_ids):
        raise TypeError("`sample_id` must be a list of strings.")

    if len(set(sample_ids)) != len(path):
        raise ValueError("The number of unique sample_ids must match the length of path.")

    # put images into list with one element per sample
    if image_sources is None:
        image_sources = [os.path.join(p, "tissue_lowres_image.png") for p in path]

    if scale_factors is None:
        scale_factors = [os.path.join(p, "scalefactors_json.json") for p in path]

    images = [[img for img in image_sources if p in img] for p in path]

    img_data = BiocFrame({"sample_id": [], "image_id": [], "data": [], "scale_factor": []})
    for i, sample_id in enumerate(sample_ids):
        with open(scale_factors[i], "r") as f:
            curr_scale_factors = json.load(f)

        for image in images[i]:
            base_name = os.path.basename(image)
            image_name = re.sub(r"\..*$", "", base_name)
            image_id = {
                "tissue_lowres_image": "lowres",
                "tissue_hires_image": "hires",
                "detected_tissue_image": "detected",
                "aligned_fiducials": "aligned",
            }.get(image_name, None)

            scale_factor_name = {"lowres": "tissue_lowres_scalef"}.get(image_id, "tissue_hires_scalef")
            scale_factor = next(
                (value for key, value in curr_scale_factors.items() if scale_factor_name in key),
                None,
            )
            curr_image_data = construct_img_data(
                img=image, scale_factor=scale_factor, sample_id=sample_id, image_id=image_id, load=load
            )
            img_data = img_data.combine_rows(curr_image_data)

    return img_data


def read_tenx_visium(
    samples: List[Union[str, os.PathLike]],
    sample_ids: Optional[List[str]] = None,
    type: str = "HDF5",
    data: str = "filtered",
    images: List[str] = "lowres",
    load: bool = True,
):
    """Create a ``SpatialExperiment`` from the Space Ranger output directories for 10x Genomics Visium spatial gene expression data.

    Args:
        samples:
            A list of strings specifying one or more directories, each corresponding to a 10x Genomics Visium sample; if provided, names will be used as sample identifiers.

        sample_ids:
            A list of strings specifying unique sample identifiers, one for each directory specified via `samples`.

        type:
            A string specifying the type of format to read count data from. Valid values are ['auto', 'sparse', 'prefix', 'HDF5'] (see [read10xCounts](https://rdrr.io/github/MarioniLab/DropletUtils/man/read10xCounts.html)).

        data:
            A string specifying whether to read in filtered (spots mapped to tissue) or raw data (all spots). Valid values are "filtered", "raw".

        images:
            A single string or a list of strings specifying which images to include. Valid values are "lowres", "hires", "fullres", "detected", "aligned".

        load:
            A boolean specifying whether the image(s) should be loaded into memory? If False, will store the path/URL instead.
            Defaults to `True`.
    """
    # check validity of input arguments
    allowed_types = ["HDF5", "sparse", "auto", "prefix"]
    allowed_data = ["filtered", "raw"]
    allowed_images = ["lowres", "hires", "detected", "aligned"]

    if type not in allowed_types:
        raise ValueError(f"`type` must be one of {allowed_types}. got `{type}`.")

    if data not in allowed_data:
        raise ValueError(f"`data` must be one of {allowed_data}. got `{data}`")

    if isinstance(images, str):
        images = [images]

    for image in images:
        if image not in allowed_images:
            raise ValueError(f"`images` must be one of {allowed_images}. got `{image}`.")

    if sample_ids is None:
        sample_ids = [f"sample{str(i).zfill(2)}" for i in range(1, len(samples) + 1)]
    elif isinstance(sample_ids, str):
        warn(f"converting string sample_id to list: [{sample_ids}]")
        sample_ids = [sample_ids]
    elif not ut.is_list_of_type(sample_ids, str):
        raise ValueError("`sample_ids` must be a list of strings")
    elif len(set(sample_ids)) != len(samples):
        raise ValueError("`sample_ids` should contain as many unique values as `samples`")

    if isinstance(samples, str):
        warn(f"converting string samples to list: [{samples}]")
        samples = [samples]

    # add "outs/" directory if not already included
    for i, sample in enumerate(samples):
        if os.path.basename(sample) != "outs":
            samples[i] = os.path.join(sample, "outs")

    # setup file paths
    ext = ".h5" if type == "HDF5" else ""
    counts_dirs = [f"{data}_feature_bc_matrix{ext}" for _ in samples]
    counts_dir_paths = [os.path.join(sample, fn) for sample, fn in zip(samples, counts_dirs)]

    # spatial parts
    spatial_dir_paths = [os.path.join(sample, "spatial") for sample in samples]

    allowed_suffixes = [
        "",
        "_list",
    ]  # `tissue_positions_list.csv` was renamed to `tissue_positions.csv` in Space Ranger v2.0.0

    tissue_positions_paths = [
        os.path.join(spatial_dir, f"tissue_positions{suffix}.csv")
        for spatial_dir in spatial_dir_paths
        for suffix in allowed_suffixes
    ]
    tissue_positions_paths = [
        tissue_positions_path
        for tissue_positions_path in tissue_positions_paths
        if os.path.exists(tissue_positions_path)
    ]
    scale_factors_paths = [os.path.join(spatial_dir, "scalefactors_json.json") for spatial_dir in spatial_dir_paths]

    # read image data
    image_files_mapper = {
        "lowres": "tissue_lowres_image.png",
        "hires": "tissue_hires_image.png",
        "detected": "detected_tissue_image.jpg",
        "aligned": "aligned_fiducials.jpg",
    }

    image_files = [image_files_mapper[image] for image in images if image in image_files_mapper]
    image_file_paths = [
        os.path.join(spatial_dir, image_file) for spatial_dir in spatial_dir_paths for image_file in image_files
    ]

    missing_files = [not os.path.exists(image_file_path) for image_file_path in image_file_paths]

    if all(missing_files):
        raise FileNotFoundError(f"No matching files found for 'images={images}'")

    elif any(missing_files):
        print(
            "Skipping missing images\n  "
            + "\n  ".join(
                image_file_path for image_file_path, missing in zip(image_file_paths, missing_files) if missing
            )
        )
        image_file_paths = [
            image_file_path for image_file_path, missing in zip(image_file_paths, missing_files) if not missing
        ]

    image = read_img_data(
        path=samples,
        sample_ids=sample_ids,
        image_sources=image_file_paths,
        scale_factors=scale_factors_paths,
        load=load,
    )

    spes = []
    for i, counts_dir_path in enumerate(counts_dir_paths):
        sce = read_tenx_mtx(counts_dir_path)
        tissue_positions = read_tissue_positions(tissue_positions_paths[i])

        barcodes = sce.column_data["barcode"]
        sce = sce.set_column_names(barcodes)

        obs = list(set(sce.col_names).intersection(set(tissue_positions.index)))
        sce = sce[:, obs]

        tissue_positions = tissue_positions.loc[obs, :]
        tissue_positions["sample_id"] = sample_ids[i]
        spatial_coords, column_data = construct_spatial_coords_from_names(
            spatial_coords_names=["pxl_col_in_fullres", "pxl_row_in_fullres"], column_data=tissue_positions
        )

        spe = SpatialExperiment(
            assays=sce.assays,
            row_data=BiocFrame({"symbol": sce.row_data["gene_symbols"]}),
            column_data=column_data,
            spatial_coords=spatial_coords,
        )
        spes.append(spe)

    spe_combined = ut.combine_columns(*spes)
    spe_combined.img_data = image

    return spe_combined
