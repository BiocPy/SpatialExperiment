"""Creates a ``SpatialExperiment`` from the Space Ranger output directories for 10x Genomics Visium spatial gene expression data"""

from typing import List, Union, Optional
import os
import re
import json

from spatialexperiment import SpatialExperiment


def read_img_data(
    path: str = ".",
    sample_id: Optional[List[str]] = None,
    image_sources: Optional[List[str]] = None,
    scale_factors: str = None,
    load: bool = False
):
    """Read in images and scale factors for 10x Genomics Visium data, and return as a valid `img_data` object.
    
    Args:
        path: A path where to find one or more images.

        sample_id: The `sample_id` for the ``SpatialExperiment`` object.

        image_sources: The source path(s) to the image(s).

        scale_factors: The .json file where to find the scale factors.

        load: A boolean specifying whether the image(s) should be loaded into memory? If False, will store the path/URL instead.
    """
    # get sample identifiers
    if sample_id is None:
        raise ValueError("`sample_id` mustn't be NULL.")
    
    if not isinstance(sample_id, list) or not all(isinstance(s, str) for s in sample_id):
        raise TypeError("`sample_id` must be a list of strings.")
    
    if len(set(sample_id)) != len(path):
        raise ValueError("The number of unique sample_ids must match the length of path.")

    # put images into list with one element per sample
    if image_sources is None:
        image_sources = [os.path.join(p, "tissue_lowres_image.png") for p in path]
    
    if scale_factors is None:
        scale_factors = [os.path.join(p, "scalefactors_json.json") for p in path]
    
    images = [[img for img in image_sources if p in img] for p in path]

    for i, sid in enumerate(sample_id):
        with open(scale_factors[i], 'r') as f:
            sfs = json.load(f)
        
        for img in images[i]:
            # get image identifier
            base_name = os.path.basename(img)
            img_nm = re.sub(r"\..*$", "", base_name)
            # TODO: left off at line 65 of readImgData


def read_10x_visium(
    samples: List[Union[str, os.PathLike]],
    sample_id: Optional[List[str]],
    type: str = "HDF5",
    data: str = ["filtered", "raw"],
    images: List[str] = "lowres",
    load: bool = True
):
    """Creates a ``SpatialExperiment`` from the Space Ranger output directories for 10x Genomics Visium spatial gene expression data

    Args:
        samples: A list of strings specifying one or more directories, each corresponding to a 10x Genomics Visium sample; if provided, names will be used as sample identifiers.
        
        sample_id: A list of strings specifying unique sample identifiers, one for each directory specified via `samples`.

        type: A string specifying the type of format to read count data from. Valid values are ['auto', 'sparse', 'prefix', 'HDF5'] (see [read10xCounts](https://rdrr.io/github/MarioniLab/DropletUtils/man/read10xCounts.html)).

        data: A string specifying whether to read in filtered (spots mapped to tissue) or raw data (all spots). Valid values are "filtered", "raw".

        images: A single string or a list of strings specifying which images to include. Valid values are "lowres", "hires", "fullres", "detected", "aligned".

        load: A boolean specifying whether the image(s) should be loaded into memory? If False, will store the path/URL instead.
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
    for img in images:
        if img not in allowed_images:
            raise ValueError(f"`images` must be one of {allowed_images}. got `{img}`.")

    if not isinstance(sample_id, list) or len(set(sample_id)) != len(samples):
        raise ValueError("`sample_id` should contain as many unique values as `samples`.")

    # add "outs/" directory if not already included
    for i, sample in enumerate(samples):
        if os.path.basename(sample) != "outs":
            samples[i] = os.path.join(sample, "outs")

    # setup file paths
    extension = ".h5" if type == "HDF5" else ""
    fns = [f"{data}_feature_bc_matrix{extension}" for _ in samples]
    counts = [os.path.join(sample, fn) for sample, fn in zip(samples, fns)]

    # spatial parts
    spatial_dirs = [os.path.join(sample, "spatial") for sample in samples]

    suffixes = ["", "_list"]  # `tissue_positions_list.csv` was renamed to `tissue_positions.csv` in Space Ranger v2.0.0

    tps = [
        os.path.join(spatial_dir, f"tissue_positions{suffix}.csv")
        for spatial_dir in spatial_dirs
        for suffix in suffixes
    ]
    tps = [tp for tp in tps if os.path.exists(tp)]
    sfs = [os.path.join(spatial_dir, "scalefactors_json.json") for spatial_dir in spatial_dirs]

    # read image data
    img_fns_mapper = {
        "lowres": "tissue_lowres_image.png",
        "hires": "tissue_hires_image.png",
        "detected": "detected_tissue_image.jpg",
        "aligned": "aligned_fiducials.jpg"        
    }

    img_fns = [img_fns_mapper[img] for img in images if img in img_fns_mapper]
    img_fns = [os.path.join(spatial_dir, img_fn) for spatial_dir in spatial_dirs for img_fn in img_fns]

    missing_files = [not os.path.exists(img_fn) for img_fn in img_fns]

    if all(missing_files):
        raise FileNotFoundError(f"No matching files found for 'images={imgs}'")

    elif any(missing_files):
        print("Skipping missing images\n  " + "\n  ".join(
            img_fn for img_fn, missing in zip(img_fns, missing_files) if missing
        ))
        img_fns = [img_fn for img_fn, missing in zip(img_fns, missing_files) if not missing]

    img = read_img_data(samples, sample_id, img_fns, sfs, load)

    