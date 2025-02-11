import numpy as np
from biocframe import BiocFrame
from xarray import DataArray, DataTree, Variable

from .SpatialImage import construct_spatial_image_class


def process_dataset_images(dt: DataTree, root_name: str) -> BiocFrame:
    """Processes image-related attributes from a :py:class:`~xarray.DataTree` object and compiles them into a :py:class:`~biocframe.BiocFrame`. The resulting BiocFrame adheres to the standards required for a ``SpatialExperiment``'s `img_data`.

    Args:
        dt: A DataTree object containing datasets with image data.
        root_name: An identifier for the highest ancestor of the DataTree to which this subtree belongs.

    Returns:
        A BiocFrame that conforms to the standards of a ``SpatialExperiment``'s `img_data`.
    """
    img_data = BiocFrame(
        {
            "sample_id": [],
            "image_id": [],
            "data": [],
            "scale_factor": []
        }
    )
    for var_name, obj in dt.dataset.items():
        if isinstance(obj, (DataArray, Variable)):
            var = obj
        else:
            dims, data, *optional = obj
            attrs = optional[0] if optional else None
            var = Variable(dims=dims, data=data, attrs=attrs)

        scale_factor = var.attrs.get("scale_factor", np.nan)
        spi = construct_spatial_image_class(np.array(var))
        img_row = BiocFrame(
            {
                "sample_id": [f"{root_name}::{dt.name}"],
                "image_id": [var_name],
                "data": [spi],
                "scale_factor": [scale_factor]
            }
        )
        img_data = img_data.combine_rows(img_row)

    return img_data        


def build_img_data(dt: DataTree, root_name: str):
    """Recursively compiles image data from a :py:class:`~xarray.DataTree` into a :py:class:`~biocframe.BiocFrame.BiocFrame` structure.

    This function traverses a `DataTree`, extracting image-related attributes from each dataset and compiling them into a `BiocFrame`. It processes the parent dataset and recursively handles dataset(s) from child nodes. The resulting `BiocFrame` adheres to the standards required for a ``SpatialExperiment``'s `img_data`.

    The following conditions are assumed:
    - `DataTree.name` will be used as the `sample_id`.
    - The keys of `dt.dataset.data_vars` will be used as the `image_id`'s of each image.
    - The `scale_factor` is extracted from the attributes of the objects in `dt.dataset.data_vars`.

    Args:
        dt: A DataTree object containing datasets with image data.
        root_name: An identifier for the highest ancestor of the DataTree to which this subtree belongs.

    Returns:
        A BiocFrame containing compiled image data for the entire DataTree.
    """
    if len(dt.children) == 0:
        return process_dataset_images(dt, root_name)

    parent_img_data = process_dataset_images(dt, root_name)

    for key, child in dt.children.items():
        child_img_data = build_img_data(child, root_name)
        parent_img_data = parent_img_data.combine_rows(child_img_data)

    return parent_img_data
