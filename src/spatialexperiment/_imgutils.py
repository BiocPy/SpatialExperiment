from typing import Union, List

import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from PIL import Image
from biocframe import BiocFrame
from .SpatialImage import construct_spatial_image_class

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def read_image(input_image):
    """Read image from PIL Image, file path, or URL.

    Args:
        input_image: PIL Image, string path to local file, or URL string.

    Returns:
        The loaded image.

    Raises:
        TypeError: If input is not PIL Image, path string, or URL string.
    """
    import requests

    if isinstance(input_image, Image.Image):
        return input_image

    if isinstance(input_image, (str, Path)):
        is_url = urlparse(str(input_image)).scheme in ("http", "https", "ftp")
        if is_url:
            response = requests.get(input_image)
            return Image.open(BytesIO(response.content))
        else:
            return Image.open(input_image)

    raise TypeError(f"Expected PIL Image, path, or URL. Got {type(input_image)}")


def construct_img_data(
    img: Union[str, os.PathLike], scale_factor: str, sample_id: str, image_id: str, load: bool = True
) -> BiocFrame:
    """
    Construct an image data dataframe.

    Args:
        img:
            A path or url to the image file.

        scale_factor:
            The scale factor associated with the image.

        sample_id:
            A unique identifier for the sample to which the image belongs.

        image_id:
            A unique identifier for the image itself.

        load:
            A boolean specifying whether the image(s) should be loaded into memory? If False, will store the path/URL instead.
            Defaults to `True`.

    Returns:
        The image data.
    """
    if load:
        img = read_image(img)

    spi = construct_spatial_image_class(img)
    return BiocFrame({"sample_id": [sample_id], "image_id": [image_id], "data": [spi], "scale_factor": [scale_factor]})


def get_img_idx(
    img_data: BiocFrame,
    sample_id: Union[str, bool, None] = None,
    image_id: Union[str, bool, None] = None,
) -> List[int]:
    """
    Retrieve the row index/indices of image(s) with matching 'sample_id' and 'image_id' from the 'img_data'.

    Args:
        img_data:
            The data from which to retrieve rows.

        sample_id:
            - `sample_id=True`: Matches all samples.
            - `sample_id=None`: Matches the first sample.
            - `sample_id="<str>"`: Matches a sample by its id.

        image_id:
            - `image_id=True`: Matches all images for the specified sample(s).
            - `image_id=None`: Matches the first image for the sample(s).
            - `image_id="<str>"`: Matches image(s) by its(their) id.

    Returns:
        The row index/indices of image(s) with matchine 'sample_id' and 'image_id' from the 'img_data'.
    """
    sample_ids = np.array(img_data["sample_id"])
    image_ids = np.array(img_data["image_id"])
    if isinstance(sample_id, str) and isinstance(image_id, str):
        sid = sample_ids == sample_id
        iid = image_ids == image_id
    elif sample_id is True and image_id is True:
        sid = iid = np.full(len(img_data), True)
    elif sample_id is None and image_id is None:
        sid = iid = np.eye(len(img_data))[0, :]
    elif isinstance(sample_id, str) and image_id is True:
        sid = sample_ids == sample_id
        iid = np.full(len(img_data), True)
    elif sample_id is True and isinstance(image_id, str):
        sid = np.full(len(img_data), True)
        iid = image_ids == image_id
    elif isinstance(sample_id, str) and image_id is None:
        sid = sample_ids == sample_id
        iid = np.zeros(len(img_data))
        iid[np.where(sid)[0][0]] = 1
    elif sample_id is None and isinstance(image_id, str):
        first_sid = img_data["sample_id"][0]
        sid = sample_ids == first_sid
        iid = image_ids == image_id
    elif sample_id is True and image_id is None:
        sid = np.full(len(img_data), True)
        iid = [img_data["sample_id"].index(x) for x in set(img_data["sample_id"])]
        iid = np.eye(len(img_data))[iid, :].sum(axis=0)
    elif sample_id is None and image_id is True:
        first_sid = img_data["sample_id"][0]
        sid = sample_ids == first_sid
        iid = np.full(len(img_data), True)

    mask = sid.astype(bool) & iid.astype(bool)
    if not any(mask):
        raise ValueError(
            f"No 'imgData' entry(ies) matched the specified image_id = '{image_id}' and sample_id = '{sample_id}'"
        )

    return np.where(mask)[0].tolist()
