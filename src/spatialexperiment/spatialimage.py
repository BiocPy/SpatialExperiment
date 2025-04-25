import shutil
import tempfile
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse
from warnings import warn

import biocutils as ut
import numpy as np
import requests
from PIL import Image, ImageChops

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche, keviny2"
__license__ = "MIT"


# Keeping the same names as the R classes
class VirtualSpatialImage(ABC):
    """Base class for spatial images."""

    def __init__(self, metadata: Optional[dict] = None):
        self._metadata = metadata if metadata is not None else {}

    #########################
    ######>> Equality <<#####
    #########################

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self.metadata == other.metadata

    def __hash__(self):
        # Note: This exists primarily to support lru_cache.
        # Generally, these classes are mutable and shouldn't be used as dict keys or in sets.
        return hash(frozenset(self._metadata.items()))

    ###########################
    ######>> metadata <<#######
    ###########################

    def get_metadata(self) -> dict:
        """
        Returns:
            Dictionary of metadata for this object.
        """
        return self._metadata

    def set_metadata(self, metadata: dict, in_place: bool = False) -> "VirtualSpatialImage":
        """Set additional metadata.

        Args:
            metadata:
                New metadata for this object.

            in_place:
                Whether to modify the ``VirtualSpatialImage`` in place.

        Returns:
            A modified ``VirtualSpatialImage`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        if not isinstance(metadata, dict):
            raise TypeError(f"`metadata` must be a dictionary, provided {type(metadata)}.")
        output = self._define_output(in_place)
        output._metadata = metadata
        return output

    @property
    def metadata(self) -> dict:
        """Alias for :py:attr:`~get_metadata`."""
        return self.get_metadata()

    @metadata.setter
    def metadata(self, metadata: dict):
        """Alias for :py:attr:`~set_metadata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'metadata' is an in-place operation, use 'set_metadata' instead",
            UserWarning,
        )
        self.set_metadata(metadata, in_place=True)

    ############################
    ######>> img props <<#######
    ############################

    def get_dimensions(self) -> Tuple[int, int]:
        """Get image dimensions (width, height)."""
        img = self.img_raster()
        return img.size

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Alias for :py:meth:`~get_dimensions`."""
        return self.get_dimensions()

    ############################
    ######>> img utils <<#######
    ############################

    @abstractmethod
    def img_source(self, as_path: bool = False) -> Union[str, None]:
        """Get the source of the image.

        Args:
            as_path: If True, returns path as string. Defaults to False.

        Returns:
            Source path/URL of the image, or None if loaded in memory.
        """
        pass

    @abstractmethod
    def img_raster(self) -> Image.Image:
        """Get the image as a PIL Image object."""
        pass

    def rotate_img(self, degrees: float = 90) -> "LoadedSpatialImage":
        """Rotate image by specified degrees clockwise."""
        img = self.img_raster()

        # PIL rotates counter-clockwise
        rotated = img.rotate(-degrees, expand=True)
        return LoadedSpatialImage(rotated)

    def mirror_img(self, axis: str = "h") -> "LoadedSpatialImage":
        """Mirror image horizontally or vertically."""
        img = self.img_raster()

        if axis == "h":
            mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif axis == "v":
            mirrored = img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError("axis must be 'h' or 'v'")

        return LoadedSpatialImage(mirrored)


def _sanitize_loaded_image(image):
    if isinstance(image, np.ndarray):
        _result = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        _result = image
    else:
        raise TypeError("image must be PIL Image or numpy array")

    return _result


class LoadedSpatialImage(VirtualSpatialImage):
    """Class for images loaded into memory."""

    def __init__(self, image: Union[Image.Image, np.ndarray], metadata: Optional[dict] = None):
        """Initialize the object.

        Args:
            image:
                Image represented as a :py:class:`~numpy.ndarray` or :py:class:`~PIL.Image.Image`.

            metadata:
                Additional image metadata. Defaults to None.
        """
        super().__init__(metadata=metadata)

        self._image = _sanitize_loaded_image(image)

    def _define_output(self, in_place: bool = False) -> "LoadedSpatialImage":
        if in_place is True:
            return self
        else:
            return self.__copy__()

    #########################
    ######>> Equality <<#####
    #########################

    def __eq__(self, other) -> bool:
        diff = ImageChops.difference(self.image, other.image)

        return super().__eq__(other) and not diff.getbbox()

    def __hash__(self):
        return hash((super().__hash__(), self._image.tobytes()))

    #########################
    ######>> Copying <<######
    #########################

    def __deepcopy__(self, memo=None, _nil=[]):
        """
        Returns:
            A deep copy of the current ``LoadedSpatialImage``.
        """
        from copy import deepcopy

        _img_copy = deepcopy(self._image)
        _metadata_copy = deepcopy(self.metadata)

        current_class_const = type(self)
        return current_class_const(
            imaage=_img_copy,
            metadata=_metadata_copy,
        )

    def __copy__(self):
        """
        Returns:
            A shallow copy of the current ``LoadedSpatialImage``.
        """
        current_class_const = type(self)
        return current_class_const(
            image=self._image,
            metadata=self._metadata,
        )

    def copy(self):
        """Alias for :py:meth:`~__copy__`."""
        return self.__copy__()

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self) -> str:
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}"
        output += ", image=" + self._image.__repr__()
        if len(self._metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self._metadata)
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"
        output += f"image: ({self._image})\n"
        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output

    ############################
    ######>> img props <<#######
    ############################

    def get_image(self) -> Image.Image:
        """Get the image as a PIL Image object."""

        return self._image

    def set_image(self, image: Union[Image.Image, np.ndarray], in_place: bool = False) -> "LoadedSpatialImage":
        """Set new image.

        Args:
            image:
                Image represented as a :py:class:`~numpy.ndarray` or :py:class:`~PIL.Image.Image`.

            in_place:
                Whether to modify the ``LoadedSpatialImage`` in place. Defaults to False.

        """
        _out = self._define_output(in_place=in_place)
        _out._image = _sanitize_loaded_image(image)
        return _out

    @property
    def image(self) -> Image.Image:
        """Alias for :py:meth:`~get_image`."""
        return self.get_image()

    @image.setter
    def image(self, image: Union[Image.Image, np.ndarray]):
        """Alias for :py:attr:`~set_image` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'image' is an in-place operation, use 'set_image' instead",
            UserWarning,
        )
        return self.set_image(image=image, in_place=True)

    def img_source(self, as_path: bool = False) -> None:
        """Get the source of the loaded image.

        Returns:
            Always returns None.
        """
        return None

    ############################
    ######>> img utils <<#######
    ############################

    def img_raster(self) -> Image.Image:
        return self._image


def _sanitize_path(path):
    _path = Path(path).resolve()
    if not _path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    return _path


class StoredSpatialImage(VirtualSpatialImage):
    """Class for images stored on local filesystem."""

    def __init__(self, path: Union[str, Path], metadata: Optional[dict] = None):
        """Initialize the object.

        Args:
            path:
                Path to the image file.

            metadata:
                Additional image metadata. Defaults to None.
        """
        super().__init__(metadata=metadata)

        self._path = _sanitize_path(path)

    def _define_output(self, in_place: bool = False) -> "LoadedSpatialImage":
        if in_place is True:
            return self
        else:
            return self.__copy__()

    #########################
    ######>> Equality <<#####
    #########################

    def __eq__(self, other):
        return super().__eq__(other) and self.path == other.path

    def __hash__(self):
        return hash((super().__hash__(), str(self._path)))

    #########################
    ######>> Copying <<######
    #########################

    def __deepcopy__(self, memo=None, _nil=[]):
        """
        Returns:
            A deep copy of the current ``StoredSpatialImage``.
        """
        from copy import deepcopy

        _path_copy = deepcopy(self._path)
        _metadata_copy = deepcopy(self.metadata)

        current_class_const = type(self)
        return current_class_const(
            path=_path_copy,
            metadata=_metadata_copy,
        )

    def __copy__(self):
        """
        Returns:
            A shallow copy of the current ``StoredSpatialImage``.
        """
        current_class_const = type(self)
        return current_class_const(
            path=self._path,
            metadata=self._metadata,
        )

    def copy(self):
        """Alias for :py:meth:`~__copy__`."""
        return self.__copy__()

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self) -> str:
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}"
        output += ", path=" + str(self._path)
        if len(self._metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self._metadata)
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"
        output += f"path: ({str(self._path)})\n"
        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output

    #############################
    ######>> path props <<#######
    #############################

    def get_path(self) -> Path:
        """Get the path to the image file."""
        return self._path

    def set_path(self, path: Union[str, Path], in_place: bool = False) -> "StoredSpatialImage":
        """Update the path to the image file.

        Args:
            path:
                New path for this image.

            in_place:
                Whether to modify the ``StoredSpatialImage`` in place.

        Returns:
            A modified ``StoredSpatialImage`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        new_path = _sanitize_path(path)

        _out = self._define_output(in_place=in_place)
        _out._path = new_path
        return _out

    @property
    def path(self) -> Path:
        """Alias for :py:meth:`~get_path`."""
        return self.get_path()

    @path.setter
    def path(self, path: Union[str, Path]):
        """Alias for :py:attr:`~set_path` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'path' is an in-place operation, use 'set_path' instead",
            UserWarning,
        )
        return self.set_path(path=path, in_place=True)

    def img_source(self, as_path: bool = False) -> str:
        """Get the source path of the image.

        Args:
            as_path: If True, returns string path. Defaults to False.

        Returns:
            Path to the image.
        """
        return str(self._path) if as_path is True else self._path

    ############################
    ######>> img utils <<#######
    ############################

    # Simple in-memory cache
    @lru_cache(maxsize=32)
    def img_raster(self) -> Image.Image:
        """Load and cache the image."""
        return Image.open(self._path)


def _validate_url(url):
    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"Invalid URL: {url}")


class RemoteSpatialImage(VirtualSpatialImage):
    """Class for remotely hosted images."""

    def __init__(self, url: str, metadata: Optional[dict] = None, validate: bool = True):
        """Initialize the object.

        Args:
            url:
                URL to the image file.

            metadata:
                Additional image metadata. Defaults to None.

            validate:
                Whether to validate if the URL is valid. Defaults to True.
        """
        super().__init__(metadata=metadata)

        self._url = url
        self._cache_dir = Path(tempfile.gettempdir()) / "spatial_image_cache"
        self._cache_dir.mkdir(exist_ok=True)

        if validate:
            _validate_url(url)

    def _define_output(self, in_place: bool = False) -> "RemoteSpatialImage":
        if in_place is True:
            return self
        else:
            return self.__copy__()

    #########################
    ######>> Equality <<#####
    #########################

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.url == other.url

    def __hash__(self):
        return hash((super().__hash__(), self._url))

    #########################
    ######>> Copying <<######
    #########################

    def __deepcopy__(self, memo=None, _nil=[]):
        """
        Returns:
            A deep copy of the current ``RemoteSpatialImage``.
        """
        from copy import deepcopy

        _url_copy = deepcopy(self._url)
        _metadata_copy = deepcopy(self.metadata)

        current_class_const = type(self)
        return current_class_const(
            url=_url_copy,
            metadata=_metadata_copy,
        )

    def __copy__(self):
        """
        Returns:
            A shallow copy of the current ``RemoteSpatialImage``.
        """
        current_class_const = type(self)
        return current_class_const(
            url=self._url,
            metadata=self._metadata,
        )

    def copy(self):
        """Alias for :py:meth:`~__copy__`."""
        return self.__copy__()

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self) -> str:
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}"
        output += ", url=" + self._url
        if len(self._metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self._metadata)
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"
        output += f"url: ({self._url})\n"
        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output

    ############################
    ######>> url props <<#######
    ############################

    def get_url(self) -> str:
        """Get the url to the image file."""
        return self._url

    def set_url(self, url: str, in_place: bool = False) -> "RemoteSpatialImage":
        """Update the url to the image file.

        Args:
            url:
                New URL for this image.

            in_place:
                Whether to modify the ``RemoteSpatialImage`` in place.

        Returns:
            A modified ``RemoteSpatialImage`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        _validate_url(url)

        _out = self._define_output(in_place=in_place)
        _out.url = url
        return _out

    @property
    def url(self) -> Path:
        """Alias for :py:meth:`~get_url`."""
        return self.get_url()

    @url.setter
    def url(self, url: Union[str, Path]):
        """Alias for :py:attr:`~set_url` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'url' is an in-place operation, use 'set_url' instead",
            UserWarning,
        )
        return self.set_url(url=url, in_place=True)

    ############################
    ######>> img utils <<#######
    ############################

    def _download_image(self) -> Path:
        """Download image to cache directory."""
        cache_path = self._cache_dir / Path(urlparse(self._url).path).name

        if not cache_path.exists():
            response = requests.get(self._url, stream=True)
            response.raise_for_status()

            with cache_path.open("wb") as f:
                shutil.copyfileobj(response.raw, f)

        return cache_path

    @lru_cache(maxsize=32)
    def img_raster(self) -> Image.Image:
        """Download (if needed) and load the image."""
        cache_path = self._download_image()
        return Image.open(cache_path)

    def img_source(self, as_path: bool = False) -> str:
        """Get the source URL or cached path of the image.

        Args:
            as_path: If True, returns downloaded path. Defaults to False.

        Returns:
            URL or cached path of the image.
        """
        if as_path:
            return str(self._download_image())
        return self._url


def construct_spatial_image_class(
    x: Union[str, Image.Image, np.ndarray], is_url: Optional[bool] = None
) -> VirtualSpatialImage:
    """Factory function to create appropriate SpatialImage object."""
    if isinstance(x, VirtualSpatialImage):
        return x
    elif isinstance(x, (Image.Image, np.ndarray)):
        return LoadedSpatialImage(x)
    elif isinstance(x, (str, Path)):
        if is_url is None:
            is_url = urlparse(str(x)).scheme in ("http", "https", "ftp")

        if is_url:
            return RemoteSpatialImage(str(x))
        else:
            return StoredSpatialImage(x)
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")
