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
from rasterio.transform import Affine

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

    ##################################
    ######>> Spatial Props <<#########
    ##################################

    def affine(self, scale_factor: float = 1.0) -> Affine:
        """Computes a simple affine transformation from the scale_factor.
        Assumes pixel (0,0) is top-left and maps to spatial origin (0,0).
        Y-axis in spatial coordinates increases downwards by default (matching pixel rows).
        Use `Affine.scale(self.scale_factor, -self.scale_factor) * Affine.translation(0, height_in_pixels)`
        if Y spatial needs to increase upwards.
        """
        return Affine.scale(scale_factor, scale_factor)

    def get_dimensions(self) -> Tuple[int, int]:
        """Get image dimensions (width, height) in pixels."""
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
    def img_source(self, as_path: bool = False) -> Union[str, Path, None]:
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

    def to_numpy(self, **kwargs) -> np.ndarray:
        """Convert the image raster to a NumPy array.

        Args:
            **kwargs:
                Additional arguments passed to `np.array()`.

        Returns:
            NumPy array representation of the image.
        """
        return np.array(self.img_raster(), **kwargs)

    def rotate_img(self, degrees: float = 90) -> "LoadedSpatialImage":
        """Rotate image by specified degrees clockwise.

        Returns:
            A new LoadedSpatialImage.
        """
        img = self.img_raster()
        # PIL rotates counter-clockwise
        rotated_pil_img = img.rotate(-degrees, expand=True)
        return LoadedSpatialImage(image=rotated_pil_img, metadata=self.metadata.copy())

    def mirror_img(self, axis: str = "h") -> "LoadedSpatialImage":
        """Mirror image horizontally or vertically.

        Args:
            axis:
                'h' for horizontal (default) or 'v' for vertical.

        Returns:
            A new LoadedSpatialImage.
        """
        img = self.img_raster()

        if axis.lower() == "h":
            mirrored_pil_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif axis.lower() == "v":
            mirrored_pil_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError("axis must be 'h' or 'v'")

        return LoadedSpatialImage(
            image=mirrored_pil_img,
            metadata=self.metadata.copy(),
        )


def _sanitize_loaded_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image, np.ndarray):
        # trying to infer mode for multi-channel arrays if not RGBA/RGB
        if image.ndim == 3:
            if image.shape[2] == 1:  # Grayscale with channel dim
                _result = Image.fromarray(image.squeeze(axis=2))
            elif image.shape[2] not in [3, 4]:  # common RGB/RGBA
                warn(
                    f"NumPy array has {image.shape[2]} channels; Pillow might not infer mode correctly. Ensure it's compatible e.g. (H,W,3) or (H,W,4)."
                )
                _result = Image.fromarray(image)  # Lets try PIL
            else:
                _result = Image.fromarray(image)
        elif image.ndim == 2:  # Grayscale
            _result = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported NumPy array shape: {image.shape}. Expected 2D (H,W) or 3D (H,W,C).")
    elif isinstance(image, Image.Image):
        _result = image
    else:
        raise TypeError(f"image must be PIL.Image.Image or numpy.ndarray, got '{type(image)}'.")
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

    #########################
    ######>> Equality <<#####
    #########################

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False
        if not isinstance(other, LoadedSpatialImage):
            return False

        # compare image content
        try:
            diff = ImageChops.difference(self.img_raster(), other.img_raster())
            return not diff.getbbox()
        except Exception as _:
            # If images are not comparable (e.g. different modes, sizes after operations)
            return False

    def __hash__(self):
        # Hashing Image directly is problematic due to internal state PIL maintains.
        # Hashing bytes is more reliable but can be slow for large images.
        try:
            img_bytes = self._image.tobytes()
        except Exception as _:
            # Fallback if tobytes fails for some reason
            img_bytes = id(self._image)  # Not ideal, but better than erroring hash
        return hash((super().__hash__(), img_bytes))

    #########################
    ######>> Copying <<######
    #########################

    def __deepcopy__(self, memo=None, _nil=[]):
        """
        Returns:
            A deep copy of the current ``LoadedSpatialImage``.
        """
        from copy import deepcopy

        _img_copy = self._image.copy()
        _metadata_copy = deepcopy(self.metadata)

        current_class_const = type(self)
        return current_class_const(
            image=_img_copy,
            metadata=_metadata_copy,
        )

    def __copy__(self):
        """
        Returns:
            A shallow copy of the current ``LoadedSpatialImage``.
        """
        current_class_const = type(self)
        return current_class_const(
            image=self._image.copy(),
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
        """Get the PIL Image object."""
        return self._image

    def set_image(self, image: Union[Image.Image, np.ndarray], in_place: bool = False) -> "LoadedSpatialImage":
        """Set new image.

        Args:
            image:
                Image represented as a :py:class:`~numpy.ndarray` or :py:class:`~PIL.Image.Image`.

            in_place:
                Whether to modify the ``LoadedSpatialImage`` in place. Defaults to False.

        Returns:
            Modified LoadedSpatialImage.
        """
        _out = self._define_output(in_place=in_place)
        _out._image = _sanitize_loaded_image(image)
        # reset lru_cache for methods that depend on image content if any were used
        return _out

    @property
    def image(self) -> Image.Image:
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
        """Get the source of the loaded image (always None for in-memory)."""
        return None

    ############################
    ######>> img utils <<#######
    ############################

    def img_raster(self) -> Image.Image:
        """Get the image as a PIL Image object."""
        return self._image


def _sanitize_path(path: Union[str, Path]) -> Path:
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

    #########################
    ######>> Equality <<#####
    #########################

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False
        return isinstance(other, StoredSpatialImage) and self.path == other.path

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
        # Clear LRU cache if path changes
        if in_place and hasattr(self.img_raster, "cache_clear"):
            self.img_raster.cache_clear()
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
        self.set_path(path=path, in_place=True)

    def img_source(self, as_path: bool = False) -> str:
        """Get the source path of the image.

        Args:
            as_path: If True, returns string path. Defaults to False.

        Returns:
            Path to the image.
        """
        return str(self._path) if as_path else self._path

    ############################
    ######>> img utils <<#######
    ############################

    # Simple in-memory cache
    @lru_cache(maxsize=32)
    def img_raster(self) -> Image.Image:
        """Load and cache the image from path."""
        return Image.open(self._path)


def _validate_url(url: str):
    parsed = urlparse(url)

    # Must have scheme (http/https) and network location (domain)
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
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if validate:
            _validate_url(url)

    #########################
    ######>> Equality <<#####
    #########################

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False
        return isinstance(other, RemoteSpatialImage) and self.url == other.url

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
            metadata=self.metadata,
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

    def set_url(self, url: str, in_place: bool = False, validate: bool = True) -> "RemoteSpatialImage":
        """Update the url to the image file.

        Args:
            url:
                New URL for this image.

            in_place:
                Whether to modify the ``RemoteSpatialImage`` in place.

            validate:
                Whether to validate the url.

        Returns:
            A modified ``RemoteSpatialImage`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        if validate:
            _validate_url(url)
        output = self._define_output(in_place=in_place)
        output._url = url
        if in_place and hasattr(self.img_raster, "cache_clear"):
            self.img_raster.cache_clear()
            if hasattr(self._get_cached_path, "cache_clear"):
                self._get_cached_path.cache_clear()
        return output

    @property
    def url(self) -> str:
        """Alias for :py:meth:`~get_url`."""
        return self.get_url()

    @url.setter
    def url(self, url: str):
        """Alias for :py:attr:`~set_url` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'url' is an in-place operation, use 'set_url' instead",
            UserWarning,
        )
        self.set_url(url=url, in_place=True)

    ############################
    ######>> img utils <<#######
    ############################

    @lru_cache(maxsize=1)
    def _get_cached_path(self) -> Path:
        """Internal method to get the cached path, downloads if not exists."""
        url_path_part = Path(urlparse(self._url).path)
        filename = url_path_part.name
        if not filename:
            import hashlib

            filename = hashlib.md5(self._url.encode()).hexdigest() + (url_path_part.suffix or ".img")

        cache_path = self._cache_dir / filename

        if not cache_path.exists():
            try:
                _validate_url(self._url)
                response = requests.get(self._url, stream=True)
                response.raise_for_status()

                with cache_path.open("wb") as f:
                    shutil.copyfileobj(response.raw, f)
            except requests.exceptions.RequestException as e:
                # If download fails, remove incomplete cache file and re-raise
                if cache_path.exists():
                    cache_path.unlink(missing_ok=True)
                raise IOError(f"Failed to download image from {self._url}: {e}.") from e
            except ValueError as e:
                raise ValueError(f"Invalid URL for download {self._url}: {e}.") from e
        return cache_path

    @lru_cache(maxsize=32)
    def img_raster(self) -> Image.Image:
        """Download (if needed) and load the image from cache."""
        try:
            cached_file_path = self._get_cached_path()
            return Image.open(cached_file_path)
        except Exception as e:
            if hasattr(self._get_cached_path, "cache_clear"):
                self._get_cached_path.cache_clear()
            raise RuntimeError(f"Could not load image from URL {self._url} via cache: {e}.")

    def img_source(self, as_path: bool = False) -> str:
        """Get the source URL or cached path of the image.

        Args:
            as_path:
                If True, returns path to the downloaded (cached) file.
                If False (default), returns the original remote URL.

        Returns:
            URL or cached path of the image.
        """
        if as_path:
            try:
                return str(self._get_cached_path())
            except Exception as e:
                warn(f"Could not obtain cached path for {self.url}: {e}. Returning original URL.")
                return self._url
        return self._url


def construct_spatial_image_class(
    x: Union[str, Path, Image.Image, np.ndarray, VirtualSpatialImage],
    metadata: Optional[dict] = None,
    is_url: Optional[bool] = None,
) -> VirtualSpatialImage:
    """Factory function to create appropriate SpatialImage object.

    Args:
        x:
            Image source (path, URL, PIL Image, NumPy array) or an existing VirtualSpatialImage.

        metadata:
            Additional metadata dictionary.

        is_url:
            Explicitly treat `x` as a URL if it's a string.

    Returns:
        An instance of a VirtualSpatialImage subclass.
    """
    if isinstance(x, VirtualSpatialImage):
        return x
    elif isinstance(x, (Image.Image, np.ndarray)):
        return LoadedSpatialImage(x, metadata)
    elif isinstance(x, (str, Path)):
        path_str = str(x)
        if is_url is None:
            try:
                parsed = urlparse(path_str)
                is_url = all([parsed.scheme, parsed.netloc]) and parsed.scheme in ("http", "https", "ftp")
            except Exception:
                is_url = False

        if is_url:
            return RemoteSpatialImage(path_str, metadata)
        else:
            return StoredSpatialImage(Path(path_str), metadata)
    else:
        raise TypeError(f"Unsupported input type for image construction: {type(x)}.")
