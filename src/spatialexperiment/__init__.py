import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "SpatialExperiment"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .io import read_tenx_visium
from .spatialexperiment import SpatialExperiment
from .spatialimage import (
    LoadedSpatialImage,
    RemoteSpatialImage,
    StoredSpatialImage,
    VirtualSpatialImage,
    construct_spatial_image_class,
)

__all__ = [
    "read_tenx_visium",
    "SpatialExperiment",
    "LoadedSpatialImage",
    "RemoteSpatialImage",
    "StoredSpatialImage",
    "VirtualSpatialImage",
    "construct_spatial_image_class",
]
