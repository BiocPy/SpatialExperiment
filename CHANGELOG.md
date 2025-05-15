# Changelog

## Version 0.0.11-0.0.12
- BUGFIX: `to_anndata()` only populates `obsm` with spatial coordinates if the original `SpatialExperiment` has spatial coordinates (PR #53, #54)


## Version 0.0.10
- Add an affine function that computes a `rasterio.Affine` object given a `scale_factor`. This assumes a simple scaling where the origin is (0,0) in the spatial coordinate system corresponding to the top-left pixel (0,0). More complex alignments would require explicit affine transforms.
- Ensure img_raster() consistently returns a PIL.Image.Image.
- Add to_numpy() method.
- Changes to how caching works in remote images.


## Version 0.0.9
- Added `to_anndata()` in main `SpatialExperiment` class (PR #50)


## Version 0.0.8
- Set the expected column names for image data slot (PR #46)


## Version 0.0.7
- Added `img_source` function in main SpatialExperiment class and child classes of VirtualSpatialExperiment (PR #36)
- Added `remove_img` function (PR #34)
- Refactored `get_img_idx` for improved maintainability
- Disambiguated `get_img_data` between `_imgutils.py` and `SpatialExperiment.py`
- Moved `SpatialFeatureExperiment` into its own package


## Version 0.0.6
- Added `read_tenx_visium()` function to load 10x Visium data as SpatialExperiment
- Added `combine_columns` function
- Implemented `__eq__` override for `SpatialImage` subclasses


## Version 0.0.5
- Implementing a placeholder `SpatialFeatureExperiment` class. This version only implements the data structure to hold various geometries but none of the methods except for slicing.


## Version 0.0.3 - 0.0.4
- Streamlining the `SpatialImage` class implementations.


## Version 0.0.1 - 0.0.2
- Initial version of the SpatialExperiment class with the additional slots.
- Allow spatial coordinates to be a numpy array
