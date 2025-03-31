[![PyPI-Server](https://img.shields.io/pypi/v/SpatialExperiment.svg)](https://pypi.org/project/SpatialExperiment/)
![Unit tests](https://github.com/BiocPy/SpatialExperiment/actions/workflows/run-tests.yml/badge.svg)

# SpatialExperiment

A Python package for storing and analyzing spatial-omics experimental data. `SpatialExperiment` extends [SingleCellExperiment](https://github.com/biocpy/singlecellexperiment) with dedicated slots for image data and spatial coordinates, making it ideal for spatial transcriptomics and other spatially-resolved omics data.

> [!NOTE]
>
> This package is in **active development**.

## Install

To get started, install the package from [PyPI](https://pypi.org/project/SpatialExperiment/)

```bash
pip install spatialexperiment
```

## Usage

The `SpatialExperiment` class extends `SingleCellExperiment` with the following key attributes:

- `spatial_coords`: A BioFrame containing spot/cell spatial coordinates relative to the image, typically including:
  - x-coordinates
  - y-coordinates
  - Additional spatial metadata

- `img_data`: A BiocFrame containing image-related information:
  - sample_ids: Unique identifiers for each sample
  - image_ids: Unique identifiers for each image
  - data: The actual image data
  - scale_factor: Scaling factors for proper image interpretation

- `column_data`: Contains sample_id mappings that link spots to their corresponding images

### Quick Start

Here's how to create a SpatialExperiment object from scratch:

```python
from spatialexperiment import SpatialExperiment, construct_spatial_image_class
import numpy as np
from biocframe import BiocFrame

# Create example data
nrows = 200  # Number of features (e.g., genes)
ncols = 500  # Number of spots/cells

# Generate random count data
counts = np.random.rand(nrows, ncols)

# Create feature annotations
row_data = BiocFrame({
    "gene_ids": [f"gene_{i}" for i in range(nrows)],
    "gene_names": [f"Gene_{i}" for i in range(nrows)]
})

# Create spot/cell annotations
col_data = BiocFrame({
    "n_genes": [50, 200] * int(ncols / 2),
    "condition": ["healthy", "tumor"] * int(ncols / 2),
    "cell_id": [f"spot_{i}" for i in range(ncols)],
    "sample_id": ["sample_1"] * int(ncols / 2) + ["sample_2"] * int(ncols / 2),
})

# Generate spatial coordinates
spatial_coords = BiocFrame({
    "x": np.random.uniform(low=0.0, high=100.0, size=ncols),
    "y": np.random.uniform(low=0.0, high=100.0, size=ncols)
})

# Create image data
img_data = BiocFrame({
    "sample_id": ["sample_1", "sample_1", "sample_2"],
    "image_id": ["aurora", "dice", "desert"],
    "data": [
        construct_spatial_image_class("tests/images/sample_image1.jpg"),
        construct_spatial_image_class("tests/images/sample_image2.png"),
        construct_spatial_image_class("tests/images/sample_image3.jpg"),
    ],
    "scale_factor": [1, 1, 1],
})

# Create SpatialExperiment object
spe = SpatialExperiment(
    assays={"counts": counts},
    row_data=row_data,
    column_data=col_data,
    spatial_coords=spatial_coords,
    img_data=img_data,
)
```

For more detailed information about available methods and functionality, please refer to the [SingleCellExperiment documentation](https://biocpy.github.io/SingleCellExperiment/).


<!-- biocsetup-notes -->

## Note

This project has been set up using [BiocSetup](https://github.com/biocpy/biocsetup)
and [PyScaffold](https://pyscaffold.org/).
