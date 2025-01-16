from typing import Any, Dict, List, Optional, Union

import pandas as pd
import biocframe
from summarizedexperiment.RangedSummarizedExperiment import (
    GRangesOrGRangesList
)
from singlecellexperiment import SingleCellExperiment

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


class SpatialExperiment(SingleCellExperiment):
    """Container class for storing data from spatial -omics experiments, extending
    :py:class:`~singlecellexperiment.SingleCellExperiment` to provide slots for
    image data and spatial coordinates.

    In contrast to R, :py:class:`~numpy.ndarray` or scipy matrices are unnamed and do
    not contain rownames and colnames. Hence, these matrices cannot be directly used as
    values in assays or alternative experiments. We strictly enforce type checks in these cases.
    """

    def __init__(
        self,
        assays: Dict[str, Any] = None,
        row_ranges: Optional[GRangesOrGRangesList] = None,
        row_data: Optional[biocframe.BiocFrame] = None,
        column_data: Optional[biocframe.BiocFrame] = None,
        row_names: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        reduced_dims: Optional[Dict[str, Any]] = None,
        main_experiment_name: Optional[str] = None,
        alternative_experiments: Optional[Dict[str, Any]] = None,
        row_pairs: Optional[Any] = None,
        column_pairs: Optional[Any] = None,

        # ============== SpatialExperiment additional arguments ===============
        sample_id: Optional[str] = "sample01",

        # Bioconductor implementation allows for both spatialCoord and spatialCoordsNames to be None
        spatial_coords: Optional[biocframe.BiocFrame] = None,
        spatial_coords_names: Optional[List[str]] = None,

        scale_factors: Optional[Union[int, float, List[Union[int, float]], str]] = 1,
        img_data: Optional[pd.DataFrame] = None,
        image_sources: Optional[Union[str, List[str]]] = None,
        image_id: Optional[List[str]] = None,
        load_image: bool = True,

        validate: bool = True,
    ) -> None:
        """Initialize a spatial experiment.

        Args:
            assays:
                A dictionary containing matrices, with assay names as keys
                and 2-dimensional matrices represented as either
                :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

                Alternatively, you may use any 2-dimensional matrix that has
                the ``shape`` property and implements the slice operation
                using the ``__getitem__`` dunder method.

                All matrices in assays must be 2-dimensional and have the
                same shape (number of rows, number of columns).

            row_ranges:
                Genomic features, must be the same length as the number of rows of
                the matrices in assays.

            row_data:
                Features, must be the same length as the number of rows of
                the matrices in assays.

                Feature information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            column_data:
                Sample data, must be the same length as the number of
                columns of the matrices in assays.

                Sample information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            row_names:
                A list of strings, same as the number of rows.Defaults to None.

            column_names:
                A list of string, same as the number of columns. Defaults to None.

            metadata:
                Additional experimental metadata describing the methods.
                Defaults to None.

            reduced_dims:
                Slot for low-dimensionality embeddings.

                Usually a dictionary with the embedding method as keys (e.g., t-SNE, UMAP)
                and the dimensions as values.

                Embeddings may be represented as a matrix or a data frame, must contain a shape.

            main_experiment_name:
                A string, specifying the main experiment name.

            alternative_experiments:
                Used to manage multi-modal experiments performed on the same sample/cells.

                Alternative experiments must contain the same cells (rows) as the primary experiment.
                It's a dictionary with keys as the names of the alternative experiments
                (e.g., sc-atac, crispr) and values as subclasses of
                :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

            row_pairs:
                Row pairings/relationships between features.

                Defaults to None.

            column_pairs:
                Column pairings/relationships between cells.

                Defaults to None.
            
            spatial_coords:


            validate:
                Internal use only.
        """
        if spatial_coords_names is None:
            if spatial_coords is None:
                raise ValueError("If `spatial_coords_names` is None, `spatial_coords` must be specified")

            self._spatial_coords = spatial_coords
        else:
            missing_names = [name for name in spatial_coords_names if name not in column_data.column_names]
            if missing_names:
                raise ValueError(f"The following names in `spatial_coords_names` are missing from `column_data`: {missing_names}")
            
            # TODO: make deep copies of sliced column_data?
            self._spatial_coords = column_data[:, spatial_coords_names]
            
            columns_to_keep = [colname for colname in column_data.column_names not in spatial_coords_names]
            column_data = column_data[:, columns_to_keep]
            

        super().__init__(
            assays=assays,
            row_ranges=row_ranges,
            row_data=row_data,
            column_data=column_data,
            row_names=row_names,
            column_names=column_names,
            metadata=metadata,
            reduced_dims=reduced_dims,
            main_experiment_name=main_experiment_name,
            alternative_experiments=alternative_experiments,
            row_pairs=row_pairs,
            column_pairs=column_pairs,
            validate=validate,
        )
        