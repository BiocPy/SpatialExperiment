from typing import Any, Dict, List, Optional, Union

import pandas as pd
import biocframe
from summarizedexperiment.RangedSummarizedExperiment import GRangesOrGRangesList
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

        # ============== SpatialExperiment arguments ===============

        sample_id: Optional[str] = "sample01",
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
        from copy import deepcopy

        # `spatial_coords_names` takes precedence
        current_column_data = column_data

        # TODO: some nuances when `column_data` has a column named `sample_id`. see 'Details' in the Bioconductor SpatialExperiment vignette (pg. 9)
        current_column_data = column_data.set_column(
            "sample_id", [sample_id] * len(column_data)
        )

        if spatial_coords_names:
            missing_names = [
                name for name in spatial_coords_names if name not in current_column_data.column_names
            ]
            if missing_names:
                raise ValueError(
                    f"The following names in `spatial_coords_names` are missing from `column_data`: {missing_names}"
                )

            spatial_coords = deepcopy(current_column_data[:, spatial_coords_names])
            current_column_data = deepcopy(current_column_data[:, [col for col in current_column_data.column_names if col not in spatial_coords_names]])

        self._spatial_coords = spatial_coords

        super().__init__(
            assays=assays,
            row_ranges=row_ranges,
            row_data=row_data,
            column_data=current_column_data,
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

    #########################
    ######>> Copying <<######
    #########################

    def __deepcopy__(self, memo=None, _nil=[]):
        """
        Returns:
            A deep copy of the current ``SingleCellExperiment``.
        """
        from copy import deepcopy

        _assays_copy = deepcopy(self._assays)
        _rows_copy = deepcopy(self._rows)
        _rowranges_copy = deepcopy(self._row_ranges)
        _cols_copy = deepcopy(self._cols)
        _row_names_copy = deepcopy(self._row_names)
        _col_names_copy = deepcopy(self._column_names)
        _metadata_copy = deepcopy(self.metadata)
        _main_expt_name_copy = deepcopy(self._main_experiment_name)
        _red_dim_copy = deepcopy(self._reduced_dims)
        _alt_expt_copy = deepcopy(self._alternative_experiments)
        _row_pair_copy = deepcopy(self._row_pairs)
        _col_pair_copy = deepcopy(self._column_pairs)

        current_class_const = type(self)
        return current_class_const(
            assays=_assays_copy,
            row_ranges=_rowranges_copy,
            row_data=_rows_copy,
            column_data=_cols_copy,
            row_names=_row_names_copy,
            column_names=_col_names_copy,
            metadata=_metadata_copy,
            reduced_dims=_red_dim_copy,
            main_experiment_name=_main_expt_name_copy,
            alternative_experiments=_alt_expt_copy,
            row_pairs=_row_pair_copy,
            column_pairs=_col_pair_copy,
        )

    def __copy__(self):
        """
        Returns:
            A shallow copy of the current ``SingleCellExperiment``.
        """
        current_class_const = type(self)
        return current_class_const(
            assays=self._assays,
            row_ranges=self._row_ranges,
            row_data=self._rows,
            column_data=self._cols,
            row_names=self._row_names,
            column_names=self._column_names,
            metadata=self._metadata,
            reduced_dims=self._reduced_dims,
            main_experiment_name=self._main_experiment_name,
            alternative_experiments=self._alternative_experiments,
            row_pairs=self._row_pairs,
            column_pairs=self._column_pairs,
        )

    def copy(self):
        """Alias for :py:meth:`~__copy__`."""
        return self.__copy__()
