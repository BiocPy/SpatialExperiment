from typing import Any, Dict, List, Optional, Union
from warnings import warn

import biocutils as ut
import geopandas as gpd
import numpy as np
from biocframe import BiocFrame
from summarizedexperiment._frameutils import _sanitize_frame
from summarizedexperiment.RangedSummarizedExperiment import GRangesOrGRangesList

from .SpatialExperiment import SpatialExperiment, _validate_column_data, _validate_sample_ids

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _sanitize_geomertries(geometries):
    """Sanitize geometry objects."""
    if geometries is None:
        return gpd.GeoDataFrame({})

    return geometries


def _sanitize_spatial_graphs(spatial_graph, sample_ids):
    """Sanitize spatial graphs."""
    if spatial_graph is None:
        return BiocFrame({}, number_of_rows=3, row_names=["row", "col", "annot"], column_names=sample_ids)

    if hasattr(spatial_graph, "dtypes"):
        return BiocFrame.from_pandas(spatial_graph)

    return spatial_graph


def _validate_geometries(geometries: Dict[str, gpd.GeoDataFrame], prop_name: str):
    """Validate geometry objects."""
    if geometries is None or len(geometries) == 0:
        return

    for i, geom in enumerate(geometries.values()):
        if not isinstance(geom, gpd.GeoDataFrame):
            raise TypeError(f"Item {i} in {prop_name} is {type(geom).__name__} " f"rather than `GeoDataFrame`.\n")


def _validate_annotgeometries(geometries, column_data):
    """Validate annotation geometries."""
    if geometries is None or len(geometries) == 0:
        return

    sample_ids = column_data.get("sample_id")
    if sample_ids is None:
        raise ValueError("No 'sample_id' column in 'column_data'")

    for i, geom in enumerate(geometries.values()):
        if "sample_id" not in geom.columns:
            raise ValueError(f"Item {i} of 'annot_geometries' does not have column 'sample_id'.\n")
        else:
            samples_seen = geom["sample_id"].unique()
            missing = [s for s in samples_seen if s not in sample_ids]
            if len(missing) > 0:
                raise ValueError(
                    f"Samples {', '.join(missing)} in item {i} of " f"annot_geometries are absent from 'column_data'.\n"
                )


def _validate_graph_sample_id(spatial_graphs, column_data):
    """Validate graph sample IDs match column data."""
    col_sample_ids = set(column_data.get_column("sample_id"))
    graph_sample_ids = set(spatial_graphs.get_column_names())

    missing = graph_sample_ids - col_sample_ids
    if missing:
        raise ValueError(f"Samples {', '.join(missing)} are in the graphs but not 'column_data'.\n")


def _validate_graph_structure(spatial_graphs):
    """Validate spatial graph structure."""
    if spatial_graphs is None:
        return

    if not isinstance(spatial_graphs, BiocFrame):
        raise TypeError(
            "'spatial_graphs' must be a `BiocFrame` whose columns are 'sample_ids' "
            "and whose rows are margins (rows, columns, annotation).\n"
        )
    elif not set(spatial_graphs.get_row_names()) == {"row", "col", "annot"}:
        raise ValueError("Row names of 'spatial_graphs' must be 'row', 'col', and 'annot'.\n")


class ProxySpatialFeatureExperiment(SpatialExperiment):
    """Container class for storing data from spatial-omics experiments with feature geometries.

    This class extends SpatialExperiment to provide slots for geometries of spots/cells,
    tissue boundaries, pathologist annotations and other spatial features.
    """

    def __init__(
        self,
        assays: Dict[str, Any] = None,
        row_ranges: Optional[GRangesOrGRangesList] = None,
        row_data: Optional[BiocFrame] = None,
        column_data: Optional[BiocFrame] = None,
        row_names: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        reduced_dims: Optional[Dict[str, Any]] = None,
        main_experiment_name: Optional[str] = None,
        alternative_experiments: Optional[Dict[str, Any]] = None,
        alternative_experiment_check_dim_names: bool = True,
        row_pairs: Optional[Any] = None,
        column_pairs: Optional[Any] = None,
        spatial_coords: Optional[Union[BiocFrame, np.ndarray]] = None,
        img_data: Optional[BiocFrame] = None,
        # SFE args
        col_geometries: Optional[Dict[str, gpd.GeoDataFrame]] = None,
        row_geometries: Optional[Dict[str, gpd.GeoDataFrame]] = None,
        annot_geometries: Optional[Dict[str, gpd.GeoDataFrame]] = None,
        spatial_graphs: Optional[BiocFrame] = None,
        unit: str = "full_res_image_pixel",
        validate: bool = True,
        **kwargs,
    ) -> None:
        """Initialize a spatial feature class.

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
                columns of the matrices in assays. For instances of the
                ``SpatialExperiment`` class, the sample data must include
                a column named `sample_id`. If any 'sample_id' in the sample data is not
                present in the 'sample_id's of 'img_data', a warning will be issued.

                If `sample_id` is not present, a column with this name
                will be created and filled with the default value `sample01`.

                Sample information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            row_names:
                A list of strings, same as the number of rows.Defaults to None.

            column_names:
                A list of strings, same as the number of columns. Defaults to None.

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

            alternative_experiment_check_dim_names:
                Whether to check if the column names of the alternative experiment match the column names
                of the main experiment. This is the equivalent to the ``withDimnames``
                parameter in the R implementation.

                Defaults to True.

            row_pairs:
                Row pairings/relationships between features.

                Defaults to None.

            column_pairs:
                Column pairings/relationships between cells.

                Defaults to None.

            spatial_coords:
                Optional :py:class:`~np.ndarray` or :py:class:`~biocframe.BiocFrame.BiocFrame`
                containing columns of spatial coordinates. Must be the same length as `column_data`.

                If `spatial_coords` is a :py:class:`~biocframe.BiocFrame.BiocFrame`, typical
                column names might include:

                    - **['x', 'y']**: For simple 2D coordinates.
                    - **['pxl_col_in_fullres', 'pxl_row_in_fullres']**: For pixel-based
                    coordinates in full-resolution images.

                If spatial coordinates is a :py:class:`~pd.DataFrame` or `None`, it is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to `None`.

            img_data:
                Optional :py:class:`~biocframe.BiocFrame.BiocFrame` containing the image data, structured with the following columns:
                    - **sample_id** (str): A string identifier for the sample to which an image corresponds.
                    - **image_id** (str): A unique string identifier for each image within each sample.
                    - **data** (VirtualSpatialImage): The image itself, represented as a `VirtualSpatialImage` object or one of its subclasses.
                    - **scale_factor** (float): A numerical value that indicates the scaling factor applied to the image.

                All 'sample_id's in 'img_data' must be present in the 'sample_id's of 'column_data'.

                Image data are coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            col_geometries:
                Dictionary of GeoDataFrames containing geometries for columns
                (e.g. cells, spots).

            row_geometries:
                Dictionary of GeoDataFrames containing geometries for rows
                (e.g. genes).

            annot_geometries:
                Dictionary of GeoDataFrames containing annotation geometries
                (e.g. tissue boundaries).

            spatial_graphs:
                BiocFrame containing spatial neighborhood graphs.

            unit:
                Unit for spatial coordinates ('full_res_image_pixel' or 'micron').

            validate:
                Internal use only.
        """
        # Initialize parent class
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
            row_pairs=row_pairs,
            column_pairs=column_pairs,
            alternative_experiments=alternative_experiments,
            alternative_experiment_check_dim_names=alternative_experiment_check_dim_names,
            img_data=img_data,
            spatial_coords=spatial_coords,
            validate=validate,
            **kwargs,
        )

        # Initialize geometries
        self._col_geometries = _sanitize_geomertries(col_geometries)
        self._row_geometries = _sanitize_geomertries(row_geometries)
        self._annot_geometries = _sanitize_geomertries(annot_geometries)
        self._spatial_graphs = _sanitize_spatial_graphs(
            spatial_graphs, list(set(self.get_column_data().get_column("sample_id")))
        )
        self._unit = unit

        if validate:
            self._validate()

    def _validate(self) -> None:
        """Validate the object."""

        # Check geometries
        _validate_geometries(self._col_geometries, "col_geometries")
        _validate_geometries(self._row_geometries, "row_geometries")
        _validate_geometries(self._annot_geometries, "annot_geometries")

        # Check annotation geometries sample IDs
        _validate_annotgeometries(self._annot_geometries, self.get_column_data())

        # Check graph structure and sample IDs
        _validate_graph_structure(self._spatial_graphs)
        _validate_graph_sample_id(self._spatial_graphs, self.get_column_data())

    #########################
    ######>> Copying <<######
    #########################

    def __deepcopy__(self, memo=None, _nil=[]):
        """
        Returns:
            A deep copy of the current ``SpatialExperiment``.
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
        _spatial_coords_copy = deepcopy(self._spatial_coords)
        _img_data_copy = deepcopy(self._img_data)
        _col_geometries_copy = deepcopy(self._col_geometries)
        _row_geometries_copy = deepcopy(self._row_geometries)
        _annot_geometries_copy = deepcopy(self._annot_geometries)
        _spatial_graphs_copy = deepcopy(self._spatial_graphs)
        _unit_copy = deepcopy(self._unit)

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
            spatial_coords=_spatial_coords_copy,
            img_data=_img_data_copy,
            col_geometries=_col_geometries_copy,
            row_geometries=_row_geometries_copy,
            annot_geometries=_annot_geometries_copy,
            spatial_graphs=_spatial_graphs_copy,
            unit=_unit_copy,
        )

    def __copy__(self):
        """
        Returns:
            A shallow copy of the current ``SpatialExperiment``.
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
            spatial_coords=self._spatial_coords,
            img_data=self._img_data,
            col_geometries=self._col_geometries,
            row_geometries=self._row_geometries,
            annot_geometries=self._annot_geometries,
            spatial_graphs=self._spatial_graphs,
            unit=self._unit,
        )

    def copy(self):
        """Alias for :py:meth:`~__copy__`."""
        return self.__copy__()

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self) -> str:
        """Get string representation."""
        output = super().__repr__()

        output = output[:-1]
        output += f", unit='{self.unit}'"

        col_geoms = len(self._col_geometries)
        row_geoms = len(self._row_geometries)
        annot_geoms = len(self._annot_geometries)

        if col_geoms:
            output += f", col_geometries={col_geoms}"
        if row_geoms:
            output += f", row_geometries={row_geoms}"
        if annot_geoms:
            output += f", annot_geometries={annot_geoms}"

        output += ")"
        return output

    def __str__(self) -> str:
        """Get detailed string representation."""
        output = super().__str__()

        output += f"\nunit: {self._unit}"

        col_geoms = self._col_geometries
        row_geoms = self._row_geometries
        annot_geoms = self._annot_geometries

        if col_geoms or row_geoms or annot_geoms:
            output += "\nGeometries:\n"

            if col_geoms:
                geom_types = [f"{k} ({v.geometry.geom_type[0]})" for k, v in col_geoms.items()]
                output += f"col_geometries: {', '.join(geom_types)}\n"

            if row_geoms:
                geom_types = [f"{k} ({v.geometry.geom_type[0]})" for k, v in row_geoms.items()]
                output += f"row_geometries: {', '.join(geom_types)}\n"

            if annot_geoms:
                geom_types = [f"{k} ({v.geometry.geom_type[0]})" for k, v in annot_geoms.items()]
                output += f"annot_geometries: {', '.join(geom_types)}\n"

        # Add graphs info
        graphs = self._spatial_graphs
        if graphs is not None:
            output += "\nGraphs:"
            for col in graphs.columns:
                output += f"\n{col}: "
                parts = []
                for row in graphs.row_names:
                    if graphs[row, col]:
                        graph_names = list(graphs[row, col].keys())
                        if graph_names:
                            parts.append(f"{row}: {', '.join(graph_names)}")
                output += "; ".join(parts)

        return output

    ####################
    #####>> unit <<#####
    ####################

    def get_unit(self) -> str:
        """Get the coordinate unit."""
        return self._unit

    def set_unit(self, unit: str, in_place: bool = False) -> "ProxySpatialFeatureExperiment":
        """Set the coordinate unit.

        Args:
            unit:
                New unit ('full_res_image_pixel' or 'micron').

            in_place:
                Whether to modify the ``ProxySpatialFeatureExperiment`` in place. Defaults to False.

        Returns:
            A modified ``ProxySpatialFeatureExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        if unit not in ("full_res_image_pixel", "micron"):
            raise ValueError("unit must be 'full_res_image_pixel' or 'micron'")

        output = self._define_output(in_place)
        output._unit = unit
        return output

    @property
    def unit(self) -> str:
        """Get coordinate unit."""
        return self.get_unit()

    @unit.setter
    def unit(self, unit: str):
        """Set coordinate unit."""
        warn(
            "Setting property 'unit' is an in-place operation, use 'set_unit' instead.",
            UserWarning,
        )
        self.set_unit(unit, in_place=True)

    #####################
    #####>> geoms <<#####
    #####################

    def get_col_geometries(self) -> Dict[str, gpd.GeoDataFrame]:
        """Get column geometries."""
        return self._col_geometries

    def get_row_geometries(self) -> Dict[str, gpd.GeoDataFrame]:
        """Get row geometries."""
        return self._row_geometries

    def get_annot_geometries(self) -> Dict[str, gpd.GeoDataFrame]:
        """Get annotation geometries."""
        return self._annot_geometries

    def set_col_geometries(
        self, geometries: Dict[str, gpd.GeoDataFrame], in_place: bool = False
    ) -> "ProxySpatialFeatureExperiment":
        """Set column geometries.

        Args:
            geometries:
                New column geometries.

            in_place:
                Whether to modify the ``ProxySpatialFeatureExperiment`` in place. Defaults to False.

        Returns:
            A modified ``ProxySpatialFeatureExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        _geoms = _sanitize_geomertries(geometries=geometries)
        _validate_geometries(_geoms, "col_geometries")

        output = self._define_output(in_place)
        output._col_geometries = _geoms
        return output

    def set_row_geometries(
        self, geometries: Dict[str, gpd.GeoDataFrame], in_place: bool = False
    ) -> "ProxySpatialFeatureExperiment":
        """Set row geometries.

        Args:
            geometries:
                New row geometries.

            in_place:
                Whether to modify the ``ProxySpatialFeatureExperiment`` in place. Defaults to False.

        Returns:
            A modified ``ProxySpatialFeatureExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        _geoms = _sanitize_geomertries(geometries=geometries)
        _validate_geometries(_geoms, "row_geometries")

        output = self._define_output(in_place)
        output._row_geometries = _geoms
        return output

    def set_annot_geometries(
        self, geometries: Dict[str, gpd.GeoDataFrame], in_place: bool = False
    ) -> "ProxySpatialFeatureExperiment":
        """Set annotation geometries.

        Args:
            geometries:
                New annotation geometries.

            in_place:
                Whether to modify the ``ProxySpatialFeatureExperiment`` in place. Defaults to False.

        Returns:
            A modified ``ProxySpatialFeatureExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        _geoms = _sanitize_geomertries(geometries=geometries)

        _validate_geometries(_geoms, "annot_geometries")
        _validate_annotgeometries(_geoms, self.get_column_data())

        output = self._define_output(in_place)
        output._annot_geometries = _geoms
        return output

    @property
    def col_geometries(self) -> Dict[str, gpd.GeoDataFrame]:
        """Get column geometries."""
        return self.get_col_geometries()

    @col_geometries.setter
    def col_geometries(self, geometries: Dict[str, gpd.GeoDataFrame]):
        """Set column geometries."""
        warn(
            "Setting property 'col_geometries' is an in-place operation, use 'set_col_geometries' instead.",
            UserWarning,
        )
        self.set_col_geometries(geometries, in_place=True)

    @property
    def row_geometries(self) -> Dict[str, gpd.GeoDataFrame]:
        """Get row geometries."""
        return self.get_row_geometries()

    @row_geometries.setter
    def row_geometries(self, geometries: Dict[str, gpd.GeoDataFrame]):
        """Set row geometries."""
        warn(
            "Setting property 'row_geometries' is an in-place operation, use 'set_row_geometries' instead.",
            UserWarning,
        )
        self.set_row_geometries(geometries, in_place=True)

    @property
    def annot_geometries(self) -> Dict[str, gpd.GeoDataFrame]:
        """Get annotation geometries."""
        return self.get_annot_geometries()

    @annot_geometries.setter
    def annot_geometries(self, geometries: Dict[str, gpd.GeoDataFrame]):
        """Set annotation geometries."""
        warn(
            "Setting property 'annot_geometries' is an in-place operation, use 'set_annot_geometries' instead.",
            UserWarning,
        )
        self.set_annot_geometries(geometries, in_place=True)

    ##############################
    #####>> spatial_graphs <<#####
    ##############################

    def get_spatial_graphs(self) -> Optional[BiocFrame]:
        """Get spatial neighborhood graphs."""
        return self._spatial_graphs

    def set_spatial_graphs(
        self, graphs: Optional[BiocFrame], in_place: bool = False
    ) -> "ProxySpatialFeatureExperiment":
        """Set spatial neighborhood graphs.

        Args:
            graphs:
                New spatial graphs as `BiocFrame`.

            in_place:
                Whether to modify the ``ProxySpatialFeatureExperiment`` in place. Defaults to False.

        Returns:
            A modified ``ProxySpatialFeatureExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        _graphs = _sanitize_spatial_graphs(graphs, list(set(self.get_column_data().get_column("sample_id"))))
        _validate_graph_structure(_graphs)
        _validate_graph_sample_id(_graphs, self.get_column_data())

        output = self._define_output(in_place)
        output._spatial_graphs = _graphs
        return output

    @property
    def spatial_graphs(self) -> Optional[BiocFrame]:
        """Get spatial graphs."""
        return self.get_spatial_graphs()

    @spatial_graphs.setter
    def spatial_graphs(self, graphs: Optional[BiocFrame]):
        """Set spatial graphs."""
        warn(
            "Setting property 'spatial_graphs' is an in-place operation, use 'set_spatial_graphs' instead.",
            UserWarning,
        )
        self.set_spatial_graphs(graphs, in_place=True)

    ################################
    #########>> slicers <<##########
    ################################

    def get_slice(
        self, rows: Optional[Union[str, int, bool, List]] = None, columns: Optional[Union[str, int, bool, List]] = None
    ) -> "ProxySpatialFeatureExperiment":
        """Get a slice of the experiment.

        Args:
            rows:
                Row indices/names to select.

            columns:
                Column indices/names to select.

        Returns:
            Sliced ProxySpatialFeatureExperiment.
        """
        sfe = super().get_slice(rows=rows, columns=columns)

        slicer = self._generic_slice(rows=rows, columns=columns)
        do_slice_cols = not (isinstance(slicer.col_indices, slice) and slicer.col_indices == slice(None))

        # Update geometries and graphs
        if do_slice_cols:
            # Get sample IDs for filtered columns
            column_sample_ids = set(sfe.column_data["sample_id"])

            # Filter annotation geometries by sample ID
            new_annot_geoms = {}
            for name, geom in self.annot_geometries.items():
                mask = [sid in column_sample_ids for sid in geom["sample_id"]]
                if any(mask):
                    new_annot_geoms[name] = geom[mask]

            # Filter column geometries
            new_col_geoms = {}
            for name, geom in self.col_geometries.items():
                new_col_geoms[name] = geom.iloc[slicer.col_indices]

            # Filter spatial graphs
            new_graphs = None
            if self.spatial_graphs is not None:
                # Keep only columns for remaining samples
                cols_to_keep = [c for c in self.spatial_graphs.columns if c in column_sample_ids]
                if cols_to_keep:
                    new_graphs = self.spatial_graphs[cols_to_keep]

        else:
            new_annot_geoms = self.annot_geometries
            new_col_geoms = self.col_geometries
            new_graphs = self.spatial_graphs

        # Filter row geometries if needed
        new_row_geoms = {}
        if rows is not None:
            for name, geom in self.row_geometries.items():
                new_row_geoms[name] = geom.iloc[slicer.row_indices]
        else:
            new_row_geoms = self.row_geometries

        current_class_const = type(self)
        return current_class_const(
            assays=sfe.assays,
            row_ranges=sfe.row_ranges,
            row_data=sfe.row_data,
            column_data=sfe.column_data,
            row_names=sfe.row_names,
            column_names=sfe.column_names,
            metadata=sfe.metadata,
            main_experiment_name=sfe.main_experiment_name,
            reduced_dims=sfe.reduced_dims,
            alternative_experiments=sfe.alternative_experiments,
            row_pairs=sfe.row_pairs,
            column_pairs=sfe.column_pairs,
            spatial_coords=sfe.spatial_coords,
            img_data=sfe.image_data,
            col_geometries=new_col_geoms,
            row_geometries=new_row_geoms,
            annot_geometries=new_annot_geoms,
            spatial_graphs=new_graphs,
            unit=self.unit,
        )

    ################################
    ###>> OVERRIDE column_data <<###
    ################################

    def set_column_data(
        self, cols: Optional[BiocFrame], replace_column_names: bool = False, in_place: bool = False
    ) -> "ProxySpatialFeatureExperiment":
        """Override: Set sample data.

        Args:
            cols:
                New sample data. If 'cols' contains a column named 'sample_id's, a check
                is performed to ensure that all 'sample_id's in the 'img_data' are present.
                If any 'sample_id' in the 'cols' is not present in the 'sample_id's of 'img_data',
                a warning will be issued.

                If 'sample_id' is not present or 'cols' is None, the original 'sample_id's are retained.

            replace_column_names:
                Whether to replace experiment's column_names with the names from the
                new object. Defaults to False.

            in_place:
                Whether to modify the ``ProxySpatialFeatureExperiment`` in place. Defaults to False.

        Returns:
            A modified ``ProxySpatialFeatureExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        cols = _sanitize_frame(cols, num_rows=self.shape[1])
        if "sample_id" not in cols.columns:
            cols["sample_id"] = self.column_data["sample_id"]

        _validate_column_data(column_data=cols)
        _validate_sample_ids(column_data=cols, img_data=self.img_data)
        _validate_annotgeometries(self._annot_geometries, column_data=cols)
        _validate_graph_sample_id(self._spatial_graphs, column_data=cols)

        output = self._define_output(in_place)
        output._cols = cols

        if replace_column_names:
            return output.set_column_names(cols.row_names, in_place=in_place)

        return output
