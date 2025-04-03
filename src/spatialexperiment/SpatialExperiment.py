from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import urlparse
from warnings import warn

import numpy as np
from PIL import Image

from biocframe import BiocFrame
import biocutils as ut
from summarizedexperiment._combineutils import (
    check_assays_are_equal,
    merge_assays,
    merge_se_colnames,
    relaxed_merge_assays,
)
from summarizedexperiment._frameutils import _sanitize_frame
from summarizedexperiment.RangedSummarizedExperiment import GRangesOrGRangesList
from singlecellexperiment import SingleCellExperiment
from singlecellexperiment._combineutils import merge_generic, relaxed_merge_generic, relaxed_merge_numpy_generic

from ._imgutils import get_img_idx
from ._validators import (
    _validate_column_data,
    _validate_id,
    _validate_img_data,
    _validate_sample_ids,
    _validate_sample_image_ids,
    _validate_spatial_coords,
    _validate_spatial_coords_names,
)
from ._combineutils import merge_spatial_frames, merge_spatial_coordinates
from .SpatialImage import construct_spatial_image_class, VirtualSpatialImage

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
        validate: bool = True,
        **kwargs,
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
                columns of the matrices in assays. For instances of the
                ``SpatialExperiment`` class, the sample data must include
                a column named `sample_id`. If any 'sample_id' in the sample data is not present in the 'sample_id's of 'img_data', a warning will be issued.

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
                Optional :py:class:`~np.ndarray` or :py:class:`~biocframe.BiocFrame.BiocFrame` containing columns of spatial coordinates. Must be the same length as `column_data`.

                If `spatial_coords` is a :py:class:`~biocframe.BiocFrame.BiocFrame`, typical column names might include:

                    - **['x', 'y']**: For simple 2D coordinates.
                    - **['pxl_col_in_fullres', 'pxl_row_in_fullres']**: For pixel-based coordinates in full-resolution images.

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

            validate:
                Internal use only.
        """
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
            alternative_experiment_check_dim_names=alternative_experiment_check_dim_names,
            validate=validate,
            **kwargs,
        )

        column_data = _sanitize_frame(column_data, num_rows=self.shape[1])

        if not column_data.has_column("sample_id"):
            column_data["sample_id"] = ["sample01"] * self.shape[1]  # hard code default sample_id as "sample01"

        spatial_coords = _sanitize_frame(spatial_coords, num_rows=self.shape[1])
        img_data = _sanitize_frame(img_data, num_rows=0)

        self._img_data = img_data
        self._cols = column_data
        self._spatial_coords = spatial_coords

        if validate:
            _validate_column_data(column_data=column_data)
            _validate_img_data(img_data=img_data)
            _validate_sample_ids(column_data=column_data, img_data=img_data)
            _validate_spatial_coords(spatial_coords=spatial_coords, column_data=column_data)

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
        output = f"{type(self).__name__}(number_of_rows={self.shape[0]}"
        output += f", number_of_columns={self.shape[1]}"
        output += ", assays=" + ut.print_truncated_list(self.assay_names)

        output += ", row_data=" + self._rows.__repr__()
        if self._row_names is not None:
            output += ", row_names=" + ut.print_truncated_list(self._row_names)

        output += ", column_data=" + self._cols.__repr__()
        if self._column_names is not None:
            output += ", column_names=" + ut.print_truncated_list(self._column_names)

        if self._spatial_coords is not None:
            output += ", spatial_coords=" + self._spatial_coords.__repr__()

        if self._img_data is not None:
            output += ", img_data=" + self._img_data.__repr__()

        if self._row_ranges is not None:
            output += ", row_ranges=" + self._row_ranges.__repr__()

        if self._alternative_experiments is not None:
            output += ", alternative_experiments=" + ut.print_truncated_list(self.alternative_experiment_names)

        if self._reduced_dims is not None:
            output += ", reduced_dims=" + ut.print_truncated_list(self.reduced_dim_names)

        if self._main_experiment_name is not None:
            output += ", main_experiment_name=" + self._main_experiment_name

        if len(self._row_pairs) > 0:
            output += ", row_pairs=" + ut.print_truncated_dict(self._row_pairs)

        if len(self._column_pairs) > 0:
            output += ", column_pairs=" + ut.print_truncated_dict(self._column_pairs)

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

        output += f"dimensions: ({self.shape[0]}, {self.shape[1]})\n"

        output += f"assays({len(self.assay_names)}): {ut.print_truncated_list(self.assay_names)}\n"

        output += (
            f"row_data columns({len(self._rows.column_names)}): {ut.print_truncated_list(self._rows.column_names)}\n"
        )
        output += f"row_names({0 if self._row_names is None else len(self._row_names)}): {' ' if self._row_names is None else ut.print_truncated_list(self._row_names)}\n"

        output += (
            f"column_data columns({len(self._cols.column_names)}): {ut.print_truncated_list(self._cols.column_names)}\n"
        )
        output += f"column_names({0 if self._column_names is None else len(self._column_names)}): {' ' if self._column_names is None else ut.print_truncated_list(self._column_names)}\n"

        output += f"main_experiment_name: {' ' if self._main_experiment_name is None else self._main_experiment_name}\n"
        output += f"reduced_dims({len(self.reduced_dim_names)}): {ut.print_truncated_list(self.reduced_dim_names)}\n"
        output += f"alternative_experiments({len(self.alternative_experiment_names)}): {ut.print_truncated_list(self.alternative_experiment_names)}\n"
        output += f"row_pairs({len(self.row_pair_names)}): {ut.print_truncated_list(self.row_pair_names)}\n"
        output += f"column_pairs({len(self.column_pair_names)}): {ut.print_truncated_list(self.column_pair_names)}\n"

        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        output += f"spatial_coords columns({len(self.spatial_coords_names)}): {ut.print_truncated_list(self.spatial_coords_names)}\n"
        output += f"img_data columns({len(self._img_data.column_names)}): {ut.print_truncated_list(self._img_data.column_names)}"

        return output

    ##############################
    #####>> spatial_coords <<#####
    ##############################

    def get_spatial_coordinates(self) -> Union[BiocFrame, np.ndarray]:
        """Access spatial coordinates.

        Returns:
            A ``BiocFrame`` containing columns of spatial coordinates.
        """
        return self._spatial_coords

    def get_spatial_coords(self) -> BiocFrame:
        """Alias for :py:meth:`~get_spatial_coordinates`."""
        return self.get_spatial_coordinates()

    def set_spatial_coordinates(
        self,
        spatial_coords: Optional[Union[BiocFrame, np.ndarray]],
        in_place: bool = False,
    ) -> "SpatialExperiment":
        """Set new spatial coordinates.

        Args:
            spatial_coords:
                Optional :py:class:`~np.ndarray` or :py:class:`~biocframe.BiocFrame.BiocFrame` containing columns of spatial coordinates. Must be the same length as `column_data`.

                If `spatial_coords` is a :py:class:`~biocframe.BiocFrame.BiocFrame`, typical column names might include:

                    - **['x', 'y']**: For simple 2D coordinates.
                    - **['pxl_col_in_fullres', 'pxl_row_in_fullres']**: For pixel-based coordinates in full-resolution images.

                To remove coordinate information, set `spatial_coords=None`.

                If spatial coordinates is a :py:class:`~pd.DataFrame` or `None`, it is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to `None`.

            in_place:
                Whether to modify the ``SpatialExperiment`` in place. Defaults to False.

        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        spatial_coords = _sanitize_frame(spatial_coords, self.shape[1])

        _validate_spatial_coords(spatial_coords, self.column_data)

        output = self._define_output(in_place)
        output._spatial_coords = spatial_coords
        return output

    def set_spatial_coords(
        self,
        spatial_coords: Optional[Union[BiocFrame, np.ndarray]],
        in_place: bool = False,
    ) -> "SpatialExperiment":
        """Alias for :py:meth:`~set_spatial_coordinates`."""
        return self.set_spatial_coordinates(spatial_coords=spatial_coords, in_place=in_place)

    @property
    def spatial_coords(self) -> BiocFrame:
        """Alias for :py:meth:`~get_spatial_coordinates`."""
        return self.get_spatial_coordinates()

    @spatial_coords.setter
    def spatial_coords(self, spatial_coords: Optional[Union[BiocFrame, np.ndarray]]):
        """Alias for :py:meth:`~set_spatial_coordinates`."""
        warn(
            "Setting property 'spatial_coords' is an in-place operation, use 'set_spatial_coordinates' instead.",
            UserWarning,
        )
        self.set_spatial_coordinates(spatial_coords=spatial_coords, in_place=True)

    @property
    def spatial_coordinates(self) -> BiocFrame:
        """Alias for :py:meth:`~get_spatial_coordinates`."""
        return self.get_spatial_coordinates()

    @spatial_coordinates.setter
    def spatial_coordinates(self, spatial_coords: Optional[Union[BiocFrame, np.ndarray]]):
        """Alias for :py:meth:`~set_spatial_coordinates`."""
        warn(
            "Setting property 'spatial_coords' is an in-place operation, use 'set_spatial_coordinates' instead.",
            UserWarning,
        )
        self.set_spatial_coordinates(spatial_coords=spatial_coords, in_place=True)

    ##############################
    ##>> spatial_coords_names <<##
    ##############################

    def get_spatial_coordinates_names(self) -> List[str]:
        """Access spatial coordinates names.

        Returns:
            The defined names of the spatial coordinates.
        """
        if not hasattr(self._spatial_coords, "columns"):
            return []

        return self._spatial_coords.columns.as_list()

    def get_spatial_coords_names(self) -> List[str]:
        """Alias for :py:meth:`~get_spatial_coordinate_names`."""
        return self.get_spatial_coordinate_names()

    def set_spatial_coordinates_names(
        self, spatial_coords_names: List[str], in_place: bool = False
    ) -> "SpatialExperiment":
        """Set new spatial coordinates names.

        Args:
            spatial_coords_names:
                New spatial coordinates names.

            in_place:
                Whether to modify the ``SpatialExperiment`` in place. Defaults to False.

        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        if not hasattr(self._spatial_coords, "set_column_names"):
            warn(
                f"The spatial coordinates object is of type {type(self._spatial_coords).__name__} which does not support setting column names.",
                UserWarning,
            )
            new_spatial_coords = self._spatial_coords
        else:
            _validate_spatial_coords_names(spatial_coords_names, self._spatial_coords)
            new_spatial_coords = self._spatial_coords.set_column_names(spatial_coords_names)

        output = self._define_output(in_place)
        output._spatial_coords = new_spatial_coords
        return output

    def set_spatial_coords_names(self, spatial_coords_names: List[str], in_place: bool = False) -> "SpatialExperiment":
        """Alias for :py:meth:`~set_spatial_coordinates_names`."""
        return self.set_spatial_coordinates_names(spatial_coords_names=spatial_coords_names, in_place=in_place)

    @property
    def spatial_coords_names(self) -> List[str]:
        """Alias for :py:meth:`~get_spatial_coordinates_names`."""
        return self.get_spatial_coordinates_names()

    @spatial_coords_names.setter
    def spatial_coords_names(self, spatial_coords_names: List[str]):
        """Alias for :py:meth:`~set_spatial_coordinates_names`."""
        warn(
            "Setting property 'spatial_coords_names' is an in-place operation, use 'set_spatial_coordinates_names' instead.",
            UserWarning,
        )
        self.set_spatial_coordinates_names(spatial_coords_names=spatial_coords_names, in_place=True)

    @property
    def spatial_coordinates_names(self) -> List[str]:
        """Alias for :py:meth:`~get_spatial_coordinates_names`."""
        return self.get_spatial_coordinates_names()

    @spatial_coordinates_names.setter
    def spatial_coordinates_names(self, spatial_coords_names: List[str]):
        """Alias for :py:meth:`~set_spatial_coordinates_names`."""
        warn(
            "Setting property 'spatial_coords_names' is an in-place operation, use 'set_spatial_coordinates_names' instead.",
            UserWarning,
        )
        self.set_spatial_coordinates_names(spatial_coords_names=spatial_coords_names, in_place=True)

    ##############################
    ########>> img_data <<########
    ##############################

    def get_image_data(self) -> BiocFrame:
        """Access image data.

        Returns:
            A BiocFrame object containing the image data.
        """
        return self._img_data

    def get_img_data(self) -> BiocFrame:
        """Alias for :py:meth:`~get_image_data`."""
        return self.get_image_data()

    def set_image_data(self, img_data: Optional[BiocFrame], in_place: bool = False) -> "SpatialExperiment":
        """Set new image data.

        Args:
            img_data:
                :py:class:`~biocframe.BiocFrame.BiocFrame` containing the image data, structured with the following columns:
                    - **sample_id** (str): A string identifier for the sample to which an image corresponds.
                    - **image_id** (str): A unique string identifier for each image within each sample.
                    - **data** (VirtualSpatialImage): The image itself, represented as a `VirtualSpatialImage` object or one of its subclasses.
                    - **scale_factor** (float): A numerical value that indicates the scaling factor applied to the image.

                Image data are coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`.

            in_place:
                Whether to modify the ``SpatialExperiment`` in place. Defaults to False.

        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        img_data = _sanitize_frame(img_data, num_rows=0)

        _validate_img_data(img_data)
        _validate_sample_ids(self.column_data, img_data)

        output = self._define_output(in_place)
        output._img_data = img_data
        return output

    def set_img_data(self, img_data: BiocFrame, in_place: bool = False) -> "SpatialExperiment":
        """Alias for :py:meth:`~set_image_data`."""
        return self.set_image_data(img_data=img_data, in_place=in_place)

    @property
    def img_data(self) -> BiocFrame:
        """Alias for :py:meth:`~get_image_data`."""
        return self.get_image_data()

    @img_data.setter
    def img_data(self, img_data: BiocFrame):
        """Alias for :py:meth:`~set_image_data`."""
        warn(
            "Setting property 'img_data' is an in-place operation, use 'set_image_data' instead.",
            UserWarning,
        )
        self.set_image_data(img_data=img_data, in_place=True)

    @property
    def image_data(self) -> BiocFrame:
        """Alias for :py:meth:`~get_image_data`."""
        return self.get_image_data()

    @image_data.setter
    def image_data(self, img_data: BiocFrame):
        """Alias for :py:meth:`~set_image_data`."""
        warn(
            "Setting property 'img_data' is an in-place operation, use 'set_image_data' instead.",
            UserWarning,
        )
        self.set_image_data(img_data=img_data, in_place=True)

    ##############################
    #####>> scale_factors <<######
    ##############################

    def get_scale_factors(
        self,
        sample_id: Union[str, bool, None] = None,
        image_id: Union[str, bool, None] = None,
    ) -> List[float]:
        """Return scale factor(s) of image(s) based on the provided sample and image ids.
            See :py:meth:`~get_img` for more details on the behavior for various
            combinations of `sample_id` and `image_id` values.

        Args:
            sample_id:
                - `sample_id=True`: Matches all samples.
                - `sample_id=None`: Matches the first sample.
                - `sample_id="<str>"`: Matches a sample by its id.

            image_id:
                - `image_id=True`: Matches all images for the specified sample(s).
                - `image_id=None`: Matches the first image for the sample(s).
                - `image_id="<str>"`: Matches image(s) by its(their) id.

        Returns:
            The scale factor(s) of the specified image(s).
        """
        _validate_id(sample_id)
        _validate_id(image_id)

        idxs = get_img_idx(img_data=self.img_data, sample_id=sample_id, image_id=image_id)

        return self.img_data[idxs,]["scale_factor"]

    ################################
    ###>> OVERRIDE column_data <<###
    ################################

    def set_column_data(
        self,
        cols: Optional[BiocFrame],
        replace_column_names: bool = False,
        in_place: bool = False,
    ) -> "SpatialExperiment":
        """Override: Set sample data.

        Args:
            cols:
                New sample data. If 'cols' contains a column
                named 'sample_id's, a check is performed to ensure
                that all 'sample_id's in the 'img_data' are present. If any 'sample_id' in the 'cols' is not present in the 'sample_id's of 'img_data', a warning will be issued.

                If 'sample_id' is not present or 'cols' is None, the original 'sample_id's are retained.

            replace_column_names:
                Whether to replace experiment's column_names with the names from the
                new object. Defaults to False.

            in_place:
                Whether to modify the ``SpatialExperiment`` in place. Defaults to False.

        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        cols = _sanitize_frame(cols, num_rows=self.shape[1])
        if "sample_id" not in cols.columns:
            cols["sample_id"] = self.column_data["sample_id"]

        _validate_column_data(column_data=cols)
        _validate_sample_ids(column_data=cols, img_data=self.img_data)

        output = self._define_output(in_place)
        output._cols = cols

        if replace_column_names:
            return output.set_column_names(cols.row_names, in_place=in_place)

        return output

    ################################
    #########>> slicers <<##########
    ################################

    def get_slice(
        self,
        rows: Optional[Union[str, int, bool, Sequence]],
        columns: Optional[Union[str, int, bool, Sequence]],
    ) -> "SpatialExperiment":
        """Alias for :py:attr:`~__getitem__`."""

        spe = super().get_slice(rows=rows, columns=columns)

        slicer = self._generic_slice(rows=rows, columns=columns)
        do_slice_cols = not (isinstance(slicer.col_indices, slice) and slicer.col_indices == slice(None))

        new_spatial_coords = None

        if do_slice_cols:
            new_spatial_coords = self.spatial_coords[slicer.col_indices, :]

        column_sample_ids = set(spe.column_data["sample_id"])
        mask = [sample_id in column_sample_ids for sample_id in self.img_data["sample_id"]]

        new_img_data = self.img_data[mask,]

        current_class_const = type(self)
        return current_class_const(
            assays=spe.assays,
            row_ranges=spe.row_ranges,
            row_data=spe.row_data,
            column_data=spe.column_data,
            row_names=spe.row_names,
            column_names=spe.column_names,
            metadata=spe.metadata,
            main_experiment_name=spe.main_experiment_name,
            reduced_dims=spe.reduced_dims,
            alternative_experiments=spe.alternative_experiments,
            row_pairs=spe.row_pairs,
            column_pairs=spe.column_pairs,
            spatial_coords=new_spatial_coords,
            img_data=new_img_data,
        )

    ################################
    #####>> img_data methods <<#####
    ################################

    def get_img(
        self,
        sample_id: Union[str, bool, None] = None,
        image_id: Union[str, bool, None] = None,
    ) -> Union[VirtualSpatialImage, List[VirtualSpatialImage]]:
        """Retrieve spatial images based on the provided sample and image ids.

        Args:
            sample_id:
                - `sample_id=True`: Matches all samples.
                - `sample_id=None`: Matches the first sample.
                - `sample_id="<str>"`: Matches a sample by its id.

            image_id:
                - `image_id=True`: Matches all images for the specified sample(s).
                - `image_id=None`: Matches the first image for the sample(s).
                - `image_id="<str>"`: Matches image(s) by its(their) id.

        Returns:
            The image(s) matching the specified criteria. Returns `None` if `img_data` is `None`.
            When a single image matches, returns a :py:class:`~VirtualSpatialImage` object.
            When multiple images match, returns a list of :py:class:`~VirtualSpatialImage` objects.

        Behavior:
            - sample_id = True, image_id = True:
                Returns all images from all samples.

            - sample_id = None, image_id = None:
                Returns the first image entry in the dataset.

            - sample_id = True, image_id = None:
                Returns the first image for each sample.

            - sample_id = None, image_id = True:
                Returns all images for the first sample.

            - sample_id = <str>, image_id = True:
                Returns all images for the specified sample.

            - sample_id = <str>, image_id = None:
                Returns the first image for the specified sample.

            - sample_id = <str>, image_id = <str>:
                Returns the image matching the specified sample and image identifiers.

        Raises:
            ValueError: If no row matches the provided sample_id and image_id pair.
        """
        _validate_id(sample_id)
        _validate_id(image_id)

        if not self.img_data:
            return None

        indices = get_img_idx(img_data=self.img_data, sample_id=sample_id, image_id=image_id)

        if len(indices) == 0:
            raise ValueError(f"No matching rows for sample_id={sample_id} and image_id={image_id}")

        images = self.img_data[indices,]["data"]
        return images[0] if len(images) == 1 else images

    def add_img(
        self,
        image_source: Union[Image.Image, np.ndarray, str, Path],
        scale_factor: float,
        sample_id: Union[str, bool, None],
        image_id: Union[str, bool, None],
        load: bool = True,
        in_place: bool = False,
    ) -> "SpatialExperiment":
        """Add a new image entry.

        Args:
            image_source:
                The file path to the image.

            scale_factor:
                The scaling factor associated with the image.

            sample_id:
                The sample id of the image.

            image_id:
                The image id of the image.

            load:
                Whether to load the image into memory. If `True`,
                the method reads the image file from
                `image_source`.
                Defaults to `True`.

            in_place:
                Whether to modify the ``SpatialExperiment`` in place.
                Defaults to False.

        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.

        Raises:
            ValueError: If the sample_id and image_id pair already exists.

        Note:
            See :py:meth:`~get_img` for detailed behavior regarding sample_id and image_id parameters.
        """
        _validate_sample_image_ids(img_data=self._img_data, new_sample_id=sample_id, new_image_id=image_id)

        if isinstance(image_source, (str, Path)):
            is_url = urlparse(str(image_source)).scheme in ("http", "https", "ftp")
            spi = construct_spatial_image_class(image_source, is_url=is_url)

            if load:
                img = spi.img_raster()
                spi = construct_spatial_image_class(img, is_url=False)
        else:
            spi = construct_spatial_image_class(image_source, is_url=False)

        new_row = BiocFrame(
            {
                "sample_id": [sample_id],
                "image_id": [image_id],
                "data": [spi],
                "scale_factor": [scale_factor],
            }
        )
        new_img_data = self._img_data.combine_rows(new_row)

        output = self._define_output(in_place)
        output._img_data = new_img_data
        return output

    def remove_img(
        self, sample_id: Union[str, bool, None] = None, image_id: Union[str, bool, None] = None, in_place: bool = False
    ) -> "SpatialExperiment":
        """Remove an image entry.

        Args:
            sample_id:
                - `sample_id=True`: Matches all samples.
                - `sample_id=None`: Matches the first sample.
                - `sample_id="<str>"`: Matches a sample by its id.

            image_id:
                - `image_id=True`: Matches all images for the specified sample(s).
                - `image_id=None`: Matches the first image for the sample(s).
                - `image_id="<str>"`: Matches image(s) by its(their) id.

            in_place:
                Whether to modify the ``SpatialExperiment`` in place.
                Defaults to False.

        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.

        Raises:
            ValueError: If no row matches the provided sample_id and image_id pair.

        Note:
            See :py:meth:`~get_img` for detailed behavior regarding sample_id and image_id parameters.
        """
        _validate_id(sample_id)
        _validate_id(image_id)

        indices = get_img_idx(img_data=self.img_data, sample_id=sample_id, image_id=image_id)

        if len(indices) == 0:
            raise ValueError(f"No matching rows for sample_id={sample_id} and image_id={image_id}")

        new_img_data = self._img_data.remove_rows(indices)

        output = self._define_output(in_place=in_place)
        output._img_data = new_img_data
        return output

    def img_source(
        self,
        sample_id: Union[str, bool, None] = None,
        image_id: Union[str, bool, None] = None,
        path=False,
    ):
        """Retrieve the source(s) for images stored in the SpatialExperiment object.

        Args:
            sample_id:
                - `sample_id=True`: Matches all samples.
                - `sample_id=None`: Matches the first sample.
                - `sample_id="<str>"`: Matches a sample by its id.

            image_id:
                - `image_id=True`: Matches all images for the specified sample(s).
                - `image_id=None`: Matches the first image for the sample(s).
                - `image_id="<str>"`: Matches image(s) by its(their) id.

            path: If True, returns path as string. Defaults to False.

        Returns:
            The image source(s) for the matching criteria. Returns `None` if `img_data` is `None`.
            When a single image matches, returns its source as a `str`, `Path`, or `None`.
            When multiple images match, returns a list of sources.

        Raises:
            ValueError: If no row matches the provided sample_id and image_id pair.

        Note:
            See :py:meth:`~get_img` for detailed behavior regarding sample_id and image_id parameters.
        """
        spis = self.get_img(sample_id=sample_id, image_id=image_id)

        if spis is None:
            return None

        if isinstance(spis, VirtualSpatialImage):
            return spis.img_source(as_path=path)

        img_sources = [spi.img_source(as_path=path) for spi in spis]

        return img_sources

    def img_raster(self, sample_id=None, image_id=None):
        # NOTE: this function seems redundant, might be an artifact of the different subclasses of SpatialImage in the R implementation? just call `get_img()` for now
        self.get_img(sample_id=sample_id, image_id=image_id)

    def rotate_img(self, sample_id=None, image_id=None, degrees=90):
        raise NotImplementedError()

    def mirror_img(self, sample_id=None, image_id=None, axis=("h", "v")):
        raise NotImplementedError()

    @staticmethod
    def to_spatial_experiment():
        raise NotImplementedError()

    ################################
    #######>> combine ops <<########
    ################################

    def relaxed_combine_columns(self, *other) -> "SpatialExperiment":
        """Wrapper around :py:func:`~relaxed_combine_columns`."""
        return relaxed_combine_columns(self, *other)

    def combine_columns(self, *other) -> "SpatialExperiment":
        """Wrapper around :py:func:`~combine_columns`."""
        return combine_columns(self, *other)


################################
#######>> combine ops <<########
################################


@ut.combine_columns.register(SpatialExperiment)
def combine_columns(*x: SpatialExperiment) -> SpatialExperiment:
    """Combine multiple ``SpatialExperiment`` objects by column.

    All assays must contain the same assay names. If you need a
    flexible combine operation, checkout :py:func:`~relaxed_combine_columns`.

    Returns:
        A combined ``SpatialExperiment``.
    """
    warn(
        "'row_pairs' and 'column_pairs' are currently ignored during this operation.",
        UserWarning,
    )

    first = x[0]
    _all_assays = [y.assays for y in x]
    check_assays_are_equal(_all_assays)
    _new_assays = merge_assays(_all_assays, by="column")

    _new_col_names = merge_se_colnames(x)

    _new_rdim = None
    try:
        _new_rdim = merge_generic(x, by="row", attr="reduced_dims")
    except Exception as e:
        warn(
            f"Cannot combine 'reduced_dimensions' across experiments, {str(e)}",
            UserWarning,
        )

    _new_alt_expt = None
    try:
        _new_alt_expt = merge_generic(x, by="column", attr="alternative_experiments")
    except Exception as e:
        warn(
            f"Cannot combine 'alternative_experiments' across experiments, {str(e)}",
            UserWarning,
        )

    _all_spatial_coords = [y._spatial_coords for y in x]
    _new_spatial_coords = merge_spatial_coordinates(_all_spatial_coords)

    _new_cols, _new_img_data = merge_spatial_frames(x)

    current_class_const = type(first)
    return current_class_const(
        assays=_new_assays,
        row_ranges=first._row_ranges,
        row_data=first._rows,
        column_data=_new_cols,
        row_names=first._row_names,
        column_names=_new_col_names,
        metadata=first._metadata,
        reduced_dims=_new_rdim,
        main_experiment_name=first._main_experiment_name,
        alternative_experiments=_new_alt_expt,
        spatial_coords=_new_spatial_coords,
        img_data=_new_img_data,
    )


@ut.relaxed_combine_columns.register(SpatialExperiment)
def relaxed_combine_columns(
    *x: SpatialExperiment,
) -> SpatialExperiment:
    """A relaxed version of the :py:func:`~biocutils.combine_rows.combine_columns` method for
    :py:class:`~SpatialExperiment` objects.  Whereas ``combine_columns`` expects that all objects have the same rows,
    ``relaxed_combine_columns`` allows for different rows. Absent columns in any object are filled in with appropriate
    placeholder values before combining.

    Args:
        x:
            One or more ``SpatialExperiment`` objects, possibly with differences in the
            number and identity of their rows.

    Returns:
        A ``SpatialExperiment`` that combines all ``experiments`` along their columns and contains
        the union of all rows. Rows absent in any ``x`` are filled in
        with placeholders consisting of Nones or masked NumPy values.
    """
    warn(
        "'row_pairs' and 'column_pairs' are currently ignored during this operation.",
        UserWarning,
    )

    first = x[0]
    _new_assays = relaxed_merge_assays(x, by="column")

    _new_col_names = merge_se_colnames(x)

    _new_rdim = None
    try:
        _new_rdim = relaxed_merge_numpy_generic(x, by="row", attr="reduced_dims")
    except Exception as e:
        warn(
            f"Cannot combine 'reduced_dimensions' across experiments, {str(e)}",
            UserWarning,
        )

    _new_alt_expt = None
    try:
        _new_alt_expt = relaxed_merge_generic(x, by="column", attr="alternative_experiments")
    except Exception as e:
        warn(
            f"Cannot combine 'alternative_experiments' across experiments, {str(e)}",
            UserWarning,
        )

    _all_spatial_coords = [y._spatial_coords for y in x]
    _new_spatial_coords = merge_spatial_coordinates(_all_spatial_coords)

    _new_cols, _new_img_data = merge_spatial_frames(x)

    current_class_const = type(first)
    return current_class_const(
        assays=_new_assays,
        row_ranges=first._row_ranges,
        row_data=first._rows,
        column_data=_new_cols,
        row_names=first._row_names,
        column_names=_new_col_names,
        metadata=first._metadata,
        reduced_dims=_new_rdim,
        main_experiment_name=first._main_experiment_name,
        alternative_experiments=_new_alt_expt,
        spatial_coords=_new_spatial_coords,
        img_data=_new_img_data,
    )
