from typing import Any, Dict, List, Optional, Union, Sequence
from warnings import warn

from PIL import Image
import biocframe
import biocutils as ut
from summarizedexperiment.RangedSummarizedExperiment import GRangesOrGRangesList
from summarizedexperiment._frameutils import _sanitize_frame
from summarizedexperiment.BaseSE import _guess_assay_shape
from singlecellexperiment import SingleCellExperiment
from .SpatialImage import SpatialImage

from .utils import flatten_list
from ._validators import (
    _validate_sample_image_ids,
    _validate_spatial_coords,
    _validate_img_data,
    _validate_id,
    _validate_column_data,
    _validate_spatial_coords_names,
)

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
        spatial_coords: Optional[biocframe.BiocFrame] = None,
        img_data: Optional[biocframe.BiocFrame] = None,
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
                columns of the matrices in assays. For instances of the
                ``SpatialExperiment`` class, the sample data must include
                a column named `sample_id`.
                # TODO: add details about default 'sample_id' as 'sample01'

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

            row_pairs:
                Row pairings/relationships between features.

                Defaults to None.

            column_pairs:
                Column pairings/relationships between cells.

                Defaults to None.

            spatial_coords:
                Optional :py:class:`~biocframe.BiocFrame.BiocFrame` containing columns of spatial coordinates. Must be the same length as `column_data`. Typical column names might include:

                    - **['x', 'y']**: For simple 2D coordinates.
                    - **['pxl_col_in_fullres', 'pxl_row_in_fullres']**: For pixel-based coordinates in full-resolution images.

                Spatial coordinates are coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            img_data:
                Optional :py:class:`~biocframe.BiocFrame.BiocFrame` containing the image data, structured with the following columns:
                    - **sample_id** (str): A string identifier for the sample to which an image corresponds.
                    - **image_id** (str): A unique string identifier for each image within each sample.
                    - **data** (SpatialImage): The image itself, represented as a SpatialImage object.
                    - **scale_factor** (float): A numerical value that indicates the scaling factor applied to the image.

                Image data are coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            validate:
                Internal use only.
        """
        # TODO: figure out how to handle the case where `spatial_coords` is not None but `column_data` is None. in this case, `column_data` should have a `sample_id` column with the default value `sample_01`. this might remove the need for _guess_assay_shape().
        shape = _guess_assay_shape(
            assays=assays if assays is not None else {},
            rows=row_data,
            cols=column_data,
            row_names=row_names,
            col_names=column_names,
        )

        if column_data is None:
            column_data = biocframe.BiocFrame({"sample_id": []})

        column_data = _sanitize_frame(column_data, num_rows=shape[1])
        spatial_coords = _sanitize_frame(spatial_coords, num_rows=shape[1])
        img_data = _sanitize_frame(img_data, num_rows=0)

        _validate_img_data(img_data=img_data)
        _validate_column_data(column_data=column_data, img_data=img_data)
        _validate_spatial_coords(spatial_coords=spatial_coords, column_data=column_data)

        self._spatial_coords = spatial_coords
        self._img_data = img_data

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

    # TODO: update this section for SpatialExperiment

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

        if self._row_ranges is not None:
            output += ", row_ranges=" + self._row_ranges.__repr__()

        if self._alternative_experiments is not None:
            output += ", alternative_experiments=" + ut.print_truncated_list(
                self.alternative_experiment_names
            )

        if self._reduced_dims is not None:
            output += ", reduced_dims=" + ut.print_truncated_list(
                self.reduced_dim_names
            )

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

        output += f"row_data columns({len(self._rows.column_names)}): {ut.print_truncated_list(self._rows.column_names)}\n"
        output += f"row_names({0 if self._row_names is None else len(self._row_names)}): {' ' if self._row_names is None else ut.print_truncated_list(self._row_names)}\n"

        output += f"column_data columns({len(self._cols.column_names)}): {ut.print_truncated_list(self._cols.column_names)}\n"
        output += f"column_names({0 if self._column_names is None else len(self._column_names)}): {' ' if self._column_names is None else ut.print_truncated_list(self._column_names)}\n"

        output += f"main_experiment_name: {' ' if self._main_experiment_name is None else self._main_experiment_name}\n"
        output += f"reduced_dims({len(self.reduced_dim_names)}): {ut.print_truncated_list(self.reduced_dim_names)}\n"
        output += f"alternative_experiments({len(self.alternative_experiment_names)}): {ut.print_truncated_list(self.alternative_experiment_names)}\n"
        output += f"row_pairs({len(self.row_pair_names)}): {ut.print_truncated_list(self.row_pair_names)}\n"
        output += f"column_pairs({len(self.column_pair_names)}): {ut.print_truncated_list(self.column_pair_names)}\n"

        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output

    ##############################
    #####>> spatial_coords <<#####
    ##############################

    def get_spatial_coordinates(self) -> biocframe.BiocFrame:
        """Access spatial coordinates.

        Returns:
            A ``BiocFrame`` containing columns of spatial coordinates.
        """
        return self._spatial_coords

    def get_spatial_coords(self) -> biocframe.BiocFrame:
        """Alias for :py:meth:`~get_spatial_coordinates`."""
        return self.get_spatial_coordinates()

    def set_spatial_coordinates(
        self, _spatial_coords: Optional[biocframe.BiocFrame], in_place: bool = False
    ) -> "SpatialExperiment":
        """Set new spatial coordinates.

        Args:
            _spatial_coords:
                :py:class:`~biocframe.BiocFrame.BiocFrame` containing columns of spatial coordinates. Must be the same length as `column_data`. Typical column names might include:

                    - **['x', 'y']**: For simple 2D coordinates.
                    - **['pxl_col_in_fullres', 'pxl_row_in_fullres']**: For pixel-based coordinates in full-resolution images.

                Spatial coordinates are coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`.

            in_place:
                Whether to modify the ``SpatialExperiment`` in place. Defaults to False.

        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        spatial_coords = _sanitize_frame(_spatial_coords)
        _validate_spatial_coords(spatial_coords, self.column_data)

        output = self._define_output(in_place)
        output._spatial_coords = spatial_coords
        return output

    def set_spatial_coords(
        self, spatial_coords: biocframe.BiocFrame, in_place: bool = False
    ) -> "SpatialExperiment":
        """Alias for :py:meth:`~set_spatial_coordinates`."""
        return self.set_spatial_coordinates(
            spatial_coords=spatial_coords, in_place=in_place
        )

    @property
    def spatial_coords(self) -> biocframe.BiocFrame:
        """Alias for :py:meth:`~get_spatial_coordinates`."""
        return self.get_spatial_coordinates()

    @spatial_coords.setter
    def spatial_coords(self, spatial_coords: biocframe.BiocFrame):
        """Alias for :py:meth:`~set_spatial_coordinates`."""
        warn(
            "Setting property 'spatial_coords' is an in-place operation, use 'set_spatial_coordinates' instead.",
            UserWarning,
        )
        self.set_spatial_coordinates(spatial_coords=spatial_coords, in_place=True)

    @property
    def spatial_coordinates(self) -> biocframe.BiocFrame:
        """Alias for :py:meth:`~get_spatial_coordinates`."""
        return self.get_spatial_coordinates()

    @spatial_coordinates.setter
    def spatial_coordinates(self, spatial_coords: biocframe.BiocFrame):
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
        _validate_spatial_coords_names(spatial_coords_names, self.spatial_coordinates)

        old_spatial_coordinates = self.get_spatial_coordinates()
        new_spatial_coordinates = old_spatial_coordinates.set_column_names(
            spatial_coords_names
        )

        output = self._define_output(in_place)
        output._spatial_coords = new_spatial_coordinates
        return output

    def set_spatial_coords_names(
        self, spatial_coords_names: List[str], in_place: bool = False
    ) -> "SpatialExperiment":
        """Alias for :py:meth:`~set_spatial_coordinates_names`."""
        return self.set_spatial_coordinates_names(
            spatial_coords_names=spatial_coords_names, in_place=in_place
        )

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
        self.set_spatial_coordinates_names(
            spatial_coords_names=spatial_coords_names, in_place=True
        )

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
        self.set_spatial_coordinates_names(
            spatial_coords_names=spatial_coords_names, in_place=True
        )

    ##############################
    ########>> img_data <<########
    ##############################

    def get_image_data(self) -> biocframe.BiocFrame:
        """Access image data.

        Returns:
            A BiocFrame object containing the image data.
        """
        return self._img_data

    def get_img_data(self) -> biocframe.BiocFrame:
        """Alias for :py:meth:`~get_image_data`."""
        return self.get_image_data()

    def set_image_data(
        self, _img_data: Optional[biocframe.BiocFrame], in_place: bool = False
    ) -> "SpatialExperiment":
        """Set new image data.

        Args:
            _img_data:
                :py:class:`~biocframe.BiocFrame.BiocFrame` containing the image data, structured with the following columns:
                    - **sample_id** (str): A string identifier for the sample to which an image corresponds.
                    - **image_id** (str): A unique string identifier for each image within each sample.
                    - **data** (SpatialImage): The image itself, represented as a SpatialImage object.
                    - **scale_factor** (float): A numerical value that indicates the scaling factor applied to the image.

                Image data are coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`.

            in_place:
                Whether to modify the ``SpatialExperiment`` in place. Defaults to False.

        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        img_data = _sanitize_frame(_img_data)
        _validate_img_data(img_data)

        output = self._define_output(in_place)
        output._img_data = img_data
        return output

    def set_img_data(
        self, img_data: biocframe.BiocFrame, in_place: bool = False
    ) -> "SpatialExperiment":
        """Alias for :py:meth:`~set_image_data`."""
        return self.set_image_data(img_data=img_data, in_place=in_place)

    @property
    def img_data(self) -> biocframe.BiocFrame:
        """Alias for :py:meth:`~get_image_data`."""
        return self.get_image_data()

    @img_data.setter
    def img_data(self, img_data: biocframe.BiocFrame):
        """Alias for :py:meth:`~set_image_data`."""
        warn(
            "Setting property 'img_data' is an in-place operation, use 'set_image_data' instead.",
            UserWarning,
        )
        self.set_image_data(img_data=img_data, in_place=True)

    @property
    def image_data(self) -> biocframe.BiocFrame:
        """Alias for :py:meth:`~get_image_data`."""
        return self.get_image_data()

    @image_data.setter
    def image_data(self, img_data: biocframe.BiocFrame):
        """Alias for :py:meth:`~set_image_data`."""
        warn(
            "Setting property 'img_data' is an in-place operation, use 'set_image_data' instead.",
            UserWarning,
        )
        self.set_image_data(img_data=img_data, in_place=True)

    ##############################
    #####>> scale_factors <<######
    ##############################

    # TODO: implement getters and setters

    ################################
    ###>> OVERRIDE column_data <<###
    ################################

    def set_column_data(
        self, _column_data: Optional[biocframe.BiocFrame], in_place: bool = False
    ) -> "SpatialExperiment":
        """Override: Set sample data.

        Args:
            _column_data:
                :py:class:`~biocframe.BiocFrame.BiocFrame` containing the new sample data.

            in_place:
                Whether to modify the ``SpatialExperiment`` in place. Defaults to False.

        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        # TODO: remove this because it is handled in the else case already
        if _column_data is None:
            column_data = self.column_data[["symbol"]]

        else:
            # TODO: always pass in num_rows to _sanitize_frame
            column_data = _sanitize_frame(_column_data)
            if "sample_id" not in column_data.columns:
                column_data["sample_id"] = self.column_data["sample_id"]
            else:
                # TODO: move out of else; should always validate no matter what
                _validate_column_data(column_data=column_data, img_data=self.img_data)

        output = self._define_output(in_place)
        output._column_data = column_data
        return output

    ################################
    #########>> slicers <<##########
    ################################

    def get_slice(
        self,
        rows: Optional[Union[str, int, bool, Sequence]],
        columns: Optional[Union[str, int, bool, Sequence]]
    ) -> "SpatialExperiment":
        """Alias for :py:attr:`~__getitem__`."""

        spe = super().get_slice(rows=rows, columns=columns)

        slicer = self._generic_slice(rows=rows, columns=columns)
        do_slice_cols = not (isinstance(slicer.col_indices, slice) and slicer.col_indices == slice(None))

        new_spatial_coords = None

        if do_slice_cols:
            new_spatial_coords = self.spatial_coords[slicer.col_indices, :]

        column_sample_ids = set(self.column_data["sample_id"])
        mask = [sample_id in column_sample_ids for sample_id in self.img_data["sample_id"]]

        new_img_data = self.img_data[mask]

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
            img_data=new_img_data
        )

    ################################
    ######>> img_data funcs <<######
    ################################

    def get_img(
        self,
        sample_id: Union[str, True, None] = None,
        image_id: Union[str, True, None] = None,
    ) -> Union[SpatialImage, List[SpatialImage]]:
        """
        Retrieve spatial images based on the provided sample and image ids.

        Args:
            sample_id: The sample id.
                - `sample_id=True`: Matches all samples.
                - `sample_id=None`: Matches the first sample.
                - `sample_id="<str>"`: Matches a sample by its id.

            image_id: The image id.
                - `image_id=True`: Matches all images for the specified sample(s).
                - `image_id=None`: Matches the first image for the sample(s).
                - `image_id="<str>"`: Matches image(s) by its(their) id.

        Returns:
            One or more `SpatialImage` objects.

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
        """
        _validate_id(sample_id)
        _validate_id(image_id)

        if self._img_data is None:
            return None

        if sample_id is True:
            if image_id is True:
                return flatten_list(self._img_data["data"])

            unique_sample_ids = list(set(self._img_data["sample_id"]))
            sample_id_groups = self._img_data.split("sample_id")
            imgs = []
            if image_id is None:
                # get the first image for all samples
                for sample_id in unique_sample_ids:
                    row = sample_id_groups[sample_id].get_row(0)
                    img = row["data"]
                    imgs.append(row)
            else:
                # get images with `image_id` for all samples
                for sample_id in unique_sample_ids:
                    bframe = sample_id_groups[sample_id]
                    img = bframe[bframe["image_id"] == image_id]["data"]
                    imgs.append(img)

                return imgs

        if sample_id is None:
            if image_id is True:
                # get all images for the first sample
                first_sample_id = self._img_data["sample_id"][0]
                imgs = flatten_list(
                    self._img_data[self._img_data["sample_id"] == first_sample_id][
                        "data"
                    ]
                )
                return imgs

            if image_id is None:
                # get the first image entry
                return self._img_data["data"][0]
            else:
                return self._img_data[self._img_data["image_id"] == image_id]["data"][0]

        # `sample_id` is a string
        subset = self._img_data[self._img_data["sample_id"] == sample_id]
        if image_id is True:
            return flatten_list(subset["data"])

        if image_id is None:
            return subset["data"][0]

        return subset[subset["image_id"] == image_id]["data"]

    def add_img(
        self,
        image_source: str,
        scale_factor: float,
        sample_id: Union[str, True, None],
        image_id: Union[str, True, None],
        load: bool = True,
    ) -> "SpatialExperiment":
        """
        Add a new image entry.

        Args:
            image_source: The file path to the image.
            scale_factor: The scaling factor associated with the image.
            sample_id: The sample id.
            image_id: The image id.
            load: Whether to load the image into memory. If `True`,
                the method reads the image file from `image_source`. Defaults to `True`.

        Returns:
            The updated SpatialExperiment object containing the
            newly added image entry.

        Raises:
            ValueError: If the sample_id and image_id pair already exists.
        """
        _validate_sample_image_ids(
            img_data=self._img_data, new_sample_id=sample_id, new_image_id=image_id
        )

        if load:
            img = Image.open(image_source)
            spi = SpatialImage(img)
        else:
            spi = SpatialImage(image_source)

        new_row = biocframe.BiocFrame(
            {
                "sample_id": sample_id,
                "image_id": image_id,
                "data": spi,
                "scale_factor": scale_factor,
            }
        )
        new_img_data = self._img_data.combine_rows(new_row)

        return self.__init__(
            assays=self.get_assays(),
            row_ranges=self.get_row_ranges(),
            row_data=self.get_row_data(),
            column_data=self.get_column_data(),
            row_names=self.get_row_names(),
            column_names=self.get_column_names(),
            metadata=self.get_metadata(),
            reduced_dims=self.get_reduced_dims(),
            main_experiment_name=self.get_main_experiment_name(),
            alternative_experiments=self.get_alternative_experiments(),
            row_pairs=self.get_row_pairs(),
            column_pairs=self.get_column_pairs(),
            spatial_coords=self.get_spatial_coordinates(),
            img_data=new_img_data,
        )

    def rmv_img(
        self,
        sample_id: Union[str, True, None] = None,
        image_id: Union[str, True, None] = None,
    ) -> "SpatialExperiment":
        raise NotImplementedError()

    def img_source(
        self,
        sample_id: Union[str, True, None] = None,
        image_id: Union[str, True, None] = None,
        path=False,
    ):
        raise NotImplementedError(
            "This function is irrelevant because it is for `RemoteSpatialImages`"
        )

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
