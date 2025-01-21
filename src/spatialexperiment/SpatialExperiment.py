from typing import Any, Dict, List, Optional, Union
from warnings import warn

from PIL import Image
import biocframe
from summarizedexperiment.RangedSummarizedExperiment import GRangesOrGRangesList
from singlecellexperiment import SingleCellExperiment
from SpatialImage import SpatialImage

from utils import flatten_list
from _validators import _validate_sample_image_ids, _validate_spatial_coords, _validate_img_data

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
        img_data: Optional[biocframe.BiocFrame] = None,
        image_sources: Optional[Union[str, List[str]]] = None,
        image_id: Optional[List[str]] = None,
        load_image: bool = False,
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

        current_column_data = column_data

        # if `column_data` does not have a column named `sample_id`, assign the value from the `sample_id` argument
        if not column_data.has_column("sample_id"):
            current_column_data = column_data.set_column(
                "sample_id", [sample_id] * len(column_data)
            )

        # `spatial_coords_names` takes precedence
        if spatial_coords_names is not None:
            missing_names = [
                name
                for name in spatial_coords_names
                if name not in current_column_data.column_names
            ]
            if missing_names:
                raise ValueError(
                    f"The following names in `spatial_coords_names` are missing from `column_data`: {missing_names}"
                )

            extracted_spatial_coords = deepcopy(
                current_column_data[:, spatial_coords_names]
            )

            current_column_data = deepcopy(
                current_column_data[
                    :,
                    [
                        col
                        for col in current_column_data.column_names
                        if col not in spatial_coords_names
                    ],
                ]
            )

            self._spatial_coords = extracted_spatial_coords
        else:
            self._spatial_coords = spatial_coords

        if img_data is not None:
            self._img_data = img_data
        else:
            # NOTE: ignoring wheter `image_id`, `image_sources` and `scale_factors` could be lists
            _img_data = {
                "sample_id": sample_id,
                "image_id": image_id if image_id is not None else sample_id,
                "data": SpatialImage(image_sources),
                "scale_factor": scale_factors,
            }
            self._img_data = biocframe.BiocFrame(_img_data)

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

        # TODO: include SpatialExperiment variables too
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

    ##############################
    #####>> spatial_coords <<#####
    ##############################

    def get_spatial_coordinates(self) -> biocframe.BiocFrame:
        """Access spatial coordinates.
        
        Returns:
            A BiocFrame object containing columns of spatial coordinates.
        """
        return self._spatial_coords

    def get_spatial_coords(self) -> biocframe.BiocFrame:
        """Alias for :py:meth:`~get_spatial_coordinates`."""
        return self.get_spatial_coordinates()

    def set_spatial_coordinates(self, spatial_coords: biocframe.BiocFrame, in_place: bool = False) -> "SpatialExperiment":
        """Set new spatial coordinates.
        
        Args:
            spatial_coords (biocframe.BiocFrame):
                New spatial coordinates.
            
            in_place (bool): Whether to modify the ``SpatialExperiment`` in place. Defaults to False.
        
        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        _validate_spatial_coords(spatial_coords, self.shape)

        output = self._define_output(in_place)
        output._spatial_coords = spatial_coords
        return output

    def set_spatial_coords(self, spatial_coords: biocframe.BiocFrame, in_place: bool = False) -> "SpatialExperiment":
        """Alias for :py:meth:`~set_spatial_coordinates`."""
        return self.set_spatial_coordinates(spatial_coords=spatial_coords, in_place=in_place)

    @property
    def spatial_coords(self) -> biocframe.BiocFrame:
        """Alias for :py:meth:`~get_spatial_coordinates`."""
        return self.get_spatial_coordinates()

    @spatial_coords.setter
    def spatial_coords(self, spatial_coords: biocframe.BiocFrame):
        """Alias for :py:meth:`~set_spatial_coordinates`."""
        warn(
            "Setting property 'spatial_coords' is an in-place operation, use 'set_spatial_coordinates' instead.",
            UserWarning
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
            UserWarning
        )
        self.set_spatial_coordinates(spatial_coords=spatial_coords, in_place=True)

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

    def set_image_data(self, img_data: biocframe.BiocFrame, in_place: bool = False) -> "SpatialExperiment":
        """Set new image data.
        
        Args:
            img_data (biocframe.BiocFrame):
                New image data.
                
            in_place (bool): Whether to modify the ``SpatialExperiment`` in place. Defaults to False.
            
        Returns:
            A modified ``SpatialExperiment`` object, either as a copy of the original or as a reference to the (in-place-modified) original.
        """
        _validate_img_data(img_data)

        output = self._define_output(in_place)
        output._img_data = img_data
        return output

    def set_img_data(self, img_data: biocframe.BiocFrame, in_place: bool = False) -> "SpatialExperiment":
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
            UserWarning
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
            UserWarning
        )
        self.set_image_data(img_data=img_data, in_place=True)


    ################################
    ######>> img_data funcs <<######
    ################################

    def get_img(
        self,
        sample_id: Union[str, True, None] = None,
        image_id: Union[str, True, None] = None,
    ) -> Union[SpatialImage, List[SpatialImage]]:
        # TODO: validate that `sample_id` and `image_id` are one of Union[str, True, None]

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
        load: bool = True
    ) -> "SpatialExperiment":
        _validate_sample_image_ids(img_data=self._img_data, new_sample_id=sample_id, new_image_id=image_id)

        if load:
            img = Image.open(image_source)
            spi = SpatialImage(img)
        else:
            spi = SpatialImage(image_source)

        new_row = biocframe.BiocFrame({
            "sample_id": sample_id,
            "image_id": image_id,
            "data": spi,
            "scale_factor": scale_factor
        })
        new_img_data = self._img_data.combine_rows(new_row)

        self.__init__(
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
            img_data=new_img_data
        )

    def rmv_img(
        self,
        sample_id: Union[str, True, None] = None,
        image_id: Union[str, True, None] = None
    ) -> "SpatialExperiment":
        raise NotImplemented()

    def img_source(
        self,
        sample_id: Union[str, True, None] = None,
        image_id: Union[str, True, None] = None,
        path=False
    ):
        raise NotImplemented("This function is irrelevant because it is for `RemoteSpatialImages`")

    def img_raster(self, sample_id=None, image_id=None):
        # NOTE: this function seems redundant, might be an artifact of the different subclasses of SpatialImage in the R implementation? just call `get_img()` for now
        self.get_img(sample_id=sample_id, image_id=image_id)

    def rotate_img(self, sample_id=None, image_id=None, degrees=90):
        raise NotImplemented()

    def mirror_img(self, sample_id=None, image_id=None, axis=("h", "v")):
        raise NotImplemented()
