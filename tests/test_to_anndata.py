import pytest
from copy import deepcopy
from pathlib import Path
import anndata as ad

from spatialexperiment import SpatialExperiment

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_to_anndata(spe):
    obj, alt_exps = spe.to_anndata()

    assert obj.shape == (500, 200)

    # check that uns has the correct components
    assert 'spatial' in obj.uns
    assert len(obj.uns['spatial']) == 3

    library_id = list(obj.uns['spatial'])[0]
    assert 'images' in obj.uns['spatial'][library_id]
    assert 'scalefactors' in obj.uns['spatial'][library_id]

    assert 'hires' in obj.uns['spatial'][library_id]['images']
    assert isinstance(obj.uns['spatial'][library_id]['images']['hires'], Path)

    # check that obsm has the correct components
    assert 'spatial' in obj.obsm
    assert obj.obsm['spatial'].shape == (500, 2)


def test_to_anndata_empty():
    tspe = SpatialExperiment()

    obj, alt_exps = tspe.to_anndata()

    assert isinstance(obj, ad.AnnData)


def test_to_anndata_spatial_key_exists(spe):
    tspe = deepcopy(spe)

    tspe.metadata['spatial'] = "123"

    with pytest.raises(ValueError):
        tspe.to_anndata()
