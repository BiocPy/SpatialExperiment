import pytest
from PIL import Image
from spatialexperiment import ProxySpatialFeatureExperiment
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

def test_init_basic():
    nrows = 200
    ncols = 500
    counts = np.random.rand(nrows, ncols)
    tspe = ProxySpatialFeatureExperiment(assays={"spots": counts})

    assert isinstance(tspe, ProxySpatialFeatureExperiment)

def test_init_empty():
    tspe = ProxySpatialFeatureExperiment()

    assert isinstance(tspe, ProxySpatialFeatureExperiment)

def test_init_with_col_geoms():
    nrows = 200
    ncols = 500
    counts = np.random.rand(nrows, ncols)
    polys = gpd.GeoSeries(
        [
            Polygon([(1, -1), (1, 0), (0, 0)]),
            Polygon([(3, -1), (4, 0), (3, 1)]),
        ]
    )

    colgeoms = {"polygons" : gpd.GeoDataFrame({"geometry": polys})}
    tspe = ProxySpatialFeatureExperiment(assays={"spots": counts}, col_geometries=colgeoms)

    assert isinstance(tspe, ProxySpatialFeatureExperiment)