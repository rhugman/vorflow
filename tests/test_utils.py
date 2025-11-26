import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from vorflow.utils import calculate_mesh_quality


@pytest.fixture
def paired_polygons():
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])

    gdf = gpd.GeoDataFrame(
        {
            "geometry": [poly_a, poly_b],
            "x": [0.5, 1.5],
            "y": [0.5, 0.5],
        }
    )
    return gdf


def test_calculate_mesh_quality_computes_expected_scalar_metrics(paired_polygons):
    quality = calculate_mesh_quality(paired_polygons, calc_ortho=True)

    assert "area" in quality.columns
    assert "compactness" in quality.columns
    assert "ortho_error" in quality.columns

    assert quality["area"].iloc[0] == pytest.approx(1.0)
    assert quality["perimeter"].iloc[0] == pytest.approx(4.0)
    assert quality["compactness"].iloc[0] == pytest.approx(np.pi / 4, rel=1e-5)
    assert quality["drift_ratio"].iloc[0] == pytest.approx(0.0, abs=1e-8)
    assert quality["ortho_error"].iloc[0] == pytest.approx(0.0, abs=1e-6)
