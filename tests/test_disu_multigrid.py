"""Different plan grids per layer: splice, global columns, overlay links and gap-free contacts."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from vorflow import disu


def _boxes(n, size, start_id=0):
    polygons = [box(i * size, j * size, (i + 1) * size, (j + 1) * size) for j in range(n) for i in range(n)]
    return gpd.GeoDataFrame({
        "node_id": np.arange(start_id, start_id + len(polygons)),
        "x": [p.centroid.x for p in polygons], "y": [p.centroid.y for p in polygons],
    }, geometry=polygons)


PIT = box(1, 1, 3, 3)


@pytest.fixture
def grids():
    base = _boxes(4, 1.0)                    # 16 cells of 1 m
    fine = _boxes(8, 0.5, start_id=500)      # 64 cells of 0.5 m
    return base, disu.splice(base, fine, PIT)


def test_splice_keeps_base_outside_and_fine_inside(grids):
    base, pit = grids
    assert len(pit) == 12 + 16
    outside = pit[pit.splice_source == "base"]
    assert list(outside.node_id) == list(outside.source_node_id)
    assert all(a.equals(b) for a, b in zip(outside.geometry, base.set_index("node_id").loc[outside.node_id].geometry))
    inside = pit[pit.splice_source == "fine"]
    assert inside.node_id.min() == 16 and inside.node_id.is_unique
    assert inside.geometry.area.sum() == pytest.approx(PIT.area)


def test_splice_rejects_cells_crossing_the_outline():
    base, fine = _boxes(4, 1.0), _boxes(8, 0.5)
    with pytest.raises(ValueError, match="cross the splice outline"):
        disu.splice(base, fine, box(0.5, 0.5, 3, 3))
    with pytest.raises(ValueError, match="cross the splice outline"):
        disu.splice(base, fine, box(1, 1, 2.75, 2.75))
