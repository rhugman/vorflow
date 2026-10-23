"""Solver-independent shared-face validation used by meshing and exports."""

import numpy as np


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 0 else []
    if hasattr(geometry, "geoms"):
        return [line for part in geometry.geoms for line in line_parts(part)]
    return []


def straight_face(shared, centres, pair, tolerance):
    """Combine contiguous collinear pieces, rejecting bent/disconnected faces."""
    parts = line_parts(shared)
    if not parts:
        return None
    longest = max(parts, key=lambda part: part.length)
    xy = np.asarray(longest.coords)[:, :2]
    origin = xy[0]
    direction = xy[-1] - origin
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError(f"Unsupported shared face between cells {pair}.")
    direction /= norm
    normal = np.array([-direction[1], direction[0]])
    intervals = []
    for part in parts:
        offsets = np.asarray(part.coords)[:, :2] - origin
        if np.max(np.abs(offsets @ normal)) > tolerance:
            raise ValueError(f"Noncollinear shared face between cells {pair}.")
        projected = offsets @ direction
        intervals.append((projected.min(), projected.max()))
    intervals.sort()
    end = intervals[0][1]
    for start, stop in intervals[1:]:
        if start > end + tolerance:
            raise ValueError(f"Disconnected shared face between cells {pair}.")
        end = max(end, stop)
    signed = (centres - origin) @ normal
    if signed[0] * signed[1] >= 0 or not np.isfinite(signed).all():
        raise ValueError(f"Centres must lie on opposite sides of the shared face for cells {pair}.")
    delta = centres[1] - centres[0]
    angle = np.degrees(np.arccos(np.clip(abs(delta @ normal) / np.linalg.norm(delta), 0, 1)))
    return {
        "cell_id_1": pair[0], "cell_id_2": pair[1],
        "shared_length": sum(part.length for part in parts),
        "distance_1": abs(signed[0]), "distance_2": abs(signed[1]),
        "ortho_error": float(angle),
    }
