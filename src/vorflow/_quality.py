"""Solver-independent diagnostics for polygon cells and their shared faces."""

from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point

from ._geometry import line_parts, straight_face


def _distance(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a finite {'positive' if positive else 'non-negative'} distance."
        )
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be a finite {'positive' if positive else 'non-negative'} distance."
        )
    return value


def _domain_geometry(domain, crs):
    if isinstance(domain, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if crs is not None and domain.crs is not None and crs != domain.crs:
            raise ValueError(
                "Grid and domain CRS must match; reproject the domain first."
            )
        if domain.crs is not None and domain.crs.is_geographic:
            raise ValueError(
                "Grid quality needs a planar domain; reproject the geographic CRS first."
            )
        geometries = domain.geometry
        if geometries.empty or any(
            geom is None
            or geom.is_empty
            or geom.geom_type not in {"Polygon", "MultiPolygon"}
            or not geom.is_valid
            or not np.isfinite(shapely.get_coordinates(geom)).all()
            for geom in geometries
        ):
            raise ValueError("domain must contain valid, finite, non-empty polygons.")
        domain = shapely.union_all(geometries.to_numpy())
    if domain is not None and (
        not isinstance(domain, shapely.Geometry)
        or domain.is_empty
        or domain.geom_type not in {"Polygon", "MultiPolygon"}
        or not domain.is_valid
        or not np.isfinite(shapely.get_coordinates(domain)).all()
    ):
        raise ValueError(
            "domain must be a valid, finite, non-empty Polygon or MultiPolygon."
        )
    return domain


def _face_record(shared, centres, pair, tolerance):
    """Keep every shared segment, including unsupported bent/disjoint faces."""
    parts = line_parts(shared)
    record = dict(
        cell_id_1=pair[0],
        cell_id_2=pair[1],
        geometry=shared,
        shared_length=sum(part.length for part in parts),
        distance_1=np.nan,
        distance_2=np.nan,
        ortho_error=np.nan,
        intersection_fraction=np.nan,
        midpoint_skew=np.nan,
        connector_misses_face=False,
        bent=False,
        disconnected=False,
        overlap_area=0.0,
        roundoff_overlap=False,
        roundoff_gap=False,
        gap_distance=0.0,
        near_contact_length=0.0,
        reasons=[],
        warnings=[],
    )
    if not parts:
        return record
    # A reference normal is only a direction: straight_face checks ALL pieces
    # for collinearity and continuity; no shorter segment is discarded.
    longest = max(parts, key=lambda part: part.length)
    xy = np.asarray(longest.coords)[:, :2]
    tangent = xy[-1] - xy[0]
    length = np.linalg.norm(tangent)
    if not np.isfinite(length) or length == 0:
        record["reasons"].append("unsupported_shared_face")
        return record
    tangent /= length
    normal = np.array([-tangent[1], tangent[0]])
    origin = xy[0]
    try:
        straight_face(
            shared, np.array([origin - normal, origin + normal]), pair, tolerance
        )
    except ValueError as error:
        message = str(error)
        reason = (
            "bent_face"
            if "Noncollinear" in message
            else (
                "disconnected_face"
                if "Disconnected" in message
                else "unsupported_shared_face"
            )
        )
        record["bent"] = reason == "bent_face"
        record["disconnected"] = reason == "disconnected_face"
        record["reasons"].append(reason)
        return record
    if not np.isfinite(centres).all():
        record["reasons"].append("nonfinite_centres")
        return record
    signed = (centres - origin) @ normal
    record["distance_1"], record["distance_2"] = np.abs(signed)
    if np.any(np.abs(signed) <= 0):
        record["reasons"].append("nonpositive_face_distance")
    try:
        record.update(straight_face(shared, centres, pair, tolerance))
    except ValueError:
        record["reasons"].append("centres_not_opposite")
        return record
    all_xy = np.vstack([np.asarray(part.coords)[:, :2] for part in parts])
    projection = (all_xy - origin) @ tangent
    start = origin + projection.min() * tangent
    face_length = float(projection.max() - projection.min())
    crossing = centres[0] + signed[0] / (signed[0] - signed[1]) * (
        centres[1] - centres[0]
    )
    fraction = float((crossing - start) @ tangent / face_length)
    record["intersection_fraction"] = fraction
    record["midpoint_skew"] = abs(fraction - 0.5)
    record["connector_misses_face"] = bool(
        fraction < -tolerance / face_length or fraction > 1 + tolerance / face_length
    )
    if record["connector_misses_face"]:
        record["warnings"].append("connector_misses_face")
    if record["ortho_error"] > 1.0:
        record["warnings"].append("nonorthogonal_face")
    return record


def grid_quality_report(gdf, *, cell_size=None, domain=None, tolerance=None):
    """Report cell geometry and shared-face quality without importing a solver.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Polygon cells with numerical-centre columns ``x`` and ``y``. Row order
        defines zero-based cell IDs; an optional ``node_id`` retains source IDs.
    cell_size : float, optional
        Local reference length in coordinate units. Cells below ``0.05*h**2``
        receive a small-area warning; no size threshold is inferred if omitted.
    domain : shapely Polygon or MultiPolygon, optional
        Expected footprint, used to detect uncovered and outside-domain areas.
        A GeoDataFrame/GeoSeries with matching CRS is also accepted.
    tolerance : float, optional
        Roundoff distance, defaulting to ``32*eps*max(1,abs(coordinates))``.
        Source polygons and numerical centres are never modified.

    Returns
    -------
    dict
        ``cells`` and ``faces`` GeoDataFrames plus a ``summary`` dictionary.
        Geometry defects appear as explicit reason tuples and ``hard_valid``
        flags. Nonconvexity, compactness below 0.4, orthogonality error above
        one degree, and connectors missing finite faces are warnings. Passing
        this report does not establish head or flow accuracy.

    Raises
    ------
    ValueError
        For an empty grid, missing centre columns, geographic coordinates,
        incompatible domain CRS, or invalid diagnostic options. Invalid cell
        geometry is reported rather than aborting the report.
    """
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        raise ValueError("Grid quality needs a non-empty GeoDataFrame.")
    if not {"x", "y"}.issubset(gdf.columns):
        raise ValueError("Grid quality requires numerical-centre columns 'x' and 'y'.")
    try:
        gdf.geometry
    except AttributeError:
        raise ValueError("Grid quality requires an active polygon geometry column.") from None
    if gdf.crs is not None and gdf.crs.is_geographic:
        raise ValueError(
            "Grid quality needs planar coordinates; reproject the geographic CRS first."
        )
    cell_size = (
        _distance(cell_size, "cell_size", positive=True)
        if cell_size is not None
        else None
    )
    domain = _domain_geometry(domain, gdf.crs)
    geoms = gdf.geometry.to_numpy()
    coordinates = shapely.get_coordinates(geoms)
    if domain is not None:
        coordinates = np.vstack([coordinates, shapely.get_coordinates(domain)])
    finite = coordinates[np.isfinite(coordinates)]
    if tolerance is None:
        scale = max(1.0, float(np.abs(finite).max())) if len(finite) else 1.0
        tolerance = 32 * np.finfo(float).eps * scale
    else:
        tolerance = _distance(tolerance, "tolerance")
    centres = (
        gdf[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    node_ids = gdf["node_id"] if "node_id" in gdf else pd.Series(np.arange(len(gdf)))
    duplicates = node_ids.duplicated(keep=False).to_numpy()
    cells, usable = [], []
    for position, geom in enumerate(geoms):
        reasons, warnings = [], []
        record = dict(
            cell_id=position,
            orig_index=gdf.index[position],
            node_id=node_ids.iloc[position],
            geometry=geom,
            x=centres[position, 0],
            y=centres[position, 1],
            area=np.nan,
            perimeter=np.nan,
            compactness=np.nan,
            convex=False,
            centre_inside=False,
            small_area=False,
            low_compactness=False,
            reasons=reasons,
            warnings=warnings,
        )
        if duplicates[position]:
            reasons.append("duplicate_node_id")
        if pd.isna(node_ids.iloc[position]):
            reasons.append("missing_node_id")
        if not np.isfinite(centres[position]).all():
            reasons.append("nonfinite_centre")
        if geom is None or geom.is_empty:
            reasons.append("missing_or_empty_geometry")
        else:
            if not np.isfinite(shapely.get_coordinates(geom)).all():
                reasons.append("nonfinite_geometry")
            if geom.geom_type != "Polygon":
                reasons.append("nonpolygon_geometry")
            elif not geom.is_valid:
                reasons.append("invalid_geometry")
            elif "nonfinite_geometry" not in reasons:
                area, perimeter = float(geom.area), float(geom.length)
                record.update(area=area, perimeter=perimeter)
                if (
                    not np.isfinite(area)
                    or not np.isfinite(perimeter)
                    or area <= 0
                    or perimeter <= 0
                ):
                    reasons.append("nonpositive_or_nonfinite_area")
                else:
                    usable.append(position)
                    record["compactness"] = 4 * np.pi * (area / perimeter) / perimeter
                    # Boundary distance catches a deep, very narrow notch
                    # whose missing area alone could look like roundoff.
                    record["convex"] = (
                        geom.boundary.hausdorff_distance(geom.convex_hull.boundary)
                        <= tolerance
                    )
                    record["low_compactness"] = record["compactness"] < 0.4
                    record["small_area"] = (
                        cell_size is not None and area < 0.05 * cell_size * cell_size
                    )
                    for key in ("small_area", "low_compactness"):
                        if record[key]:
                            warnings.append(key)
                    if not record["convex"]:
                        warnings.append("nonconvex_cell")
                if geom.interiors:
                    reasons.append("cell_has_holes")
                if np.isfinite(centres[position]).all():
                    record["centre_inside"] = bool(
                        shapely.dwithin(geom, Point(centres[position]), tolerance)
                    )
                    if not record["centre_inside"]:
                        reasons.append("centre_outside")
        cells.append(record)

    faces, corner_contacts, overlaps = [], 0, 0
    if usable:
        polygons = geoms[usable]
        tree = shapely.STRtree(polygons)
        left, right = tree.query(polygons, predicate="intersects")
        for a, b in sorted(zip(left[left < right], right[left < right])):
            i, j = usable[a], usable[b]
            first, second = geoms[i], geoms[j]
            overlap = first.intersection(second)
            shared = first.boundary.intersection(second.boundary)
            material_overlap = False
            roundoff_overlap = False
            if overlap.area > 0:
                roundoff_overlap = tolerance > 0 and all(
                    overlap.difference(poly.boundary.buffer(tolerance)).is_empty
                    for poly in (first, second)
                )
                material_overlap = not roundoff_overlap
                if roundoff_overlap:
                    shared = shapely.intersection(
                        first.boundary, second.boundary, grid_size=tolerance
                    )
            parts = line_parts(shared)
            if not parts and not material_overlap and not roundoff_overlap:
                corner_contacts += int(not shared.is_empty)
                continue
            shared = shapely.union_all(parts) if parts else shared
            record = _face_record(shared, centres[[i, j]], (i, j), tolerance)
            record.update(
                overlap_area=float(overlap.area),
                roundoff_overlap=bool(roundoff_overlap),
            )
            if material_overlap:
                overlaps += 1
                record["reasons"].append("material_overlap")
                for position in (i, j):
                    cells[position]["reasons"].append("material_overlap")
            elif roundoff_overlap:
                record["warnings"].append("roundoff_overlap")
                if not parts:
                    record["reasons"].append("unresolved_roundoff_overlap")
            faces.append(record)

        # A small domain-area residual can be harmless roundoff while a
        # missing numerical connection is not. Audit near-but-disjoint faces
        # separately; never create a shared face by silently snapping cells.
        if tolerance > 0:
            left, right = tree.query(polygons, predicate="dwithin", distance=tolerance)
            for a, b in sorted(zip(left[left < right], right[left < right])):
                i, j = usable[a], usable[b]
                first, second = geoms[i], geoms[j]
                if first.intersects(second):
                    continue
                nearby = first.boundary.intersection(second.boundary.buffer(tolerance))
                near_length = sum(part.length for part in line_parts(nearby))
                # The short pieces within a tolerance radius of one corner
                # are not faces. A substantial near-boundary interval needs
                # an explicit diagnosis even when the gap area is tiny.
                if near_length <= 4 * tolerance:
                    continue
                record = _face_record(
                    shapely.GeometryCollection(), centres[[i, j]], (i, j), tolerance
                )
                record.update(
                    roundoff_gap=True,
                    gap_distance=float(first.distance(second)),
                    near_contact_length=float(near_length),
                )
                record["reasons"].append("unresolved_roundoff_gap")
                for position in (i, j):
                    cells[position]["reasons"].append("unresolved_roundoff_gap")
                faces.append(record)

    union = shapely.union_all(geoms[usable]) if usable else shapely.GeometryCollection()
    gap_area = outside_area = gap_material_area = outside_material_area = None
    domain_reasons = []
    if domain is not None:
        gap, outside = domain.difference(union), union.difference(domain)
        gap_area, outside_area = float(gap.area), float(outside.area)
        gap_material = (
            gap.difference(union.boundary.buffer(tolerance)) if tolerance else gap
        )
        outside_material = (
            outside.difference(domain.boundary.buffer(tolerance))
            if tolerance
            else outside
        )
        gap_material_area, outside_material_area = (
            float(gap_material.area),
            float(outside_material.area),
        )
        if gap_material_area > 0:
            domain_reasons.append("domain_gap")
        if outside_material_area > 0:
            domain_reasons.append("outside_domain")
    for record in [*cells, *faces]:
        record["reasons"] = tuple(dict.fromkeys(record["reasons"]))
        record["warnings"] = tuple(dict.fromkeys(record["warnings"]))
        record["hard_valid"] = not record["reasons"]
    invalid_cells = sum(not row["hard_valid"] for row in cells)
    invalid_faces = sum(not row["hard_valid"] for row in faces)
    face_columns = list(
        _face_record(shapely.GeometryCollection(), np.zeros((2, 2)), (0, 1), tolerance)
    )
    faces_frame = gpd.GeoDataFrame(
        faces, columns=[*face_columns, "hard_valid"], geometry="geometry", crs=gdf.crs
    )
    return dict(
        cells=gpd.GeoDataFrame(cells, geometry="geometry", crs=gdf.crs),
        faces=faces_frame,
        summary=dict(
            hard_valid=not (invalid_cells or invalid_faces or domain_reasons),
            cells=len(cells),
            faces=len(faces),
            hard_invalid_cells=invalid_cells,
            hard_invalid_faces=invalid_faces,
            overlapping_pairs=overlaps,
            roundoff_gap_pairs=sum(row["roundoff_gap"] for row in faces),
            corner_contacts=corner_contacts,
            total_area=float(
                sum(row["area"] for row in cells if np.isfinite(row["area"]))
            ),
            union_area=float(union.area),
            domain_area=float(domain.area) if domain is not None else None,
            domain_gap_area=gap_area,
            outside_domain_area=outside_area,
            domain_gap_material_area=gap_material_area,
            outside_domain_material_area=outside_material_area,
            domain_reasons=tuple(domain_reasons),
            tolerance=tolerance,
            cell_size=cell_size,
            reason_counts=dict(
                Counter(reason for row in [*cells, *faces] for reason in row["reasons"])
            ),
            warning_counts=dict(
                Counter(
                    warning for row in [*cells, *faces] for warning in row["warnings"]
                )
            ),
            warning_cells=sum(bool(row["warnings"]) for row in cells),
            warning_faces=sum(bool(row["warnings"]) for row in faces),
        ),
    )
