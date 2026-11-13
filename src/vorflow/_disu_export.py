"""DISU array serialization from a link table (see ``_connect3d``).

``mf6_gridprops`` returns the keyword arguments of ``flopy.mf6.ModflowGwfdisu``
and ``mfusg_gridprops`` those of ``flopy.mfusg.MfUsgDisU`` (behind
``DisuGrid.get_gridprops_mf6_disu`` and ``get_gridprops_mfusg_disu``), with the
keys of ``flopy.utils.gridgen.Gridgen.get_gridprops_disu6`` and ``..._disu5``.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _csr(nodes, links, diagonal):
    """Both directions of every link plus one diagonal entry per node, row by row.

    Each row starts with its diagonal, then neighbours in ascending order.
    Returns the entry arrays sorted into rows and ``iac``.
    """
    n, m = links["n"], links["m"]
    # Horizontal faces point the other way from m; vertical links carry no angle.
    reverse = np.where(links["ihc"] > 0, (links["angle"] + 180.0) % 360.0, 0.0)
    entries = {
        "row": np.concatenate([np.arange(nodes), n, m]),
        "col": np.concatenate([np.arange(nodes), m, n]),
        "ihc": np.concatenate([diagonal, links["ihc"], links["ihc"]]),
        "cl12": np.concatenate([np.zeros(nodes), links["cl_n"], links["cl_m"]]),
        "hwva": np.concatenate([np.zeros(nodes), links["hwva"], links["hwva"]]),
        "angldegx": np.concatenate([np.zeros(nodes), links["angle"], reverse]),
        "fahl": np.concatenate([np.zeros(nodes), links["fahl"], links["fahl"]]),
    }
    is_offdiagonal = entries["row"] != entries["col"]
    order = np.lexsort((entries["col"], is_offdiagonal, entries["row"]))
    entries = {key: value[order] for key, value in entries.items()}
    pairs = entries["row"][is_offdiagonal[order]] * nodes + entries["col"][is_offdiagonal[order]]
    if len(np.unique(pairs)) != len(pairs):
        raise ValueError("Duplicate DISU connection; each pair of nodes may be linked once.")
    iac = np.bincount(entries["row"], minlength=nodes).astype(np.int64)
    return entries, iac


def mf6_gridprops(nodes, top, bot, area, layer, links, planar, icpl):
    """MODFLOW 6 DISU arrays.

    Args:
        nodes: Number of model nodes.
        top, bot, area, layer, icpl: Per-node arrays (``icpl`` indexes the
            node's plan cell in ``planar``).
        links: Link table from ``_connect3d``.
        planar: ``grid_to_disv`` output for the plan mesh.

    The layer number (one-based) is written on the IHC diagonal, which
    MODFLOW 6 does not read, so FloPy and post-processors can recover layers.
    """
    entries, iac = _csr(nodes, links, np.asarray(layer, dtype=np.int64) + 1)
    cell2d = [
        [node, *planar["cell2d"][cell][1:]] for node, cell in enumerate(np.asarray(icpl).tolist())
    ]
    return {
        "nodes": int(nodes),
        "nja": int(iac.sum()),
        "nvert": int(planar["nvert"]),
        "top": np.asarray(top, dtype=float).copy(),
        "bot": np.asarray(bot, dtype=float).copy(),
        "area": np.asarray(area, dtype=float).copy(),
        "iac": iac,
        "ja": entries["col"].astype(np.int64),
        "ihc": entries["ihc"].astype(np.int64),
        "cl12": entries["cl12"],
        "hwva": entries["hwva"],
        "angldegx": entries["angldegx"],
        "vertices": [list(vertex) for vertex in planar["vertices"]],
        "cell2d": cell2d,
    }


def mfusg_gridprops(nodes, nlay, top, bot, area, layer, links):
    """MODFLOW-USG DISU arrays, for ``flopy.mfusg.MfUsgDisU``.

    ``ivsd=1`` with an explicit ``ivc``: with ``ivsd`` 0 or -1, MODFLOW-USG
    treats every connection to a node in a lower layer as vertical, so a
    staggered horizontal connection would take its conductance from the
    vertical conductivity (this matters when HK differs from VKA or layers
    are convertible). ``cl12``
    holds each node's own distance to the face (full, non-symmetric storage,
    ``idsymrd=0``) and ``fahl`` the interface area.
    """
    layer = np.asarray(layer, dtype=np.int64)
    if len(layer) and np.any(np.diff(layer) < 0):
        raise ValueError("MODFLOW-USG needs nodes numbered layer by layer.")
    nodelay = np.bincount(layer, minlength=nlay).astype(np.int64)
    if (nodelay == 0).any():
        raise ValueError(
            f"Layer {int(np.flatnonzero(nodelay == 0)[0])} has no active cells; MODFLOW-USG needs "
            "at least one node in every layer."
        )
    entries, iac = _csr(nodes, links, np.zeros(nodes, dtype=np.int64))
    is_diagonal = entries["row"] == entries["col"]
    return {
        "nodes": int(nodes), "nlay": int(nlay), "njag": int(iac.sum()),
        "ivsd": 1, "idsymrd": 0, "nodelay": nodelay,
        "top": np.asarray(top, dtype=float).copy(), "bot": np.asarray(bot, dtype=float).copy(),
        "area": np.asarray(area, dtype=float).copy(),
        "iac": iac, "ja": entries["col"].astype(np.int64),
        "ivc": np.where(is_diagonal, 0, entries["ihc"] == 0).astype(np.int64),
        "cl12": entries["cl12"], "fahl": entries["fahl"],
    }


def mfusg_ksat(nodes, links, k, k33):
    """Per-entry K for MODFLOW-USG ``ikcflag=1``, in the ``ja`` order of ``mfusg_gridprops``.

    Each link gets the distance-weighted harmonic mean of its two nodes' K
    (``k33`` for vertical links); each diagonal entry its node's ``k``.
    """
    vertical = links["ihc"] == 0
    k_n = np.where(vertical, k33[links["n"]], k[links["n"]])
    k_m = np.where(vertical, k33[links["m"]], k[links["m"]])
    ksat = (links["cl_n"] + links["cl_m"]) / (links["cl_n"] / k_n + links["cl_m"] / k_m)
    # Same row order as the DISU arrays: carry ksat through _csr in place of fahl.
    entries, _ = _csr(nodes, {**links, "fahl": ksat}, np.zeros(nodes, dtype=np.int64))
    is_diagonal = entries["row"] == entries["col"]
    return np.where(is_diagonal, k[entries["row"]], entries["fahl"])


def write_gsf(filepath, planar, icpl, top, bot, layer, nlay):
    """Original 3D GSF: one top and one bottom vertex ring per node, all IDs one-based."""
    xy = np.asarray(planar["vertices"])[:, 1:]
    vertices, vertex_ids, cells = [], {}, []
    for node, (cell, z_top, z_bot, k) in enumerate(zip(icpl, top, bot, layer)):
        ring = planar["cell2d"][cell]
        ids = []
        for elevation in (z_top, z_bot):
            for vertex in ring[4:]:
                xyz = (*xy[vertex], elevation)
                if xyz not in vertex_ids:
                    vertex_ids[xyz] = len(vertices) + 1
                    vertices.append(xyz)
                ids.append(vertex_ids[xyz])
        # Midpoint avoids top+bottom overflow for otherwise finite elevations.
        z = z_bot + (z_top - z_bot) / 2
        cells.append((node + 1, ring[1], ring[2], z, int(k) + 1, ids))
    path = Path(filepath)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"UNSTRUCTURED\n{len(cells)} {nlay} 1 1\n{len(vertices)}\n")
        for xyz in vertices:
            stream.write(" ".join(format(float(value), ".17g") for value in xyz) + "\n")
        for node, x, y, z, k, ids in cells:
            stream.write(
                f"{node} {x:.17g} {y:.17g} {z:.17g} {k} {len(ids)} "
                + " ".join(map(str, ids)) + "\n"
            )
    logger.info("Saved original 3D GSF to %s", path)
    return path
