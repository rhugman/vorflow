"""DISU array serialization from a link table (see ``_connect3d``).

``mf6_gridprops`` returns the keyword arguments of ``flopy.mf6.ModflowGwfdisu``
(behind ``DisuGrid.get_gridprops_mf6_disu``), with the keys of
``flopy.utils.gridgen.Gridgen.get_gridprops_disu6``.
"""

import numpy as np


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
