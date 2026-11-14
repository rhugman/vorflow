"""Ghost-node corrections: the records and their MODFLOW 6 and MODFLOW-USG package input.

Nothing here imports FloPy or modifies a model; the methods return FloPy constructor arguments. Node numbers are
zero-based.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GncCorrections:
    """Ghost-node correction records from :meth:`vorflow.disu.DisuGrid.estimate_gnc`.

    Each record has zero-based ``n`` (the cell holding the ghost), ``m`` (the other end of the connection),
    ``contributors`` and ``weights``: the ghost head is ``h[n] + sum(w * (h[j] - h[n]))``. ``target`` is the ghost
    point, ``link`` the connection index in :attr:`DisuGrid.links` and ``kind`` "vertical" or "horizontal". A
    connection can have one record, or two (one ghost in each cell). ``unsupported`` lists the connections that
    could not be corrected (they keep the standard conductance) with the reason.
    """

    records: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    nodes: int = 0

    @property
    def corrected_connections(self):
        """Number of corrected connections (a connection with two records counts once)."""
        return len({tuple(sorted((r["n"], r["m"]))) for r in self.records})

    @property
    def record_count(self):
        """Number of ghost records: NUMGNC in both MODFLOW 6 and MODFLOW-USG."""
        return len(self.records)

    def _rows(self):
        """(n, m, contributors, weights) per record, all padded to the same number of contributors.

        Padding splits the first contributor's weight over repeated copies of it, which leaves the ghost head
        unchanged.
        """
        width = max((len(r["weights"]) for r in self.records), default=0)
        rows = []
        for r in self.records:
            js, ws = list(r["contributors"]), list(r["weights"])
            repeat = width - len(ws) + 1
            rows.append((r["n"], r["m"], [js[0]] * repeat + js[1:], [ws[0] / repeat] * repeat + ws[1:]))
        return width, rows

    def get_gridprops_mf6_gnc(self, explicit=False):
        """Keyword arguments for ``flopy.mf6.ModflowGwfgnc``, or None when nothing needs correcting.

        Implicit (the default) needs ``linear_acceleration="BICGSTAB"`` in IMS, because the matrix is no longer
        symmetric. Do not combine GNC with XT3D in the same model.
        """
        width, rows = self._rows()
        if not rows:
            return None
        return {"explicit": bool(explicit), "numgnc": len(rows), "numalphaj": width,
                "gncdata": [((n,), (m,), *[(j,) for j in js], *ws) for n, m, js, ws in rows]}

    def get_gridprops_mfusg_gnc(self, explicit=True):
        """Keyword arguments for ``flopy.mfusg.MfUsgGnc``, or None when nothing needs correcting.

        FloPy writes the weights in single precision. After ``model.write_input()``, call
        :meth:`write_mfusg_gnc` with the package's ``fn_path`` and the same ``explicit`` to write them in full.
        """
        width, rows = self._rows()
        if not rows:
            return None
        return {"numgnc": len(rows), "numalphaj": width, "isymgncn": int(explicit),
                "gncdata": [(n, m, *js, *ws) for n, m, js, ws in rows]}

    def write_mfusg_gnc(self, filepath, explicit=True):
        """Write the MODFLOW-USG GNC file (one-based nodes, 17-digit weights); returns the path, or None if empty."""
        width, rows = self._rows()
        if not rows:
            return None
        lines = ["# GNC written by vorflow: one-based nodes, double-precision weights",
                 f"0 0 {len(rows)} {width} 0 {int(explicit)} 0"]
        lines += [" ".join([str(n + 1), str(m + 1), *[str(j + 1) for j in js], *[f"{w:.17g}" for w in ws]])
                  for n, m, js, ws in rows]
        path = Path(filepath)
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path
