"""CI entry points for .github/workflows/benchmark.yml, one per job.

    python ci.py windows   # VOROGRIDGEN builds and MF6 problems (Windows runner)
    python ci.py others    # vorflow, mf6Voronoi and FloPy builds and MF6 problems
    python ci.py report    # compile both jobs' result rows, draw the figures

Locally, edit and run workflow.py instead.
"""

import sys

import workflow

JOBS = {
    "windows": {"tools": ("vorogridgen",), "report": False},
    "others": {"tools": ("vorflow", "vorflow_lloyd", "mf6voronoi", "flopy"), "report": False},
    "report": {"build": False, "report": True},
}

if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in JOBS, f"usage: python ci.py {{{'|'.join(JOBS)}}}"
    workflow.main(**JOBS[sys.argv[1]])
