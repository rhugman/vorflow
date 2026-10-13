"""One adapter per tool: ``build(case, scale, ws, **tool_kwargs) -> Grid``.

Adapters are imported on demand, so a job that runs one tool (the Windows CI
job runs only VOROGRIDGEN) never needs the others installed.
"""

import importlib

TOOLS = ("vorflow", "vorflow_lloyd", "mf6voronoi", "flopy", "vorogridgen")


def get_adapter(tool: str):
    """Import and return the adapter module for one tool."""
    assert tool in TOOLS, f"unknown tool {tool!r}; expected one of {TOOLS}"
    return importlib.import_module(f"{__name__}.{tool}_adapter")
