API reference
=============

The pipeline runs three classes in order: :class:`~vorflow.ConceptualMesh`
collects and cleans the features, :class:`~vorflow.MeshGenerator` meshes them
with Gmsh, and :class:`~vorflow.VoronoiTessellator` builds the Voronoi grid.

Pipeline
--------

.. currentmodule:: vorflow

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConceptualMesh
   MeshGenerator
   VoronoiTessellator

Mesh-size fields
----------------

Pass these to the ``fields`` argument of ``add_polygon``, ``add_line`` or
``add_point`` to replace the default size transition around a feature.

.. autosummary::
   :toctree: generated
   :nosignatures:

   MeshField
   GeometricGrowthField
   ExponentialField
   ThresholdField

:class:`~vorflow.GapField` is model-wide: pass it to
``MeshGenerator(gap_field=...)``, not to a feature's ``fields``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   GapField

Logging
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   set_verbosity

Fully unstructured DISU
-----------------------

``vorflow.disu`` builds MODFLOW 6 DISU grids with full connectivity from
vorflow grids and FloPy-style layer arrays; import it as
``from vorflow import disu``.

.. currentmodule:: vorflow.disu

.. autosummary::
   :toctree: generated
   :nosignatures:

   DisuGrid

Grid quality and utilities
--------------------------

``vorflow.utils`` is not re-exported by ``vorflow``; import it as
``from vorflow.utils import ...``.

.. currentmodule:: vorflow.utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_connectivity
   boundary_connectivity_report
   calculate_orthogonality
   calculate_mesh_quality
   summarize_quality
   check_geometry_resolution
   resample_geometry
   grid_to_disv
   grid_to_mfusg_disu
   grid_to_gsf
