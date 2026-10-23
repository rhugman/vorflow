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

Logging
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   set_verbosity

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
   grid_barrier_report
   grid_quality_report
   calculate_orthogonality
   calculate_mesh_quality
   summarize_quality
   check_geometry_resolution
   resample_geometry
   cells_on
   faces_on
   grid_to_disv

Barrier source layouts and optional repairs
-------------------------------------------

``cells_on`` returns a GeoDataFrame for zero, one or many matches, with
``cell_id`` as zero-based row positions and the original index retained.
``faces_on`` takes connectivity from ``build_connectivity`` and uses the
additive ``full_shared_edge`` geometry to select and measure whole interfaces.
Its returned active geometry is the complete shared edge. Legacy manually
constructed connectivity without that column is treated as already complete.
Neither lookup assigns hydraulic properties or certifies numerical suitability.

Keep authoring geometry with ``ConceptualMesh.add_line`` and ``add_polygon``.
On a line, ``is_barrier=True`` selects straddle sites unless ``quad_buffer=True``
selects a quad band. ``barrier_policy="preserve"`` leaves the source cells
unchanged; ``"cut"`` enables clipping. ``barrier_mirrors`` independently controls
reflected generators. With clipping, ``barrier_merge="restore"`` conservatively
restores a failed parent cut; ``"checked"`` retains rejected fragments and
``"none"`` disables merging. Existing defaults remain compatible.

Polygon bands use ``add_polygon(..., quad_buffer=True)``. An ordinary line may
also use ``quad_buffer=True`` or a positive ``straddle_width`` without being a
barrier. No geometry authoring call assigns hydraulic resistance. Multiple
features and multipart input remain supported; inspect crossing diagnostics
where buffered features intersect.

``MeshGenerator.source_report`` records actual source-buffer coverage and
Gmsh element types and meshing modes. ``VoronoiTessellator.quality_report``
separates hard cell/shared-face defects from quality warnings. Neither certifies
head or flow accuracy. ``barrier_report(approximate=True)`` additionally proposes
real shared-face approximations, with offsets and topology diagnostics, without
changing the grid or assigning hydraulic resistance.
``closed_ties="majority"`` explicitly assigns closed-loop centre ties by polygon
area; balanced areas go outside. The default ``"exclude"`` retains conservative
exclusion. ``approximate_ties`` identifies every choice, while usable-route
topology and offsets in both directions expose incomplete or unsafe routes.
