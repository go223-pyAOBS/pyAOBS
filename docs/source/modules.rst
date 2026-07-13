Modules overview
================

This page mirrors the GitHub ``README.md`` module map. For the full Chinese
guide (tables, workflows, links), open the repository home page.

Workbench (``pyAOBS.workbench``)
--------------------------------

Project GUI: plugins for ``data.gui``, ``zplotpy.gui``, ``tomo2d``, ``imodel.gui``,
``iphase.gui``, ``petrology.lip.gui``.

Manual: ``pyAOBS/workbench/README.md``

Visualization (``pyAOBS.visualization``)
----------------------------------------

* **imodel Qt** — velocity interpretation, lithology, gravity, petrology export
* **zplotpy** — phase picking / Fast Viewer
* **iphase** — ``tx.in`` select / combine / QC
* **show_model** — ``ZeltModelVisualizer``, ``GridModelVisualizer``

Model building (``pyAOBS.model_building``)
--------------------------------------------

``ZeltVelocityModel2d``, ``EnhancedZeltModel``, ``SlownessMesh2D``, SU export helpers.

Modeling (``pyAOBS.modeling``)
------------------------------

* **tomo2d** — Python wrappers + GUI + C++ ``tt_forward`` / ``tt_inverse``
* **rayinvr** — ray tracing wrappers
* **vedit** — Tk ``v.in`` editor (``python pyAOBS/modeling/vedit/main.py``)
* **hybrid** — research scripts (ML / scanning)

Processors (``pyAOBS.processors``)
----------------------------------

* **raw2sac / idata** — RAW, OBEM TSM, SAC, SEGY conversion
* **SU** — ``readsu`` / ``writesu`` / …
* **denoise** — section/trace denoise backends for zplot
* **relocation** — OBS orientation (incl. vendored Ppol)

Utils (``pyAOBS.utils``)
------------------------

Rock database, ``isrock``, ``classify_velocity_model``, empirical formulas.
Manual: ``pyAOBS/utils/README.md``

.. note::

   imodel lithology reference state is often **200 MPa / 25°C**;
   petrology forward models use **600 MPa / 400°C**. Do not mix them.

Petrology (``pyAOBS.petrology``)
--------------------------------

KKHS02-style Steps 1–4: eq.(1), bulk bounds, fractional crystallization ΔVp
(raw W&L+Langmuir by default), active-upwelling H–Vp; Modern track optional.

Manual: ``pyAOBS/petrology/README.md``

Field (``pyAOBS.field``)
------------------------

OBS station layout / recovery path planners (Tk). Not wired into Workbench.
