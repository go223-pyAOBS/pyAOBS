Welcome to pyAOBS's documentation!
==================================

.. image:: ../../images/logo.png
   :width: 420
   :align: center
   :alt: pyAOBS logo

**pyAOBS** (Python Active-source Ocean Bottom Seismology) is a toolkit for
**active-source OBS** workflows: data conversion, phase picking, traveltime
tomography, velocity-model interpretation, lithology classification, and
KKHS02 / LIP petrology constraints — with a unified **Workbench** and
standalone GUIs / APIs.

.. list-table::
   :widths: 25 75
   :stub-columns: 1

   * - **Release**
     - ``3.0.0rc2``
   * - **Repository**
     - https://github.com/go223-pyAOBS/pyAOBS
   * - **Changelog**
     - https://github.com/go223-pyAOBS/pyAOBS/blob/main/CHANGELOG.md
   * - **GitHub README**
     - Full Chinese feature guide on the repository home page

.. note::

   The **authoritative, complete feature description** lives in the repository
   root ``README.md`` (shown on the GitHub home page). This documentation site
   summarizes the same scope and points to subpackage manuals.

What is included
----------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Module
     - Purpose
     - Main entry
   * - Workbench
     - Project shell: runs, history, batch rerun
     - ``python -m pyAOBS.workbench.app``
   * - idata
     - OBS format conversion (RAW / OBEM / SAC / SEGY)
     - Workbench ``data.gui``
   * - zplotpy
     - Phase picking / section viewer
     - ``python -m pyAOBS.visualization.zplotpy.gui``
   * - tomo2d
     - Traveltime forward / inversion
     - ``python -m pyAOBS.modeling.tomo2d.gui``
   * - imodel
     - Velocity-model interpretation (Qt)
     - ``python -m pyAOBS.visualization.imodel_qt``
   * - iphase
     - ``tx.in`` phase tools
     - ``python -m pyAOBS.visualization.iphase.iphase_gui``
   * - model_building
     - Zelt ``v.in`` / TOMO2D mesh APIs
     - Python API
   * - petrology
     - KKHS02 melting / H–Vp / ΔVp / invert
     - ``python -m pyAOBS.petrology.gui``
   * - utils
     - Rock database & Vp→lithology
     - ``classify_velocity_model``
   * - field
     - OBS station layout / recovery paths
     - ``field/*_gui.py``

Recommended OBS workflow
------------------------

::

   data.gui → zplotpy.gui → tomo2d → imodel.gui → iphase.gui → (optional) petrology.lip.gui

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   modules
   api
   contributing
   authors
   history

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`