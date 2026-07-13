Installation
============

Requirements
------------

* Python ≥ 3.9 (development / acceptance primarily on **3.11**)
* NumPy ``>=1.24,<2.1``

From source (recommended)
-------------------------

.. code-block:: bash

   git clone https://github.com/go223-pyAOBS/pyAOBS.git
   cd pyAOBS
   git checkout v3.0.0rc2
   pip install -e ".[gui-qt]"
   # optional petrology thermo:
   # pip install -e ".[gui-qt,petrology]"
   # everything optional:
   # pip install -e ".[full]"

From a Git tag (Direct URL)
---------------------------

Modern pip does **not** accept ``#egg=pyAOBS[gui-qt]`` (raises ``invalid-egg-fragment``).
Use:

.. code-block:: bash

   pip install "pyAOBS[gui-qt] @ git+https://github.com/go223-pyAOBS/pyAOBS.git@v3.0.0rc2"

Core package only:

.. code-block:: bash

   pip install "pyAOBS @ git+https://github.com/go223-pyAOBS/pyAOBS.git@v3.0.0rc2"

Extras
------

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Extra
     - Packages
     - Use
   * - gui-qt
     - PySide6
     - Workbench, imodel Qt, LIP GUI
   * - petrology
     - burnman≥1.0
     - Thermodynamic properties
   * - gmt
     - pygmt≥0.5
     - GMT plotting
   * - full
     - gui-qt + petrology + pygmt + pyproj + obspy
     - Full optional set

Core dependencies (installed automatically) include numpy, xarray, scipy,
matplotlib, pandas, scikit-learn, seaborn, openpyxl, dearpygui — see ``setup.py``.

Binaries / external programs
----------------------------

* **TOMO2D**: compile the C++ kernel or set ``PYAOBS_TOMO2D_BIN`` / ``TOMO2D_BIN``.
  See ``pyAOBS/modeling/tomo2d/src/README_OMP_BUILD.md``.
* **RAYINVR**: external Rayinvr ecosystem; Python wrappers + ``vedit`` editor in-tree.