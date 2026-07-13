Quick start
===========

Workbench
---------

.. code-block:: bash

   python -m pyAOBS.workbench.app

Tk fallback if PySide6 is unavailable:

.. code-block:: bash

   # PowerShell
   $env:PYAOBS_WORKBENCH_UI="tk"
   python -m pyAOBS.workbench.app

Zelt velocity model (API)
-------------------------

.. code-block:: python

   from pyAOBS.model_building import ZeltVelocityModel2d, EnhancedZeltModel

   model = ZeltVelocityModel2d("velocity.in")
   v = model.at(100.0, 1.5)

   enhanced = EnhancedZeltModel("velocity.in")
   avg = enhanced.compute_average_velocities()

Plot a Zelt model
-----------------

.. code-block:: python

   from pyAOBS.model_building import ZeltVelocityModel2d
   from pyAOBS.visualization import ZeltModelVisualizer

   model = ZeltVelocityModel2d("velocity.in")
   ZeltModelVisualizer(model).plot_zeltmodel(
       output_file="velocity_model.png",
       title="Velocity Model",
       colorbar_label="Velocity (km/s)",
   )

SU I/O
------

.. code-block:: python

   from pyAOBS import readsu, writesu, plotsu

Standalone GUIs
---------------

.. code-block:: bash

   python -m pyAOBS.visualization.imodel_qt
   python -m pyAOBS.visualization.zplotpy.gui
   python -m pyAOBS.visualization.iphase.iphase_gui
   python -m pyAOBS.modeling.tomo2d.gui
   python -m pyAOBS.petrology.gui
