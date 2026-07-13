API reference
=============

Autodoc is enabled for stable library surfaces. Many GUI modules require
optional dependencies (PySide6, etc.) and are mocked at build time.

Model building
--------------

.. automodule:: pyAOBS.model_building
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

For detailed class docs, prefer the source and the GitHub README examples::

   from pyAOBS.model_building import ZeltVelocityModel2d, EnhancedZeltModel

Package version
---------------

.. code-block:: python

   import pyAOBS
   print(pyAOBS.__version__)  # 3.0.0rc2

Further reading
---------------

* Repository README (complete feature guide)
* ``pyAOBS/workbench/README.md``
* ``pyAOBS/petrology/README.md``
* ``pyAOBS/utils/README.md``
* ``pyAOBS/visualization/IMODEL_README.md``
