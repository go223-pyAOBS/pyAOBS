.. highlight:: shell

============
Installation
============

From PyPI
---------

To install pyAOBS, run this command in your terminal:

.. code-block:: console

    $ pip install pyAOBS

This is the preferred method to install pyAOBS, as it will always install the most recent stable release.

From Source
----------

The sources for pyAOBS can be downloaded from the `Github repo`_.

You can clone the public repository:

.. code-block:: console

    $ git clone git://github.com/go223-pyAOBS/pyAOBS

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ cd pyAOBS
    $ pip install -e .

.. _Github repo: https://github.com/go223-pyAOBS/pyAOBS

Dependencies
-----------

pyAOBS requires the following Python packages:

* numpy >= 1.20.0
* xarray >= 0.16.0
* scipy >= 1.6.0
* matplotlib >= 3.3.0
* pandas >= 1.2.0
* pygmt >= 0.5.0 