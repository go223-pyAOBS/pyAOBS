==========
Quickstart
==========

This guide will help you get started with pyAOBS quickly.

Basic Usage
----------

Reading and Processing Velocity Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyAOBS.model_building import ZeltVelocityModel2d

    # Load a velocity model
    model = ZeltVelocityModel2d("velocity.in")
    
    # Get velocity at a specific point
    velocity = model.at(100.0, 1.5)  # x=100.0 km, z=1.5 km
    
    # Get velocities along a profile
    x_profile = np.linspace(0, 100, 1000)
    z_profile = np.ones_like(x_profile) * 2.0
    velocities = model.get_profile(x_profile, z_profile)

Visualization
~~~~~~~~~~~~

.. code-block:: python

    from pyAOBS.visualization import ZeltModelVisualizer

    # Create a visualizer
    visualizer = ZeltModelVisualizer(model)
    
    # Plot the model
    visualizer.plot_zeltmodel(
        output_file="velocity_model.png",
        title="Velocity Model",
        colorbar_label="Velocity (km/s)"
    )

Rock Classification
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyAOBS.utils.isrock import RockClassifier

    # Create a classifier
    classifier = RockClassifier()
    
    # Classify a rock based on P-wave velocity
    rock_type = classifier.classify_by_vp(6.5)  # Vp in km/s
    print(f"Rock type: {rock_type}")

Advanced Features
---------------

Enhanced Model Processing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyAOBS.model_building import EnhancedZeltModel

    # Create an enhanced model
    enhanced_model = EnhancedZeltModel("velocity.in")
    
    # Compute average velocities
    avg_velocities = enhanced_model.compute_average_velocities()
    
    # Export to different formats
    enhanced_model.export_to_xyz("model.xyz")

For more detailed examples and advanced usage, please refer to the :doc:`examples` section. 