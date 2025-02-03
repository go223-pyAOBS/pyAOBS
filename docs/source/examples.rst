========
Examples
========

This section provides detailed examples of using pyAOBS for various tasks.

Basic Model Operations
--------------------

Reading and Writing Models
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyAOBS.model_building import ZeltVelocityModel2d
    
    # Read a model
    model = ZeltVelocityModel2d("input_model.in")
    
    # Modify velocities
    model.scale_velocities(1.1)  # Increase all velocities by 10%
    
    # Save the modified model
    model.save("output_model.in")

Velocity Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from pyAOBS.model_building import ZeltVelocityModel2d
    
    model = ZeltVelocityModel2d("velocity.in")
    
    # Create a profile
    x = np.linspace(0, 100, 1000)  # 100 km profile
    z = np.ones_like(x) * 2.0      # At 2 km depth
    
    # Get velocities along the profile
    velocities = model.get_profile(x, z)
    
    # Calculate statistics
    mean_vel = np.mean(velocities)
    std_vel = np.std(velocities)
    print(f"Mean velocity: {mean_vel:.2f} km/s")
    print(f"Standard deviation: {std_vel:.2f} km/s")

Advanced Visualization
-------------------

Custom Plotting
~~~~~~~~~~~~~

.. code-block:: python

    from pyAOBS.visualization import ZeltModelVisualizer
    import matplotlib.pyplot as plt
    
    model = ZeltVelocityModel2d("velocity.in")
    visualizer = ZeltModelVisualizer(model)
    
    # Create a custom figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot velocity model
    visualizer.plot_zeltmodel(ax=ax1)
    ax1.set_title("Velocity Model")
    
    # Plot velocity profile
    x = np.linspace(0, 100, 1000)
    z = np.ones_like(x) * 2.0
    velocities = model.get_profile(x, z)
    ax2.plot(x, velocities)
    ax2.set_title("Velocity Profile at 2 km Depth")
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Velocity (km/s)")
    
    plt.tight_layout()
    plt.savefig("velocity_analysis.png")

Rock Classification Examples
-------------------------

Basic Classification
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyAOBS.utils.isrock import RockClassifier
    
    classifier = RockClassifier()
    
    # Single velocity classification
    vp = 6.5  # km/s
    rock_type = classifier.classify_by_vp(vp)
    print(f"Rock type for Vp={vp} km/s: {rock_type}")
    
    # Batch classification
    velocities = [5.5, 6.0, 6.5, 7.0]
    rock_types = [classifier.classify_by_vp(v) for v in velocities]
    for v, r in zip(velocities, rock_types):
        print(f"Vp={v} km/s -> {r}")

Model Analysis
~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from pyAOBS.model_building import ZeltVelocityModel2d
    from pyAOBS.utils.isrock import RockClassifier
    
    # Load model and classifier
    model = ZeltVelocityModel2d("velocity.in")
    classifier = RockClassifier()
    
    # Create a grid of points
    x = np.linspace(0, 100, 100)
    z = np.linspace(0, 10, 50)
    X, Z = np.meshgrid(x, z)
    
    # Get velocities at all points
    velocities = model.get_velocities(X.flatten(), Z.flatten())
    
    # Classify rocks at all points
    rock_types = [classifier.classify_by_vp(v) for v in velocities]
    
    # Count rock types
    from collections import Counter
    rock_distribution = Counter(rock_types)
    for rock, count in rock_distribution.most_common():
        print(f"{rock}: {count} points") 