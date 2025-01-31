# pyAOBS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

A Python package for seismic data processing and visualization, with a focus on ZELT format velocity models.

## Features

- Read and write ZELT format velocity models (v.in files)
- Process and manipulate velocity models
- Interpolate velocities at arbitrary points
- Average layer velocities
- Export to various formats
- Visualization tools using PyGMT and Matplotlib
- Support for time and depth domain data

## Installation

### From PyPI (Recommended)
```bash
pip install pyAOBS
```

### From Source
```bash
git clone https://github.com/go223-pyAOBS/pyAOBS.git
cd pyAOBS
pip install -e .
```

## Dependencies

- numpy >= 1.20.0
- xarray >= 0.16.0
- scipy >= 1.6.0
- matplotlib >= 3.3.0
- pandas >= 1.2.0
- pygmt >= 0.5.0

## Usage

```python
from pyAOBS.model_building import ZeltVelocityModel2d, EnhancedZeltModel

# Basic usage
model = ZeltVelocityModel2d("velocity.in")
velocity = model.at(100.0, 1.5)  # Get velocity at point (100.0, 1.5)

# Enhanced features
enhanced_model = EnhancedZeltModel("velocity.in")
avg_velocities = enhanced_model.compute_average_velocities()

# Visualization
from pyAOBS.visualization import ZeltModelVisualizer

visualizer = ZeltModelVisualizer(model)
visualizer.plot_zeltmodel(
    output_file="velocity_model.png",
    title="Velocity Model",
    colorbar_label="Velocity (km/s)"
)
```

## Documentation

For detailed documentation and examples, please visit our [documentation page](https://go223-pyAOBS.github.io/pyAOBS).

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Haibo Huang (go223@scsio.ac.cn)

## Citation

If you use pyAOBS in your research, please cite:

```bibtex
@software{pyAOBS2024,
  author = {Haibo Huang},
  title = {pyAOBS: A Python Package for Seismic Data Processing and Visualization},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/go223-pyAOBS/pyAOBS}
}
``` 