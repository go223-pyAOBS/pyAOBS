from setuptools import setup, find_packages
import os

# Read README.md for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pyAOBS",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'xarray>=0.16.0',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
        'pandas>=1.2.0',
        'pygmt>=0.5.0'
    ],
    author="Haibo Huang",
    author_email="go223@scsio.ac.cn",
    description="A package for seismic data processing and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/go223-pyAOBS/pyAOBS",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    project_urls={
        "Bug Reports": "https://github.com/go223-pyAOBS/pyAOBS/issues",
        "Source": "https://github.com/go223-pyAOBS/pyAOBS",
    },
) 