from setuptools import find_packages, setup
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Packages that must never appear in an sdist/wheel (venvs, research dumps, test data).
_DROP_EXACT = {
    "pyAOBS.tests",
    "pyAOBS.output",
    "pyAOBS.petrology.article",
    "pyAOBS.petrology.validation",
    "pyAOBS.petrology._tmp_magmars",
    "pyAOBS.petrology.Perple_X_v7.2.5_Linux_64_gfortran",
    "pyAOBS.petrology.ScienceDirect_files_02Jul2026_03-22-55.744",
    "pyAOBS.utils.rockphypy-main",
    "pyAOBS.field.tests",
    "pyAOBS.visualization.zplotpy.test",
}


def _iter_packages():
    for name in find_packages(include=["pyAOBS", "pyAOBS.*"]):
        if ".venv" in name or "site-packages" in name:
            continue
        if any(name == drop or name.startswith(drop + ".") for drop in _DROP_EXACT):
            continue
        # Chinese-named duplicate tree / Cursor metadata
        if "副本" in name or name.startswith("pyAOBS..cursor"):
            continue
        yield name


setup(
    name="pyAOBS",
    version="3.0.0rc2",
    packages=list(_iter_packages()),
    install_requires=[
        # NumPy 2.0.x：imodel/zplot/workbench 主路径已验证；zplot f2py 内核会在版本变化时自动重编
        "numpy>=1.24,<2.1",
        "xarray>=2023.1.0",
        "scipy>=1.13.0",
        "matplotlib>=3.8.0",
        "dearpygui>=1.11.1",
        "pandas>=2.0.0",
        "scikit-learn>=1.5.0",  # imodel 岩性分类（utils/rocks.py）
        "seaborn>=0.13.0",  # utils/isrock.py 硬依赖
        "openpyxl>=3.0.0",  # 用于读取Excel文件（rocks.xlsx）
    ],
    extras_require={
        "gmt": ["pygmt>=0.5.0"],
        "gui-qt": [
            "PySide6>=6.4.0",
            "matplotlib>=3.8.0",
        ],
        "petrology": [
            # Prefer pip burnman when available; in-tree vendored copy remains a fallback
            "burnman>=1.0.0",
        ],
        "full": [
            "pygmt>=0.5.0",
            "pyproj>=3.0.0",
            "obspy>=1.2.0",
            "PySide6>=6.4.0",
            "burnman>=1.0.0",
        ],
    },
    author="Haibo Huang",
    author_email="go223@scsio.ac.cn",
    description=(
        "Active-source ocean-bottom seismology toolkit: Zelt models, "
        "workbench GUIs, and KKHS02 petrology workflows"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/go223-pyAOBS/pyAOBS",
    keywords="seismology, ocean bottom, velocity model, tomography, visualization, petrology",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    project_urls={
        "Bug Reports": "https://github.com/go223-pyAOBS/pyAOBS/issues",
        "Source": "https://github.com/go223-pyAOBS/pyAOBS",
        "Changelog": "https://github.com/go223-pyAOBS/pyAOBS/blob/main/CHANGELOG.md",
        "Documentation": "https://github.com/go223-pyAOBS/pyAOBS",
    },
    include_package_data=True,
    package_data={
        "pyAOBS": [
            "utils/*.xlsx",
            "utils/*.csv",
        ],
        "pyAOBS.petrology": [
            "data/**/*.json",
            "data/**/*.csv",
            "data/**/*.yaml",
            "data/**/*.yml",
            "data/**/*.txt",
            "figures/*.png",
            "reference/*.yaml",
        ],
    },
)
