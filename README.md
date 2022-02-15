
[![Python package](https://github.com/HornungLab/py-lmd/actions/workflows/python-package.yml/badge.svg?branch=release)](https://github.com/HornungLab/py-lmd/actions/workflows/python-package.yml) [![Python package](https://img.shields.io/badge/version-v1.0.1-blue)](https://github.com/HornungLab/py-lmd/actions/workflows/python-package.yml) [![Python package](https://img.shields.io/badge/license-MIT-blue)](https://github.com/HornungLab/py-lmd/actions/workflows/python-package.yml)


# py-lmd

Read, create and write cutting data for the Leica LMD6 & LMD7 microscope.
Build reproducible workflows to calibrate, import SVG files and convert single-cell segmentation masks.


Installation from Github
========================
py-lmd has been tested with **Python 3.8 and 3.9**.
To install the py-lmd library clone the Github repository and use pip to install the library in your current environment.
It is recommended to use the library with a conda environment. Please make sure that the package is installed editable
like described. Otherwise static glyph files might not be available.

```
git clone https://github.com/HornungLab/py-lmd
pip install -e .
```

If you are installing on an M1 apple silicon Mac you will need to install `numba` via conda instead of pip before proceeding with the installation of the py-lmd library.

```
conda install numba
```
  
Documentation
========================
The current documentation can be found under `docs\_build` as pdf and html.
