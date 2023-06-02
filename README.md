
[![Python package](https://github.com/MannLabs/py-lmd/actions/workflows/python-package.yml/badge.svg?branch=release)](https://github.com/MannLabs/py-lmd/actions/workflows/python-package.yml) [![Python package](https://img.shields.io/badge/version-v1.0.0-blue)](https://github.com/MannLabs/py-lmd/actions/workflows/python-package.yml) [![Python package](https://img.shields.io/badge/license-MIT-blue)](https://github.com/MannLabs/py-lmd/actions/workflows/python-package.yml)
[![website](https://img.shields.io/website?url=https%3A%2F%2Fmannlabs.github.io/py-lmd/html/index.html)](https://mannlabs.github.io/py-lmd/html/index.html)

![py-lmd_border](https://github.com/MannLabs/py-lmd/assets/15019107/035dfa13-c1a3-4ed0-8bbe-ddd5df29c367)

Read, create and write cutting data for the Leica LMD6 & LMD7 microscope.
Build reproducible workflows to calibrate, import SVG files and convert single-cell segmentation masks.


Installation from Github
========================
py-lmd has been tested with **Python 3.8 and 3.9**.
To install the py-lmd library clone the Github repository and use pip to install the library in your current environment.
It is recommended to use the library with a conda environment. Please make sure that the package is installed editable
like described. Otherwise static glyph files might not be available. 

We recommend installing the non-python dependencies with conda before installing py-lmd:

```
git clone https://github.com/MannLabs/py-lmd

conda create -n "py-lmd-env"
conda activate py-lmd-env
conda install python=3.9 scipy scikit-image>=0.19 numpy numba -c conda-forge
pip install -e .

```

If you are installing on an M1 apple silicon Mac you will need to install `numba` via conda instead of pip before proceeding with the installation of the py-lmd library.

```
conda install numba
```
  
Documentation
========================
The current documentation can be found under https://mannlabs.github.io/py-lmd/html/index.html.

Citing our Work
=================

py-lmd was developed by Georg Wallmann, Sophia Mädler and Niklas Schmacke in the labs of Veit Hornung and Matthias Mann. If you use our code please cite the [following manuscript](https://www.biorxiv.org/content/10.1101/2023.06.01.542416v1):

SPARCS, a platform for genome-scale CRISPR screening for spatial cellular phenotypes
Niklas Arndt Schmacke, Sophia Clara Maedler, Georg Wallmann, Andreas Metousis, Marleen Berouti, Hartmann Harz, Heinrich Leonhardt, Matthias Mann, Veit Hornung
bioRxiv 2023.06.01.542416; doi: https://doi.org/10.1101/2023.06.01.542416


