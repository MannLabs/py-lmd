.. py-lmd documentation master file, created by
   sphinx-quickstart on Fri Mar  5 15:14:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


*******************
py-lmd
*******************

The py-lmd package provides an opensource python interface to generate cutting directions for the Leica LMD6 & LMD7 laser microdissection microscopes. These cutting directions can be used to perform laser microdissection on arbitrary sample types.

py-lmd interfaces directly with your existing python workflows to make your samples accessible via LMD. For example you could use the output of your ML-based region detection on clinical tissue samples to export cutting shapes to excise these same regions on the LMD.

In addition, you can build reproducible workflows to calibrate slides, import SVG files and convert them to cutting masks, and can convert existing segmentation masks to cutting XMLs.

Installation
============

py-lmd has been tested with Python 3.9, 3.10 and 3.11. To install the py-lmd library clone the Github repository and use pip to install the library in your current environment. 
It is recommended to use the library with a conda environment. 
Please make sure that the package is installed editable like described. 
Otherwise static glyph files might not be available.

We recommend installing the non-python dependencies with conda before installing py-lmd:

.. code::

   git clone https://github.com/MannLabs/py-lmd

   conda create -n "py-lmd-env"
   conda activate py-lmd-env
   conda install python=3.9 scipy scikit-image>=0.19 numpy numba -c conda-forge
   pip install -e .

If you are installing on an M1 apple silicon Mac you will need to install numba via conda instead of pip before 
proceeding with the installation of the py-lmd library.

.. code::

   conda install numba

Citing our Work
===============

py-lmd was developed by Georg Wallmann, Sophia Mädler and Niklas Schmacke in the labs of Veit Hornung and
Matthias Mann. If you use our code please cite the `following manuscript <https://doi.org/10.1101/2023.06.01.542416>`_:

SPARCS, a platform for genome-scale CRISPR screening for spatial cellular phenotypes
Niklas Arndt Schmacke, Sophia Clara Maedler, Georg Wallmann, Andreas Metousis, Marleen Berouti, Hartmann Harz, Heinrich Leonhardt, Matthias Mann, Veit Hornung
bioRxiv 2023.06.01.542416; doi: https://doi.org/10.1101/2023.06.01.542416

.. toctree::
   :maxdepth: 5
   :caption: Contents:
   :numbered:
   
   pages/quickstart
   pages/segmentation_loader
   pages/example_notebooks
   pages/modules