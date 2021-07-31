# py-lmd

The py-lmd library allows to read, modify and write shape files for the Leica LMD6 & LMD7 microscope.
Additionally, functionality for the calibration, generation of final pulses and import of vektor files is included.

Installation from pypi
======================
If you would like to install the py-lmd for use in your project, you can directly install it from the python package index.
```
pip install py-lmd
```

Installation from Github
========================
You can also install py-lmd directly from Github for development purposes.
```
git clone https://github.com/GeorgWa/py-lmd.git
pip install -e py-lmd
```
  
Release a new Version
========================
Make sure the version number on the main branch is up to date.

```
py-lmd/setup.py
py-lmd/docs/conf.py
```

Create a new build.
```
python3 -m build
```

Upload it to PyPi:
```
python3 -m twine upload dist/*
```
