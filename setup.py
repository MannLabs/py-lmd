import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
# get path for requirements file
parent_folder = os.path.dirname(os.path.realpath(__file__))
requirementPath = os.path.join(parent_folder, 'requirements.txt')

# load requirements
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="py-lmd",
    version="1.0.2",
    author="Georg Wallmann",
    author_email="g.wallmann@campus.lmu.de",
    description="Read, Modify and Create new shape files for the Leica LMD6 & LMD7",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeorgWa/py-lmd",
    project_urls={
        "Documentation":"https://py-lmd.readthedocs.io/en/latest/",
        "Bug Tracker": "https://github.com/GeorgWa/py-lmd/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)