import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py-lmd",
    version="0.1.3",
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
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

