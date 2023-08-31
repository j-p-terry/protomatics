#! /usr/bin/env python

import os

import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setuptools.setup(
    name="protomatics",
    version="0.0.1",
    author="Jason Terry",
    author_email="jpterry@uga.edu",
    packages=["protomatics"],
    url="https://github.com/j-p-terry/protomatics",
    license="LICENSE.md",
    description=("Kinematic analysis of protoplanetary disk data."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "astropy",
        "bettermoments",
        "matplotlib",
        "numpy",
        "scipy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    readme="README.md",
    zip_safe=True,
)
