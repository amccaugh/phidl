#!/usr/bin/env python

from os import path

from setuptools import setup

install_requires = [
    "gdspy>=1.5",
    "numpy",
    "matplotlib",
]

extras_require = {}
extras_require["all"] = ["freetype-py", "klayout", "rectpack"]
extras_require["test"] = extras_require["all"] + ["pytest"]

# read the contents of your README file

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="phidl",
    version="1.7.1",
    description="PHIDL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require,
    author="Adam McCaughan",
    author_email="amccaugh@gmail.com",
    packages=["phidl"],
    py_modules=["phidl.geometry", "phidl.routing", "phidl.utilities", "phidl.path"],
    package_dir={"phidl": "phidl"},
)
