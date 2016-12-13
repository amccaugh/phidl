#!/usr/bin/env python

from setuptools import setup

install_requires=[
   'gdspy>=1.0',
   'numpy',
   'matplotlib',
   'pyyaml',
   'scikit-image>=0.11',
   'webcolors',
]

setup(name='phidl',
      version='0.5.6',
      description='PHIDL',
      install_requires=install_requires,
      author='Adam McCaughan',
      author_email='adam.mccaughan@nist.gov',
      packages=['phidl'],
      py_modules=['phidl.geometry'],
      package_dir = {'phidl': 'phidl'},
     )