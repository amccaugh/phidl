#!/usr/bin/env python

from setuptools import setup

setup(name='phidl',
      version='0.2.2',
      description='Python Distribution Utilities',
      author='Adam McCaughan',
      author_email='adam.mccaughan@nist.gov',
      packages=['phidl'],
      py_modules=['geometry'],
      package_dir = {'phidl': 'phidl'},
     )