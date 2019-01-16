#!/usr/bin/env python

from setuptools import setup

install_requires=[
   'gdspy>=1.3.1',
   'numpy==1.15.4',  # note: 1.16.0 exists, but there is a name change that breaks scikit-image
   'matplotlib',
   'pyyaml',
   'scikit-image>=0.11',
   'webcolors',
]

setup(name='phidl',
      version='0.9.1',
      description='PHIDL',
      install_requires=install_requires,
      author='Adam McCaughan',
      author_email='amccaugh@gmail.com',
      packages=['phidl'],
      py_modules=['phidl.geometry', 'phidl.routing', 'phidl.utilities'],
      package_dir = {'phidl': 'phidl'},
     )
