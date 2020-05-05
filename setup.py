#!/usr/bin/env python

from setuptools import setup

install_requires=[
   'gdspy>=1.5',
   'numpy',
   'matplotlib',
   'webcolors',
]

setup(name='phidl',
      version='1.3.0',
      description='PHIDL',
      install_requires=install_requires,
      author='Adam McCaughan',
      author_email='amccaugh@gmail.com',
      packages=['phidl'],
      py_modules=['phidl.geometry', 'phidl.routing', 'phidl.utilities'],
      package_dir = {'phidl': 'phidl'},
     )
