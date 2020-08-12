#!/usr/bin/env python

from setuptools import setup

install_requires=[
   'gdspy>=1.5',
   'numpy',
   'matplotlib',
]

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='phidl',
      version='1.3.0',
      description='PHIDL',
      long_description = long_description,
      long_description_content_type='text/markdown',
      install_requires=install_requires,
      author='Adam McCaughan',
      author_email='amccaugh@gmail.com',
      packages=['phidl'],
      py_modules=['phidl.geometry', 'phidl.routing', 'phidl.utilities'],
      package_dir = {'phidl': 'phidl'},
     )
