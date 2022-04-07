- pip install sphinxcontrib-napoleon
- Include 'nbsphinx', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon' as extensions
- Change directory to phidl/docs/source
- Run "python gen_geometry.py" to build images
- Run "python gen_API.py" to build API rst file and make html
    - First input argument must end with .rst if a new file is being made or
      be "add <.rst file>" if a docstring is being added to an existing file
      or be "build" if no docstrings need to be made but the sphinx docs
      should be built.
    - Every other argument must be a Python file
    - Ex: "python gen_API.py add API.rst file1.py file2.py file3.py"
- Open the index.html in the phidl/docs/build folder
