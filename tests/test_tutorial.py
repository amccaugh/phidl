import pytest
import matplotlib
import phidl

def test_tutorial():
    matplotlib.rcParams['figure.max_open_warning'] = 0  # prevents a memory warning
    import phidl.phidl_tutorial_example  # importing runs the module code
