import pytest
import matplotlib
import phidl
import os

def test_tutorial():
    matplotlib.rcParams['figure.max_open_warning'] = 0  # prevents a memory warning

    # Artifact clean up 1: store the initial file snapshot
    initialFiles = []
    for fname in os.listdir('.'):
        initialFiles.append(fname)

    import phidl.phidl_tutorial_example  # importing runs the module code

    # Artifact clean up 2: delete any new files
    for fname in os.listdir('.'):
        if fname not in initialFiles:
            os.remove(fname)
