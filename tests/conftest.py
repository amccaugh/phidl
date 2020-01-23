# make sure phidl is on path
import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

# specify directories for XOR test files
import lytest
test_root = os.path.dirname(__file__)
lytest.utest_buds.test_root = test_root

import phidl.waveguides
phidl.waveguides.minimum_bent_edge_length = 0.1