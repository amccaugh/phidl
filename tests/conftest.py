# make sure phidl is on path
import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

# make sure geometry tests go to the right place
import lytest
test_root = os.path.dirname(__file__)
lytest.utest_buds.test_root = test_root

# ensure that this is repeatable. We don't need high resolution for testing
import phidl.waveguides
phidl.waveguides.minimum_bent_edge_length = 0.1