''' This runs before every test, even if they happen in parallel.
    Use it for setting up a consistent state of global variables or setting up directories
'''
# make sure phidl is on path
import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
