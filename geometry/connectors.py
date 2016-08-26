import numpy as np
import gdspy

def eastwest(center = [0,0], width = 1):
    d = Device(name = 'twoportEW')
    d.add_port(name = 'E', midpoint = [dx/2, 0],  width = width, orientation = 0)
    d.add_port(name = 'W', midpoint = [-dx/2, 0], width = width, orientation = 180)
    return d



def northsouth(center = [0,0], width = 1):
    d = Device(name = 'twoportNS')
    d.add_port(name = 'N', midpoint = [0, dy/2],  width = width, orientation = 90)
    d.add_port(name = 'S', midpoint = [0, -dy/2], width = width, orientation = -90)
    return d