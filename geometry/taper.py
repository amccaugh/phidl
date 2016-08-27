import numpy as np
import gdspy
from numpy import sqrt, pi

def taper(length, width = None, end_width = None, port = None, layer = 0, datatype = 0):
    if type(port) is Port and width is None: width = port.width
    if end_width is None: end_width = width
    xpts = [0, length, length, 0]
    ypts = [width/2, end_width/2, -end_width/2, -width/2]
    
    d = Device('taper')
    d.add_polygon([xpts,ypts], layer = layer, datatype = datatype)
    d.add_port(name = 1, midpoint = [0, 0], width = width, orientation = 180)
    d.add_port(name = 2, midpoint = [length, 0], width = end_width, orientation = 0)
    if type(port) is Port: 
        d.rotate(angle = port.orientation, center = [0,0])
        d.move(origin = [0,0], destination = port.midpoint)
    return d
    

quickplot(d)


    
def ramp(length, width, end_width = None, layer = 0, datatype = 0):
    if end_width is None: end_width = width
    xpts = [0, width/2, width/2, 0]
    ypts = [width/2, end_width/2, 0, 0]
    d = Device('ramp')
    d.add_polygon([xpts,ypts], layer = layer, datatype = datatype)
    d.add_port(name = 1, midpoint = [0, 0], width = width, orientation = 180)
    d.add_port(name = 2, midpoint = [length, end_width/2], width = end_width, orientation = 0)
    return d
    

def racetrack_gradual(t, R, N, layer = 0, datatype = 0):
    curve_fun = lambda t: racetrack_gradual_parametric(t, R = 5, N = 3)
    route_path = gdspy.Path(width = 0.3, initial_point = [0,0])
    route_path.parametric(curve_fun, number_of_evaluations=99,\
            max_points=199,  final_distance=None, layer=layer, datatype=datatype)
    d = Device()
    d.add(route_path)
    return d
    

def _racetrack_gradual_parametric(t, R, N):
    """ Takes in a parametric value ``t`` on (0,1), returns the x,y coordinates
    of a racetrack bent according to 20090810_EOS4_modulator_designs_excerptForJasonGradualBends.ppt """
    x0 = R/2**(1/N)
    Rmin = 2**(0.5-1/N)/(N-1)*R
    R0 = R-(x0-Rmin/sqrt(2))
    t = np.array(t)
    x,y = np.zeros(t.shape), np.zeros(t.shape)
    
    # Doing the math
    x = np.cos(t*np.pi/2)*R0 # t (0-1) while x (0 to R0)
    ii =  (Rmin/sqrt(2) < x) & (x <= R0)
    jj =  (0 < x) & (x <= Rmin/sqrt(2))
    y[ii] = (R**N - (x[ii]+(x0-Rmin/sqrt(2)))**N)**(1/N)
    y[jj] = (x0-Rmin/sqrt(2))+sqrt(Rmin**2-x[jj]**2)
    return x,y
    
    
    
#==============================================================================
# Example code
#==============================================================================

#d = racetrack_gradual(t, R = 5, N=3)
#quickplot(d)



#t = np.linspace(0,1)
#x,y = _racetrack_gradual_parametric(t, R = 5, N = 3)
#plt.plot(x,y)
#plt.axis('equal')