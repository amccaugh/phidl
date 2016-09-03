import numpy as np
import gdspy
from numpy import sqrt, pi



def racetrack_gradual(width = 0.3, R = 5, N = 3, layer = 0, datatype = 0):
    curve_fun = lambda t: racetrack_gradual_parametric(t, R = 5, N = 3)
    route_path = gdspy.Path(width = width, initial_point = [0,0])
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

d = racetrack_gradual(width = 0.3, R = 5, N = 3)
quickplot(d)



t = np.linspace(0,1)
x,y = _racetrack_gradual_parametric(t, R = 5, N = 3)
plt.plot(x,y)
plt.axis('equal')