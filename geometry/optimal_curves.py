import numpy as np
import gdspy
import scipy.optimize


def optimal_hairpin(width = 100.0, pitch = 200.0, length = 3000, num_pts = 50):

    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    a = (pitch + width)/2
    y = -(pitch - width)/2
    x = -pitch
    dl = width/(num_pts*2)
    n = 0

    # Get points of ideal curve from conformal mapping
    # TODO This is an inefficient way of finding points that you need
    xpts = [x]; ypts = [y]
    while (y < 0) & (n<1e6):
        s = x + 1j*y
        w = np.sqrt(1 - np.exp(np.pi*s/a))
        wx = np.real(w); wy = np.imag(w)
        wx = wx/np.sqrt(wx**2+wy**2); wy = wy/np.sqrt(wx**2+wy**2)
        x = x + wx*dl; y = y + wy*dl
        xpts.append(x); ypts.append(y)
        n = n+1
    ypts[-1] = 0 # Set last point be on the x=0 axis for sake of cleanliness
    ds_factor = int(len(xpts)/num_pts) # Downsample the total number of points
    xpts = xpts[::-ds_factor]; xpts = xpts[::-1]    # This looks confusing, but it's just flipping the arrays around
    ypts = ypts[::-ds_factor]; ypts = ypts[::-1]    # so the last point is guaranteed to be included when downsampled

    # Add points for the rest of meander
    xpts.append(xpts[-1] + 4*width); ypts.append(0)
    xpts.append(xpts[-1]); ypts.append(-a)
    xpts.append(xpts[0]); ypts.append(-a)
    xpts.append(xpts[0]-length); ypts.append(-a)
    xpts.append(xpts[-1]); ypts.append(ypts[0])
    xpts.append(xpts[0]); ypts.append(ypts[0])
    
    xpts = np.array(xpts)
    ypts = np.array(ypts)

    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    d = Device(name = 'hairpin')
    d.add_polygon(gdspy.Polygon(zip(xpts,ypts)))
    d.add_polygon(gdspy.Polygon(zip(xpts,-ypts)))
    
    xports = min(xpts)
    yports = min(ypts) + width/2
    d.add_port(name = 1, midpoint = [xports,-yports], width = width, orientation = 90)
    d.add_port(name = 2, midpoint = [xports,yports], width = width, orientation = -90)
    
    return d
    
    

    
    
def optimal_step(start_width = 10, end_width = 22, num_pts = 50, width_tol = 1e-3):

    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    def step_points(eta, W, a):
        # Returns points from a unit semicircle in the w (= u + iv) plane to 
        # the optimal curve in the zeta (= x + iy) plane which transitions
        # a wire from a width of 'W' to a width of 'a'
        # eta takes value 0 to pi

        W = np.complex(W)
        a = np.complex(a)

        gamma = (a*a + W*W)/(a*a - W*W)

        w = np.exp(1j*eta)

        zeta = 4*1j/np.pi*(W*np.arctan(np.sqrt((w-gamma)/(gamma+1))) \
                           + a*np.arctan(np.sqrt((gamma-1)/(w-gamma))))

        x = np.real(zeta)
        y = np.imag(zeta)
        return x,y


    def invert_step_point(x_desired = -10, y_desired = None, W = 1, a = 2):
        # Finds the eta associated with the value x_desired along the optimal curve
        def fh(eta):
            guessed_x, guessed_y = step_points(eta, W = W, a = a)
            if y_desired is None:   return (guessed_x-x_desired)**2 # The error
            else:                   return (guessed_y-y_desired)**2

        found_eta = scipy.optimize.fminbound(fh, x1 = 0, x2 = np.pi, args=())
        return step_points(found_eta, W = W, a = a)

    if start_width > end_width:
        reverse = True
        start_width, end_width = end_width, start_width
        
    xmin,ymin = invert_step_point(y_desired = start_width*(1+width_tol), W = start_width, a = end_width)
    xmax,ymax = invert_step_point(y_desired = end_width*(1-width_tol), W = start_width, a = end_width)
    
    xpts = np.linspace(xmin, xmax, num_pts).tolist()
    ypts = []
    for x in xpts:
        x,y = invert_step_point(x_desired = x, W = start_width, a = end_width)
        ypts.append(y)
    
    ypts[-1] = end_width
    ypts[0] =  start_width
    xpts.append(xpts[-1])
    ypts.append(0)
    xpts.append(xpts[0])
    ypts.append(0)

    if reverse is True:
        xpts = (-np.array(xpts)).tolist()
        start_width, end_width = end_width, start_width

    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    d = Device(name = 'step')
    d.add_polygon(gdspy.Polygon(zip(xpts,ypts)))
    
    d.add_port(name = 1, midpoint = [min(xpts),start_width/2], width = start_width, orientation = 180)
    d.add_port(name = 2, midpoint = [max(xpts),end_width/2], width = end_width, orientation = 0)
    
    return d
    
    
    
    
    
#==============================================================================
# Example code
#==============================================================================
    
#hairpin = optimal_hairpin(width = 100.0, pitch = 200.0, length = 3000, num_pts = 50)
#quickplot(hairpin)


#step = optimal_step(start_width = 10, end_width = 1, num_pts = 30, width_tol = 1e-3)
#quickplot(step)