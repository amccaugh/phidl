import numpy as np
import gdspy
from numpy import sqrt, pi

#def sine_curve_polygon(width_begin, width_end, length, layer = 1, num_pts = 20, mirrored = False):
#            
#    sine_curve_x = linspace(0, length, num_pts)
#    sine_curve_y = (sin(linspace(-pi/2, pi/2, num_pts)) + 1)*(width_end-width_begin)
#    sine_curve_y += width_begin
#    
#    xpts = sine_curve_x.tolist() + [length, 0]
#    ypts = sine_curve_y.tolist() + [0, 0]
#    
#    # Ports
#    ports = {
#        'begin_midpoint': [0, width_begin/2],
#        'end_midpoint': [length, width_end/2],
#    }
#    
#    
#    if mirrored: ypts = -ypts
#    return gdspy.Polygon(zip(xpts,ypts), layer), ports


def ytron_round(rho_intersection = 1, theta_intersection = 5, arm_length = 500, source_length = 500, \
                  width_right = 200, width_left = 200, theta_resolution = 10, layer = 0, datatype = 0):
    
    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    rho = rho_intersection
    theta = theta_intersection*pi/180
    theta_resolution = theta_resolution*pi/180
    thetalist = linspace(-(pi-theta),-theta, int((pi-2*theta)/theta_resolution) + 2)
    semicircle_x = rho_intersection*cos(thetalist)
    semicircle_y = rho_intersection*sin(thetalist)+rho_intersection

    # Rest of yTron
    xc = rho_intersection*cos(theta) 
    yc = rho_intersection*sin(theta) 
    arm_x = arm_length*sin(theta) 
    arm_y = arm_length*cos(theta) 

    # Write out x and y coords for yTron polygon
    xpts = semicircle_x.tolist() + [xc+arm_x, xc+arm_x+width_right, xc+width_right, \
           xc+width_right, 0, -(xc+width_left), -(xc+width_left), -(xc+arm_x+width_left), -(xc+arm_x)] 
    ypts = semicircle_y.tolist() + [yc+arm_y,      yc+arm_y,      yc,   yc-source_length, yc-source_length,  \
            yc-source_length,        yc,        yc+arm_y,    yc+arm_y] 
    
    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    d = Device(name = 'ytron')
    d.add_polygon([xpts,ypts], layer = layer, datatype = datatype)
    d.add_port(name = 'left', midpoint = [-(xc+arm_x+width_left/2), yc+arm_y],  width = width_left, orientation = 90)
    d.add_port(name = 'right', midpoint = [xc+arm_x+width_right/2, yc+arm_y],  width = width_right, orientation = 90)
    d.add_port(name = 'source', midpoint = [0+(width_right-width_left)/2, -source_length+yc],  width = width_left + width_right + 2*xc, orientation = -90)
    
    #==========================================================================
    #  Record any parameters you may want to access later
    #==========================================================================
    d.parameters['radius_curvature'] = rho_intersection
    d.parameters['left_arm_width'] = width_left
    d.parameters['right_arm_width'] = width_right

    return d
    
    
    
#==============================================================================
# Example code
#==============================================================================

#y = ytron_round(rho_intersection = 5, theta_intersection = 5, theta_resolution = 10, arm_length = 500, \
#                source_length = 500, width_right = 200, width_left = 200, layer = 0, datatype = 0)
#quickplot(y)
