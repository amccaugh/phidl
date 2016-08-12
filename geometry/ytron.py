

#%%

def sine_curve_polygon(width_begin, width_end, length, layer = 1, num_pts = 20, mirrored = False):
            
    sine_curve_x = linspace(0, length, num_pts)
    sine_curve_y = (sin(linspace(-pi/2, pi/2, num_pts)) + 1)*(width_end-width_begin)
    sine_curve_y += width_begin
    
    xpts = sine_curve_x.tolist() + [length, 0]
    ypts = sine_curve_y.tolist() + [0, 0]
    
    # Ports
    ports = {
        'begin_midpoint': [0, width_begin/2],
        'end_midpoint': [length, width_end/2],
    }
    
    
    if mirrored: ypts = -ypts
    return gdspy.Polygon(zip(xpts,ypts), layer), ports
    

def ytron_polygon(rho_c = 5, theta_c = 5.0/180*pi, theta_res = 10.0/180*pi, L_arm = 500, L_gnd = 500, W_ch = 200, W_g = 200, layer = 1):
    # theta_c = theta_c*pi/180
    # theta_res = theta_c*pi/180
    thetalist = linspace(-(pi-theta_c),-theta_c, (pi-theta_c)/theta_res)
    semicircle_x = rho_c*cos(thetalist)
    semicircle_y = rho_c*sin(thetalist)+rho_c

    # Rest of yTron
    xc = rho_c*cos(theta_c) 
    yc = rho_c*sin(theta_c) 
    L_arm_x = L_arm*sin(theta_c) 
    L_arm_y = L_arm*cos(theta_c) 

    # Write out x and y coords for yTron polygon
    xpts = semicircle_x.tolist() + [xc+L_arm_x, xc+L_arm_x+W_ch, xc+W_ch,    xc+W_ch, 0, -(xc+W_g), -(xc+W_g), -(xc+L_arm_x+W_g), -(xc+L_arm_x)] 
    ypts = semicircle_y.tolist() + [yc+L_arm_y,      yc+L_arm_y,      yc,   yc-L_gnd, yc-L_gnd,  yc-L_gnd,        yc,        yc+L_arm_y,    yc+L_arm_y] 
    
    # Ports
    ports = {
        'left_arm_midpoint': [-(xc+L_arm_x+W_g/2), yc+L_arm_y],
        'right_arm_midpoint': [xc+L_arm_x+W_ch/2, yc+L_arm_y],
        'source_midpoint': [0, -L_gnd],
    }

    return gdspy.Polygon(zip(xpts,ypts), layer), ports


def connect_ports(poly_to_move, point1, point2, rotation):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    poly_to_move.translate(dx, dy)
    poly_to_move.rotate(angle = rotation, center = point1)
    
    


rho_c = 10
theta_c = 5*pi/180.0
theta_res = 10*pi/180.0
L_arm = 100
L_gnd = 100
W_ch = 100
W_g = 100

ytron_poly, ytron_ports = ytron_polygon(rho_c, theta_c, theta_res, L_arm, L_gnd, W_ch, W_g)
sine_poly, sine_ports = sine_curve_polygon(width_begin = L_arm, width_end = L_arm*2, length = L_arm*4, mirrored = False, num_pts = 20)

#sine_poly.translate(dx = left_port[0], dy = left_port[1])
connect_ports(sine_poly,   ytron_ports['left_arm_midpoint'], sine_ports['begin_midpoint'],rotation = np.pi/2)

quickplot_polygon(ytron_poly)
quickplot_polygon(sine_poly)

quickplot_ports(ytron_ports)
quickplot_ports(sine_ports)
plt.axis('square')


xpts, ypts, ports = ytron_points(rho_c, theta_c, theta_res, L_arm, L_gnd, W_ch, W_g)
plot(xpts,ypts, '.-')
axis('square')
