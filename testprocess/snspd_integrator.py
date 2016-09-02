
    
#==============================================================================
# SNSPD Integrator
#==============================================================================

nanowire_width = 0.3
fill_factor = 1/3
contact_pad_width = 5
snspd_size = [20,20]
pad_flag_size = [250,250]
pad_pole_size = [contact_pad_width*10, 50]
connector_size = [250,100]
num_devices = 10

# yTron parameters
rho_intersection = 1
theta_intersection = 5
arm_length = 50
source_length = 50
width_right = 20
width_left = 20


# Add contact pads to for individual SNSPDs on north side
#S.connect(port = f.ports[1], destination = t_north.ports[2])

# Create yTron


# Add components to a blank device
D = Device('SNSPD-yTron-integator')

cpm = D.add_device(compass_multi, size = connector_size, center = [0,-200], ports = {'N':num_devices,'S':1}, layer = 0, datatype = 0)
f = D.add_device(flagpole(flag_size = connector_size, pole_size = [width_left,width_left], shape = 'p', taper_type = 'fillet', layer = 0, datatype = 0))

pad_array = D.add_array(tee(top_size = pad_flag_size, leg_size = pad_pole_size, taper_type = 'fillet', layer = 0, datatype = 0),
                        start = [0,1000], spacing = [f.width+50, 0], num_devices = num_devices)
                        
snspd_array = D.add_array(snspd_expanded(wire_width = nanowire_width, wire_pitch = nanowire_width*3, size = snspd_size, connector_width = contact_pad_width,
                                         terminals_same_side = False, layer = 0, datatype = 0),
                                         start = [-100,300], spacing = [S.width+20, 0], num_devices = num_devices)
           
y =  D.add_device(ytron_round(rho_intersection, theta_intersection, arm_length, source_length,
                  width_right, width_left, theta_resolution = 10, layer = 0, datatype = 0))
D.connect(port = f.ports[2], destination = cpm.ports['S1'])
D.connect(port = y.ports['right'], destination = f.ports[1])

gnd = D.add_device(compass(size = pad_flag_size, layer = 0, datatype = 0))
gnd.move(origin = gnd.center, destination = y.ports['source'])
gnd.move([0,-1000])

quickplot(D)

# Route components
for n in range(num_devices):
    s = snspd_array[n]; p = pad_array[n]
    D.route(port1 = s.ports[1], port2 = cpm.ports['N%s' % (n+1)], path_type = 'sine', width_type = 'sine', width1 = None, width2 = contact_pad_width*10, num_path_pts = 99, layer = 0, datatype = 0)
    D.route(port1 = p.ports['S'], port2 = s.ports[2], path_type = 'sine', width_type = 'sine', width1 = None, width2 = None, num_path_pts = 99, layer = 0, datatype = 0)
D.route(y.ports['source'], gnd.ports['N'], path_type = 'sine', width_type = 'sine')
quickplot(D)


D.write_gds('SNSPD Integrator.gds', precision = 1e-10)



d.add(ytron, config = 'myytron.yaml', width)