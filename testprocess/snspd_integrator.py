
    
#==============================================================================
# SNSPD Integrator
#==============================================================================

nanowire_width = 0.3
fill_factor = 1/3
contact_pad_width = 2
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



# Make the SNSPD with expansions to match the contact pads
S = Device('SNSPD with expansions')
snspd_device = snspd(wire_width = nanowire_width, wire_pitch = nanowire_width/fill_factor, size = snspd_size, num_pts = 20, terminals_same_side = False)
s = S.add_device(snspd_device)
S.rotate(90)
step_device = optimal_step(start_width = nanowire_width, end_width = contact_pad_width, num_pts = 50, anticrowding_factor = 2, width_tol = 1e-3)
step1 = S.add_device(step_device)
step2 = S.add_device(step_device)
S.connect(step1.ports[1], destination = s.ports[1])
S.connect(step2.ports[1], destination = s.ports[2])
S.add_port(name = 1, port = step1.ports[2])
S.add_port(name = 2, port = step2.ports[2])

quickplot(S)

# Add contact pads to for individual SNSPDs on north side
#S.connect(port = f.ports[1], destination = t_north.ports[2])

# Create yTron


# Add components to a blank device
D = Device('SNSPD-yTron-integator')

cpm = D.add_device(compass_multi(size = connector_size, center = [0,-200], ports = {'N':num_devices,'S':1}, layer = 0, datatype = 0) )
f = D.add_device(flagpole(flag_size = connector_size, pole_size = [width_left,width_left], shape = 'p', taper_type = 'fillet', layer = 0, datatype = 0))
T = tee(top_size = pad_flag_size, leg_size = pad_pole_size, taper_type = 'fillet', layer = 0, datatype = 0)
pad_array = D.add_array(T, start = [0,1000], direction = 'E', spacing = F.width+50, num_devices = num_devices)

snspd_array = D.add_array(S, start = [-100,300], direction = 'E', spacing = S.width+20, num_devices = num_devices)
y =  D.add_device(ytron_round(rho_intersection, theta_intersection, arm_length, source_length, \
                  width_right, width_left, theta_resolution = 10, layer = 0, datatype = 0))
D.connect(port = f.ports[2], destination = cpm.ports['S1'])
D.connect(port = y.ports['right'], destination = f.ports[1])

gnd = D.add_device(compass(size = pad_flag_size, layer = 0, datatype = 0))
gnd.move(origin = gnd.center, destination = y.ports['source'])
gnd.move([0,-1000])

quickplot(D)

# Route components
for n in range(num_devices):
    print(n)
    s = snspd_array[n]; p = pad_array[n]
    D.route(port1 = s.ports[1], port2 = cpm.ports['N%s' % (n+1)], path_type = 'sine', width_type = 'sine', width1 = None, width2 = contact_pad_width*10, num_path_pts = 99, layer = 0, datatype = 0)
    D.route(port1 = p.ports['S'], port2 = s.ports[2], path_type = 'sine', width_type = 'sine', width1 = None, width2 = None, num_path_pts = 99, layer = 0, datatype = 0)
D.route(y.ports['source'], gnd.ports['N'], path_type = 'sine', width_type = 'sine')
quickplot(D)


D.write_gds('SNSPD Integrator.gds', precision = 1e-10)