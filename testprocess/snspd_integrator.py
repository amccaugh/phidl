
    
#==============================================================================
# SNSPD Integrator
#==============================================================================

nanowire_width = 0.3
fill_factor = 1/3
contact_pad_width = 2
snspd_size = [10,10]
pad_flag_size = [250,250]
pad_pole_size = [contact_pad_width, 20]



# Make the SNSPD with expansions to match the contact pads
S = Device('SNSPD with expansions')
snspd_device = snspd(wire_width = nanowire_width, wire_pitch = nanowire_width/fill_factor, size = snspd_size, num_pts = 20, terminals_same_side = False)
s = S.add_device(snspd_device)
S.rotate(90)
step_device = optimal_step(start_width = nanowire_width, end_width = contact_pad_width, num_pts = 50, anticrowding_factor = 1.2, width_tol = 1e-3)
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

cpm = D.add_device(compass_multi(size = [800,100], center =[0,-200], ports = {'N':3,'S':1}, layer = 0, datatype = 0) )
s = D.add_device(S)
f = D.add_device(flagpole(flag_size = pad_flag_size, pole_size = pad_pole_size, shape = 'q', taper_type = 'fillet')).translate([0,200])

quickplot(D)

# Route components
D.route(port1 = s.ports[1], port2 = cpm.ports['N1'], path_type = 'sine', width_type = 'sine', width1 = None, width2 = contact_pad_width*10, layer = 0, datatype = 0)
D.route(port1 = f.ports[1], port2 = s.ports[2], path_type = 'sine', width_type = 'sine', width1 = None, width2 = None, layer = 0, datatype = 0)

quickplot(D)


S.write_gds('hellokitty2343214.gds', precision = 1e-11)