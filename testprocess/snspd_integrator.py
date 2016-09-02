
    
#==============================================================================
# SNSPD yTron-based integrator
#==============================================================================
def snspd_integrator(
                    nanowire_width = 0.3,
                    fill_factor = 1/3,
                    contact_pad_width = 5,
                    snspd_size = [20,20],
                    pad_flag_size = [250,250],
                    pad_pole_size = [contact_pad_width*10, 50],
                    pad_spacing = 50,
                    connector_size = [400,150],
                    inset_distance = 2,
                    num_devices = 14,
                    label = 'A1',
                    rho_intersection = 1,
                    theta_intersection = 5,
                    arm_length = 50,
                    source_length = 50,
                    width_right = 20,
                    width_left = 20,
                    ):

    # Create blank device
    D = Device('SNSPD-yTron-integator')


    #==============================================================================
    # Create and place components*
    #==============================================================================
    cpm = D.add_device(compass_multi, size = connector_size, center = [0,-200], ports = {'N':num_devices,'S':1}, layer = 0, datatype = 0)
    f = D.add_device(flagpole(flag_size = connector_size, pole_size = [width_left,width_left], shape = 'p', taper_type = 'fillet', layer = 0, datatype = 0))
    fy = D.add_device(flagpole(flag_size = [connector_size[0],connector_size[1]*2], pole_size = [width_left,width_left], shape = 'q', taper_type = 'fillet', layer = 0, datatype = 0))
    
    t = tee(top_size = pad_flag_size, leg_size = pad_pole_size, taper_type = 'fillet', layer = 0, datatype = 0)
    pad_array = D.add_array(t, start = [-(t.width+pad_spacing)*num_devices/2, 1000], spacing = [t.width+pad_spacing, 0], num_devices = num_devices)
    s = snspd_expanded(wire_width = nanowire_width, wire_pitch = nanowire_width*3, size = snspd_size, connector_width = contact_pad_width,
                                             terminals_same_side = False, layer = 0, datatype = 0).rotate(90)
    snspd_array = D.add_array(s, start = [-100,300], spacing = [s.width+20, 0], num_devices = num_devices)
               
    y =  D.add_device(ytron_round(rho_intersection, theta_intersection, arm_length, source_length,
                      width_right, width_left, theta_resolution = 10, layer = 0, datatype = 0))
    D.connect(port = f.ports[2], destination = cpm.ports['S1'])
    D.connect(port = y.ports['right'], destination = f.ports[1])
    D.connect(port = fy.ports[1], destination = y.ports['left']).move([-200*np.sin(5/180*np.pi), 200])
    
    gnd = D.add_device(compass(size = pad_flag_size, layer = 0, datatype = 0))
    gnd.move(origin = gnd.center, destination = y.ports['source'])
    gnd.move([0,-500])
    
    
    #==============================================================================
    # Route components
    #==============================================================================
    for n in range(num_devices):
        s = snspd_array[n]; p = pad_array[n]
        D.route(port1 = s.ports[1], port2 = cpm.ports['N%s' % (n+1)], path_type = 'sine', width_type = 'sine', width1 = None, width2 = contact_pad_width*2, num_path_pts = 99, layer = 0, datatype = 0)
        D.route(port1 = p.ports['S'], port2 = s.ports[2], path_type = 'sine', width_type = 'sine', width1 = None, width2 = None, num_path_pts = 99, layer = 0, datatype = 0)
    r_ytron_gnd = D.route(y.ports['source'], gnd.ports['N'], path_type = 'sine', width_type = 'sine')
    r_ytron_pad = D.route(y.ports['left'], fy.ports[1], path_type = 'straight', width_type = 'sine')
    
    
    #==============================================================================
    # Add gold pads which are slightly inset
    #==============================================================================
    D.add_polygon( inset([r_ytron_gnd, gnd], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([r_ytron_gnd, gnd], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset(pad_array, distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([f,cpm], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([fy, r_ytron_pad], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    
    
    #==============================================================================
    # Label device
    #==============================================================================
    D.add_device( text(label, justify = 'right', size = 200, layer = 1) ).center = [-500,-500]

    return D


D = snspd_integrator(label = 'A4')
snspd_integrator(label = 'A3')
D.add_device( text('hello', justify = 'right', size = 200, layer = 1) )
quickplot(D)
D.write_gds('SNSPD Integrator.gds')

