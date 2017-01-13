from __future__ import division # Makes it so 1/4 = 0.25 instead of zero
import numpy as np


from phidl import Device, Layer, quickplot
import phidl.geometry as pg
import phidl.routing as pr
#==============================================================================
# We'll start by assuming we have a function waveguide() which already exists
# and makes us a simple waveguide rectangle
#==============================================================================

def waveguide(width = 10, height = 1):
    WG = Device('waveguide')
    WG.add_polygon( [(0, 0), (width, 0), (width, height), (0, height)] )
    WG.add_port(name = 'wgport1', midpoint = [0,height/2], width = height, orientation = 180)
    WG.add_port(name = 'wgport2', midpoint = [width,height/2], width = height, orientation = 0)
    return WG
   

#==============================================================================
# Create a blank device
#==============================================================================
# Create a new device ``D`` which will act as a blank canvas,
# and add a few waveguides to it to start out.  add_ref() returns
# the referenced object you added, allowing you to manipulate it later
D = Device('MultiWaveguide')

# We can instantiate the waveguide device by itself.  Note that when we make
# a Device, we usually assign it a variable name with a capital letter
WG1 = waveguide(width=10, height = 1)
WG2 = waveguide(width=12, height = 2)

# We can add references from the devices WG1 and WG2 to our blank device D.
# After adding WG1, we see that the add_ref() function returns a handle to our
# reference, which we will label with lowercase letters wg1 and wg2
wg1 = D.add_ref(WG1)
wg2 = D.add_ref(WG2)

# Alternatively, we can do this all on one line
wg3 = D.add_ref(waveguide(width=14, height = 3))

quickplot(D)


#==============================================================================
# Creating polygons
#==============================================================================
# Create and add a polygon from separate lists of x points and y points
# e.g. [(x1, x2, x3, ...), (y1, y2, y3, ...)]
poly1 = D.add_polygon( [(8,6,7,9), (6,8,9,5)] )

# Alternatively, create and add a polygon from a list of points
# e.g. [(x1,y1), (x2,y2), (x3,y3), ...] using the same function
poly2 = D.add_polygon( [(0, 0), (1, 1), (1, 3), (-3, 3)] )

quickplot(D)


#==============================================================================
# Manipulating geometry 1 - Basic movement and rotation
#==============================================================================
# There are several actions we can take to move and rotate the geometry.  These
# actions include movement, rotation, and reflection.

wg1.move([10,4]) # Shift the second waveguide we created over by dx = 10, dy = 4
wg2.move(origin = [1,1], destination = [2,2]) # Shift the second waveguide over by dx = 1, dy = 1
wg3.move([1,1], [5,5], axis = 'y') # Shift the third waveguide over by dx = 0, dy = 4 (motion only along y-axis)
poly1.movey(4) # Same as specifying axis='y' in the move() command
poly2.movex(4) # Same as specifying axis='x'' in the move() command
wg3.movex(30,40) # Moves "from" x=30 "to" x=40 (e.g. shifts wg3 by +10 in the x-direction)

wg1.rotate(45) # Rotate the first waveguide by 45 degrees around (0,0)
wg2.rotate(30, center = [1,1]) # Rotate the second waveguide by 30 degrees around (1,1)

wg1.reflect(p1 = [1,1], p2 = [1,3]) # Reflects wg3 across the line formed by p1 and p2


#==============================================================================
# Manipulating geometry 2 - Properties
#==============================================================================
# Each Device and DeviceReference object has several properties which can be used to learn
# information about the object (for instance where it's center coordinate is).  Several
# of these properties can actually be used to move the geometry by assigning them
# new values

wg1.bbox # Will return the bounding box of wg1 in terms of [(xmin, ymin), (xmax, ymax)]
wg1.xsize # Will return the width of wg1 in the x dimension
wg1.ysize # Will return the height of wg1 in the y dimension

wg1.center # Gives you the center coordinate of its bounding box
wg1.center = [4,4] # Shift wg1 such that the center coordinate of its bounding box is at (4,4)

wg2.xmax # Gives you the rightmost (+x) edge of the wg2 bounding box
wg2.xmax = 25 # Moves wg2 such that it's rightmost edge is at x = 25

wg3.ymin # Gives you the bottommost (+y) edge of the wg3 bounding box
wg3.ymin = -14 # Moves wg3 such that it's bottommost edge is at y = -14


quickplot(D)


#==============================================================================
# Manipulating geometry 3 - Smarter movement with ports
#==============================================================================
# All the waveguides we made have two ports: 'wgport1' and 'wgport2'  We can 
# use these names in place of (x,y) pairs.  For instance, if we want to move
# wg1 such that its port 'wgport1' rests on the origin, we do:
wg1.move(origin = 'wgport1', destination = [0,0])
# Alternatively, we can use the Port object itself in the same manner.  We can
# access the Port objects for any Device (or DeviceReference) by calling device.ports,
# --which returns a Python dictionary--and accessing its value with the key
wg3.move(origin = wg3.ports['wgport1'], destination = [0,0])
# We can even move one port to another 
wg2.move(origin = wg2.ports['wgport1'], destination = wg3.ports['wgport2'])
# Several functions beyond just move() can take Ports as inputs
wg1.rotate(angle = -60, center = wg1.ports['wgport2'])
wg3.reflect(p1 = wg3.ports['wgport1'].midpoint, p2 = wg3.ports['wgport1'].midpoint + np.array([1,0]))

quickplot(D)


#==============================================================================
# Manipulating geometry 4 - Chaining commands
#==============================================================================
# Many of the functions in Device return the object they manipulate.  We can use
# this to chain commands in a single line. For instance this:
wg1.rotate(angle = 15, center = [0,0])
wg1.move([10,20])
# ... is equivalent to this expression
wg1.rotate(angle = 15, center = [0,0]).move([10,20])



#==============================================================================
# Connecting devices with connect()
#==============================================================================
wg1.connect(port = 'wgport1', destination = wg2.ports['wgport2'])
wg3.connect(port = 'wgport2', destination = wg2.ports['wgport1'])

quickplot(D)



#==============================================================================
# Adding ports
#==============================================================================
# Although our waveguides have ports, ``D`` itself does not -- it only draws
# the subports (ports of wg1, wg2, wg3) as a convience.  We need to add ports
# that we specifically want in our new device ``D``
D.add_port(port = wg1.ports['wgport2'], name = 1)
D.add_port(port = wg3.ports['wgport1'], name = 2)

quickplot(D)



#==============================================================================
# Taking things a level higher
#==============================================================================
# Now that we have our device ``D`` which is a multi-waveguide device, we
# can add references to that device in a new blank canvas we'll call ``D2``.
# We'll add two copies of ``D`` to D2, and shift one so we can see them both
D2 = Device('MultiMultiWaveguide')
mwg1 = D2.add_ref(D)
mwg2 = D2.add_ref(D)
mwg2.move(destination = [10,10])

quickplot(D2)

# Like before, let's connect mwg1 and mwg2 together then offset them slightly
mwg1.connect(port = 1, destination = mwg2.ports[2])
mwg2.move(destination = [30,30])

quickplot(D2)


#==============================================================================
# Routing
#==============================================================================
# Routing allows us to connect two ports which face each other with a smooth
# polygon.  Since we connected our two 
D2.add_ref( pr.route_basic(port1 = mwg1.ports[1], port2 = mwg2.ports[2], path_type = 'sine', width_type = 'straight') )
quickplot(D2)


#==============================================================================
# Adding text
#==============================================================================
# The function text() creates a Device, just like waveguide.  Use it and 
# manipulate it like any other Device
t = D2.add_ref( pg.text('Hello\nworld!', size = 10, justify = 'center'))
t.move([0,40]).rotate(45)
quickplot(D2)



#==============================================================================
# Labeling
#==============================================================================
# This label will display in a GDS viewer, but will not be rendered
# or printed like the polygons created by the text()
D2.annotate('First label', mwg1.center)
D2.annotate('Second label', mwg2.center)


#==============================================================================
# Saving the file as a .gds
#==============================================================================
D2.write_gds('MultiMultiWaveguideTutorial.gds')



#==============================================================================
# Using Layers
#==============================================================================
# Let's make a new blank device DL and add some text to it, but this time on
# different layers
DL = Device()

# You can specify any layer in one of three ways:
# 1) as a single number 0-255 representing the gds layer number, e.g. layer = 1
# where the gds layer datatype will be automatically set to zero
DL.add_ref( pg.text('Layer1', size = 10, layer = 1) )


# 2) as a 2-element list [0,1] or tuple (0,1) representing the gds layer 
# number (0-255) and gds layer datatype (0-255)  
DL.add_ref( pg.text('Layer2', size = 10, layer = [2,5]) ).movey(-20)


# 3) as a Layer object  
gold = Layer(name = 'goldpads', gds_layer = 3, gds_datatype = 0,
                 description = 'Gold pads liftoff')
DL.add_ref( pg.text('Layer3', size = 10, layer = gold) ).movey(-40)


# What you can also do is make a dictionary of layers, which lets you
# conveniently call each Layer object just by its name.  You can also specify
# the layer color using an RGB triplet e.g (0.1, 0.4, 0.2), an HTML hex color 
# (e.g. #a31df4), or a CSS3 color name (e.g. 'gold' or 'lightblue'
# see http://www.w3schools.com/colors/colors_names.asp )
# The 'alpha' argument also lets you specify how transparent that layer should
# look when using quickplot (has no effect on the written GDS file)
layers = {
        'titanium' : Layer(gds_layer = 4, gds_datatype = 0, description = 'Titanium resistor', color = 'gray'),
        'niobium'  : Layer(gds_layer = 5, gds_datatype = 0, description = 'Niobium liftoff', color = (0.4,0.1,0.1)),
        'nb_etch'  : Layer(gds_layer = 6, gds_datatype = 3, description = 'Niobium etch', color = 'lightblue', alpha = 0.2),
         }

# Now that our layers are defined, we can pass them to our text function
l1 = DL.add_ref( pg.text('Titanium layer', size = 10, layer = layers['titanium']) ).movey(-60)
l2 = DL.add_ref( pg.text('Niobium layer', size = 10, layer = layers['niobium']) ).movey(-80)
l3 = DL.add_ref( pg.text('Nb Etch layer', size = 10, layer = layers['nb_etch']) ).movey(-90).movex(5)

quickplot(DL)

DL.write_gds('MultipleLayerText.gds')

#==============================================================================
# Annotation
#==============================================================================
# We can also annotate our devices, in order to record information directly
# into the final GDS file without putting any extra geometry onto any layer

# Let's add an annotation to our Multi-Layer Text GDS file
DL.annotate(text = 'This is layer1\nit will be titanium', position = l1.center)
DL.annotate(text = 'This is niobium', position = l2.center)

# It's very useful for recording information about the devices or layout
DL.annotate(text = 'The x size of this\nlayout is %s' % DL.xsize,
            position = (DL.xmax, DL.ymax), layer = 255)

# Again, note we have to write the GDS for it to be visible (view in KLayout)
DL.write_gds('MultipleLayerText.gds')




#==============================================================================
# Constructing a Device from set of parameters (dictionary or config file)
#==============================================================================
# Say we want to make a more complicated waveguide which requires more
# parameters.  Instead of passing them individually, we can store them in a
# dictionary (or configuration file) and pass that dictionary to the Device()
# function.

def complicated_waveguide(width = 10, height = 1, x = 10, y = 25, rotation = 15):
    C = Device('complicated_waveguide')
    C.add_polygon( [(0, 0), (width, 0), (width, height), (0, height)] )
    C.add_port(name = 1, midpoint = [0,height/2], width = height, orientation = 180)
    C.add_port(name = 2, midpoint = [width,height/2], width = height, orientation = 0)
    C.rotate(angle = rotation, center = (0,0))
    C.move((x,y))
    return C
    
cwg_parameters = {
            'width' : 14,
            'height' : 1,
            'x' : 15,
            'y' : 20,
            'rotation' : 0
            }

# We can either create the complicated_waveguide() the normal way
C1 = complicated_waveguide(width = 14, height = 1, x = 15, y = 20, rotation = 0)
quickplot(C1)

# Or we can pass the complicated_waveguide function and our parameter list
# to the Device() function which will generate it for us using our config
C2 = Device(complicated_waveguide, config = cwg_parameters)
quickplot(C2)


# We can also override any parameter we like in our dictionary of parameters
# by adding keyword arguments -- the input dictionary is untouched afterwards
C3 = Device(complicated_waveguide, config = cwg_parameters, width = 500, rotation = 35)
quickplot(C3)


# The most useful implementation of this is to keep a standard set of 
# parameters and then override certain parameters each iteration of the for 
# loop. Say we want to use our standard cwg_parameters but change the height
#  each time:
D = Device()
for h in [0.1, 0.5, 1, 2, 4]:
    C4 = Device(complicated_waveguide, config = cwg_parameters, height = h)
    c4 = D.add_ref( C4 )
    c4.ymin = D.ymax + 10
quickplot(D)




#==============================================================================
# Keeping track of geometry using the "alias" functionality
#==============================================================================
# It can be useful to keep track of our DeviceReferences without
# needing to assign the reference to a variable.  We can do this by specifying
# an 'alias' for the added DeviceReference.

# For instance, if we wanted to keep track of a circle references twice in D,
# we might normally assign each reference to a separate variable:
D = Device()
C = pg.circle()
c1 = D.add_ref(C)   # Add first reference
c2 = D.add_ref(C)   # Add second reference
c2.x += 15          # Move the second circle over by 10
quickplot(c2)
quickplot(D)


# But rather than cluttering up the list of variables with these refernces,
# we can instead create 'aliases' to each reference, and call them directly
# out of D like you would with a Python dictionary.  For example:
D = Device()
C = pg.circle()
D.add_ref(C, alias = 'circle1') # Add first reference 
D.add_ref(C, alias = 'circle2') # Add second reference
D['circle2'].x += 15            # Moving the second circle over by 10
# Note that at this point, D['circle2'] is equivalent to the variable c2
# we made above
quickplot(D['circle2'], label_aliases = True)
quickplot(D, label_aliases = True)

# You can also access the list of aliases for your Device whenever you want 
# to by accessing Device.aliases, which is a Python dictionary.  For example:
print(D.aliases)
print(D.aliases.keys())


#==============================================================================
# Extracting shapes
#==============================================================================
# Say you want to copy a complicated shape from one layer to another.  You 
# can do this using the D.extract() function, which will strip out the raw
# polygon points from D and allow you to add them to another layer
D = Device()
E1 = pg.ellipse(layer = 1)
E2 = pg.ellipse(layer = 2)
E3 = pg.ellipse(layer = 1)
D.add_ref(E1)
D.add_ref(E2).movex(15)
D.add_ref(E3).movex(30)
quickplot(D)

D2 = Device()
ellipse_polygons = D.extract(layers = 1)
D2.add_polygon(ellipse_polygons, layer = 3)
quickplot(D2)
