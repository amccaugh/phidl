from __future__ import division # Makes it so 1/4 = 0.25 instead of zero
import numpy as np


from phidl import Device, Layer, quickplot
import phidl.geometry as pg
#==============================================================================
# We'll start by assuming we have a function waveguide() which already exists
# and makes us a simple waveguide rectangle
#==============================================================================

def waveguide(width = 10, height = 1):
    wg = Device('waveguide')
    wg.add_polygon( [(0, 0), (width, 0), (width, height), (0, height)] )
    wg.add_port(name = 'wgport1', midpoint = [0,height/2], width = height, orientation = 180)
    wg.add_port(name = 'wgport2', midpoint = [width,height/2], width = height, orientation = 0)
    return wg
   

#==============================================================================
# Create a blank device
#==============================================================================
# Create a new device ``D`` which will act as a blank canvas,
# and add a few waveguides to it to start out.  add_ref() returns
# the referenced object you added, allowing you to manipulate it later
D = Device('MultiWaveguide')

# We can instantiate the waveguide device by itself.  Note that when we make
# a device, we usually assign it a variable name with a capital letter
Wg1 = waveguide(width=10, height = 1)
Wg2 = waveguide(width=12, height = 2)

# 

wg1 = D.add_ref(Wg1)
wg2 = D.add_ref(Wg2)
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
D2.add_ref( pg.route(port1 = mwg1.ports[1], port2 = mwg2.ports[2], path_type = 'sine', width_type = 'straight') )
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

########
# You can specify any layer in one of three ways:
########
# 1) as a single number 0-255 representing the gds layer number, e.g. layer = 1
#    where the gds layer datatype will be automatically set to zero
DL.add_ref( pg.text('Layer1', size = 10, layer = 1) )


# 2) as a 2-element list [0,1] or tuple (0,1) representing the gds layer 
#    number (0-255) and gds layer datatype (0-255)  
DL.add_ref( pg.text('Layer2', size = 10, layer = [2,5]) ).movey(-20)


# 3) as a Layer object  
gold = Layer(name = 'goldpads', gds_layer = 3, gds_datatype = 0,
                 description = 'Gold pads liftoff', inverted = False)
DL.add_ref( pg.text('Layer3', size = 10, layer = gold) ).movey(-40)


# What you can also do is make a dictionary of layers, which lets you
# conveniently call each Layer object just by its name
layers = {
        'titanium' : Layer(gds_layer = 4, gds_datatype = 1, description = 'Gold pads liftoff', inverted = False),
        'niobium'  : Layer(gds_layer = 5, gds_datatype = 2, description = 'Gold pads liftoff', inverted = False),
        'nb_etch'  : Layer(gds_layer = 6, gds_datatype = 3, description = 'Niobium etch', inverted = False),
         }

# Now that our layers are defined, we can pass them to our text function
DL.add_ref( pg.text('Titanium', size = 10, layer = layers['titanium']) ).movey(-60)
DL.add_ref( pg.text('Niobium', size = 10, layer = layers['niobium']) ).movey(-80)
DL.add_ref( pg.text('Nb Etch', size = 10, layer = layers['nb_etch']) ).movey(-100)

quickplot(DL)

DL.write_gds('MultipleLayerText.gds')
