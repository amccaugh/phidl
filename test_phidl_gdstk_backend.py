from phidl import Device
from phidl import quickplot as qp # Rename "quickplot()" to the easier "qp()"
import phidl.geometry as pg
import numpy as np
# Create a blank device (essentially an empty GDS cell with some special features)
D = Device('mydevice')

poly1 = D.add_polygon( [(-8,6,7,9), (-6,8,17,5)], layer = 0)
# T = pg.text('Hello!')
# print("HERE")
C = pg.arc(radius = 25, width = 2, theta = 45, layer = 1)
c = D << C # Add the arc we created

# R = pg.rectangle(size = [5,10], layer = 2)
# r = D << R

def waveguide(width = 10, height = 1, layer = 0):
    WG = Device('waveguide')
    WG.add_polygon( [(0, 0), (width, 0), (width, height), (0, height)], layer = layer)
    WG.add_port(name = 'wgport1', midpoint = [0,height/2], width = height, orientation = 180)
    WG.add_port(name = 'wgport2', midpoint = [width,height/2], width = height, orientation = 0)
    return WG

wg1 = D << waveguide(width=6, height = 2.5, layer = 1)
wg2 = D << waveguide(width=11, height = 2.5, layer = 2)
wg3 = D << waveguide(width=15, height = 2.5, layer = 3)

wg2.movey(10).rotate(10)
wg3.movey(20).rotate(15)
# print(D.references)

# text2 = D << T.flatten()
# text2.movex(-10)
# print(text2)
D.write_gds("modified_phidl_test.gds")
# D.write_oas("modified_phidl_test.oas")
print("END OF FILE")