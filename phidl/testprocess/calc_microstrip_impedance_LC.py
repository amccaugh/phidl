# %% Equations taken from
# Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
# Computer-Aided Design.  http://doi.org/10.1109/MWSYM.1980.1124303

from __future__ import division # Otherwise integer division e.g.  2 / 3 = 0

import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from scipy.optimize import fmin


def microstrip_Z(wire_width, dielectric_thickness, eps_r):
    # Note these equations can be further corrected for thick films (Hammersted Eqs 6-9)
    # and also for frequency since microstrips are dispersive  (Hammersted Eqs 10-12)

    u = wire_width/dielectric_thickness
    eta = 376.73 # Vacuum impedance
    
    a = 1 + log((u**4 + (u/52)**2)/(u**4 + 0.432))/49 + log(1 + (u/18.1)**3)/18.7;
    b = 0.564*((eps_r-0.9)/(eps_r+3))**0.053;
    F = 6 + (2*pi-6)*exp(-(30.666/u)**0.7528);
    eps_eff = 0.5*(eps_r+1) + 0.5*(eps_r-1)*(1 + 10/u)**(-a*b);
    Z = eta/(2*pi) * log(F/u + sqrt(1+(2/u)**2)) /sqrt(eps_eff);
    return Z,eps_eff


def microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r):
    # Use the fact that v = 1/sqrt(L_m*C_m) = 1/sqrt(eps*mu) and
    # Z = sqrt(L_m/C_m)   [Where L_m is inductance per meter]

    Z, eps_eff =  microstrip_Z(wire_width, dielectric_thickness, eps_r)
    eps0 =  8.854e-12
    mu0 = 4*pi*1e-7
    
    
    eps = eps_eff*eps0
    mu = mu0
    L_m = sqrt(eps*mu)*Z
    C_m = sqrt(eps*mu)/Z
    return L_m, C_m


def microstrip_Z_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq):
    # Add a kinetic inductance and recalculate the impedance, be careful
    # to input Lk as a per-meter inductance

    L_m, C_m = microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r)
    Lk_m = Lk_per_sq*(1.0/wire_width)
    Z = sqrt((L_m+Lk_m)/C_m)
    
    return Z
    
    
def find_microstrip_wire_width(Z_target, dielectric_thickness, eps_r, Lk_per_sq):
    
    def error_fun(wire_width):
        Z_guessed = microstrip_Z_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq)
        return (Z_guessed-Z_target)**2 # The error
    
    x0 = dielectric_thickness
    w = fmin(error_fun, x0, args=(), disp=False)
    return w[0]



# %% Plot example: Microstrip impedance vs wire width

# Define relative permittivity, microstrip conductor width, and thickness of dielectric
eps_r = 3.9 # SiO2 relative permittivity
wire_width = 100e-9
dielectric_thickness = 250e-9
Lk_per_sq = 250e-12 # Henry/sq

w = np.linspace(0.1e-6, 30e-6, 1000)
Z = microstrip_Z_with_Lk(w, dielectric_thickness, eps_r, Lk_per_sq)

plt.plot(w*1e6,Z)
plt.xlabel('Wire width (um)'); plt.ylabel('Microstrip impedance Z (Ohms)')
plt.title('Microstrip impedance vs wire width \n for Lk = %s pH/sq on %s nm SiO2' \
     % (Lk_per_sq*1e12, dielectric_thickness*1e9))



# %% Plot example: Wire width vs dielectric thickness

d_list = np.linspace(50e-9, 250e-9, 100)
w = [find_microstrip_wire_width(50, d, eps_r, Lk_per_sq) for d in d_list]

plot(np.array(d_list)*1e9, np.array(w)*1e6)
plt.xlabel('Dielectric thickness (nm)'); plt.ylabel('Wire width (um)')



# %% Calculate taper impedances for Hecken and hyperbolic tapers

from scipy.special import iv as besseli
from scipy import integrate

def G_integrand(xip, B):
    return besseli(0, B*sqrt(1-xip**2))

def G(xi, B):
    return B/sinh(B)*integrate.quad(G_integrand, 0, xi, args = (B))[0]


def hecken_taper(x, l, B = 4.0091, Z1 = 50, Z2 = 75):
    # xi refers to the normalized length of the wire [-1 to +1]
    xi = 2*x/l-1
    Z = exp( 0.5*log(Z1*Z2) + 0.5*log(Z2/Z1)*G(xi, B) )
    return Z


def hyperbolic_taper(x, l = 1e-3, Z1 = 50, Z2 = 75):
    # l is the total length of the wire, x is an array from 0 to l
    # Returns Z(x)
    a = 6
    Z = sqrt(Z1*Z2)*exp(tanh(a*(x/l-0.5))/(2*tanh(a/2))*log(Z2/Z1))
    return Z





# %% Calculate wire contours for hyperbolic or hecken taper

dielectric_thickness = 250e-9
eps_r = 3.9 # SiO2 relative permittivity
SNSPD_wire_width = 300e-9
Lk_per_sq = 250e-12 # Henry/sq

# Impedance and length of taper
Z1 = 50
Z2 = microstrip_Z_with_Lk(SNSPD_wire_width, dielectric_thickness, eps_r, Lk_per_sq)
taper_length = 2e-3
x_taper = linspace(0,taper_length,100)

# Hecken taper parameters
B = 4.0091
emax = B/sinh(B)*0.21723
ripple_max_dB = 20*log10(1/emax)



# Calculate the impedances and their corresponding microstrip widths
Z_hyperbolic = [hyperbolic_taper(x, taper_length, Z1, Z2) for x in x_taper]
Z_hecken = [hecken_taper(x, taper_length, B, Z1, Z2) for x in x_taper]

w_hecken = [find_microstrip_wire_width(Z, dielectric_thickness, eps_r, Lk_per_sq) for Z in Z_hecken]
w_hyperbolic = [find_microstrip_wire_width(Z, dielectric_thickness, eps_r, Lk_per_sq) for Z in Z_hyperbolic]

subplot(2,1,1)
plot(x_taper*1e3, Z_hyperbolic, x_taper*1e3, Z_hecken)
xlabel('Distance (mm)')
ylabel('Impedance (Ohm)')
legend(['Hecken taper', 'Hyperbolic taper'],loc='upper left')
subplot(2,1,2)
plot(x_taper*1e3, np.array(w_hyperbolic)*1e6, x_taper*1e3, np.array(w_hecken)*1e6)
xlabel('Distance (mm)')
ylabel('Taper width (um)')



# %% Make GDS out of wire width
import gdspy


w_taper = w_hecken

gdspy.Cell.cell_dict.clear()
poly_cell = gdspy.Cell('POLYGONS')
xpts = x_taper.tolist() + [x_taper[-1], x_taper[0]]
ypts = w_taper + [0, 0]
xpts_um = np.array(xpts)*1e6
ypts_um = np.array(ypts)*1e6
poly1 = gdspy.Polygon(zip(xpts_um,ypts_um), layer=2, datatype=3)
poly_cell.add(poly1)
gdspy.LayoutViewer()
gdspy.gds_print('hecken_taper.gds', unit=1.0e-6, precision=1.0e-9)