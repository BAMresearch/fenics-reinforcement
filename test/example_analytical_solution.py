#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:10:15 2023

@author: mcappone
"""

from analytical_solution import analytical_solution
import matplotlib.pyplot as plt
import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()
#example on how to derive an analytical solution for the three-point bending test
force = (50*ureg.kilonewton).to_base_units().magnitude
B = (2 * ureg.meter).to_base_units().magnitude
L = (30 * ureg.centimeter).to_base_units().magnitude 
H = (30 * ureg.centimeter).to_base_units().magnitude
#B = 2.
#H = 0.3
#L = 0.3
q0 = force  
zr = (5*ureg.centimeter).to_base_units().magnitude
#params_rebar = {"Youngs_Modulus_steel":10, "density_steel":5, "amount": 2, "area":3}
#params_concrete = {"Youngs_Modulus_concrete":5,"density_concrete":3}
parameters_steel = {
    "E": (210. * ureg.gigapascal).to_base_units().magnitude,
    "nu": 0.3,
    "A": (np.pi * (0.75 * ureg.centimeters)**2).to_base_units().magnitude,
    "rho":(7850 * ureg.kilogram/ureg.meter**3).to_base_units().magnitude,
    "amount":0,
    }
parameters_concrete = {
    "E": (25 * ureg.gigapascal).to_base_units().magnitude,
    "nu": 0.3,
    "rho": (2.4*ureg.gram/ureg.centimeter**3).to_base_units().magnitude,
    }
y0 = 0.5
y = np.linspace(0.,B,100)
example = analytical_solution(q0,B,H,L,zr,parameters_steel,parameters_concrete)
w = example.evaluate(y)
print(max(w))
plt.plot(y,w)
plt.show()