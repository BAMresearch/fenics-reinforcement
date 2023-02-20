#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:10:15 2023

@author: mcappone
"""

from test_analytical_solution import analytical_solution
# example (no real values)
q0 = 10
B = 1
H = 2
L = 1.5
zr = 0.1
params_rebar = {"Youngs_Modulus_steel":10, "density_steel":5, "amount": 2, "area":3}
params_concrete = {"Youngs_Modulus_concrete":5,"density_concrete":3}
y0 = 0.5
example = analytical_solution(q0,B,H,L,zr,params_rebar,params_concrete)
w = example.evaluate(y0)
print(w)