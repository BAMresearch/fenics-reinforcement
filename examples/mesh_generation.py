#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:14:24 2023

@author: mcappone
"""
# example on how to use mesh.py
from reinforcement import create_concrete_slab
l = 1.5 # length 
w = 2 # width
h = 1 # height 
point1 = [0,0,0]
point2 = [l,w,h]
margin=0.25 # minimum distance from the outer edges of the concrete mesh to the reinforcement nodes 
nx = 12 # reinforcement density (number of elements) in x direction
ny=7 # reinforcement density (number of elements) in y direction
s_exp = 0.1 # maximal element size
filename = "test_mesh.msh"
create_concrete_slab(point1=point1, point2=point2, n_x=nx, n_y=ny, margin=margin, s=s_exp, where="upper")