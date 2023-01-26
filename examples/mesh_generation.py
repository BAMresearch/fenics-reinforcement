"""Example for mesh creation using the mesh submodule"""

from reinforcement import create_concrete_slab, create_xdmf
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

create_concrete_slab(point1, point2, nx, ny, margin, s_exp, filename, where="lower")

concrete_mesh,rebar_mesh = create_xdmf(msh_file=filename)