"""Example for mesh creation using the mesh submodule"""

from reinforcement import create_concrete_slab
l = 1.5 # length 
w = 2 # width
h = 1 # height 
point1 = [0,0,0]
point2 = [l,w,h]
margin=0.25 # minimum distance from the outer edges of the concrete mesh to the reinforcement nodes 
nx = 1 # reinforcement density (number of rebars) in x direction
ny=5 # reinforcement density (number of rebars) in y direction
s_exp = 0.1 # maximal element size
msh_filename = "test_mesh.msh"
xdmf_filenames = ["concrete_mesh.xdmf", "rebar_mesh.xdmf"]

create_concrete_slab(point1, point2, nx, ny, margin, s_exp, msh_filename, xdmf_filenames, where="lower")
#create_concrete_slab([0.,0.,0.], [42.,42.,42.], 0, 2, 0.5, h, "msh_filename", ["xdmf_concrete","xdmf_rebar"], where="lower")

