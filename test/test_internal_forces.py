from reinforcement.mesh import create_concrete_slab, read_xdmf
from reinforcement.rebar import ElasticTrussRebar
import dolfinx as dfx
import numpy as np
import ufl
from pint import UnitRegistry

ureg = UnitRegistry()

parameters_steel = {
    "E": (210. * ureg.gigapascal).to_base_units().magnitude,
    "nu": 0.3,
    "A": (np.pi * (0.75 * ureg.centimeters)**2).to_base_units().magnitude,
    }
parameters_concrete = {
    "E": (25 * ureg.gigapascal).to_base_units().magnitude,
    "nu": 0.3,
    }

length = (10 * ureg.centimeter).to_base_units().magnitude
width = (10 * ureg.centimeter).to_base_units().magnitude 
height = (10 * ureg.centimeter).to_base_units().magnitude

point1 = [0., 0., 0.]
point2 = [length, width, height]

margin = (2 * ureg.centimeter).to_base_units().magnitude
nx = 3
ny = 3
h = (2 * ureg.centimeter).to_base_units().magnitude
z_rebar = (5 * ureg.centimeter).to_base_units().magnitude
msh_filename = "test_mesh.msh"
xdmf_filenames = ["concrete_mesh.xdmf", "rebar_mesh.xdmf"]


create_concrete_slab(
    point1, point2, nx, ny, margin, h, msh_filename, xdmf_filenames, z=[z_rebar]
)

concrete_mesh, rebar_mesh = read_xdmf(xdmf_filenames)

P1 = dfx.fem.VectorFunctionSpace(concrete_mesh, ("CG", 1))

rebar = ElasticTrussRebar(concrete_mesh, rebar_mesh, P1, parameters_steel)

u = dfx.fem.Function(P1)
b = dfx.fem.Function(P1)
u_ = ufl.TrialFunction(P1)
v_ = ufl.TestFunction(P1)
a = ufl.inner(u_,v_) * ufl.dx

A = dfx.fem.petsc.create_matrix(dfx.fem.form(a))
f_int = b.vector

#u.vector.array[:] = np.random.random(u.vector.array.size)
u.interpolate(lambda x:(0.005*x[0], -0.005*x[1], 0.*x[2]))

rebar.apply_to_stiffness(A, u.vector)
rebar.apply_to_forces(f_int, u.vector)

def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s

A_dense = A.convert("dense")
A_arr = petsc2array(A_dense)

def test_f_int_equals_Ku():
    Au = A_arr@u.vector.array
    f_arr = f_int.array
    assert np.linalg.norm(Au-f_arr)/np.linalg.norm(f_arr) < 1e-12
