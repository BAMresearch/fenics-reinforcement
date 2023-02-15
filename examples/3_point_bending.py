from reinforcement.mesh import create_concrete_slab, read_xdmf
from reinforcement.rebar import ElasticTrussRebar
import dolfinx as dfx
import ufl
import numpy as np
from petsc4py import PETSc
from pint import UnitRegistry

ureg = UnitRegistry()

class NonlinearReinforcementProblem(dfx.fem.petsc.NonlinearProblem):
    """
    This class demonstrates how the reinforcement could be used in a nonlinear problem.
    """
    def __init__(self, R, dR, u, rebar, bcs = [], form_compiler_params={}, jit_params={}):
        super().__init__(R, u, bcs, dR,form_compiler_params, jit_params)
        self.rebar = rebar
    
    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        super().F(x,b)
        self.rebar.apply_to_forces(b, x, sign=-1.)
        # The implementation in a real nonlinear case might look a little different

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        super().J(x,A)
        self.rebar.apply_to_stiffness(A, x)
        # The implementation in a real nonlinear case might look a little different


parameters_steel = {
    "E": (210. * ureg.gigapascal).to_base_units().magnitude,
    "nu": 0.3,
    "A": (np.pi * (0.75 * ureg.centimeters)**2).to_base_units().magnitude,
    }
parameters_concrete = {
    "E": (25 * ureg.gigapascal).to_base_units().magnitude,
    "nu": 0.3,
    }

force = (50 * ureg.kilonewton).to_base_units().magnitude

length = (2 * ureg.meter).to_base_units().magnitude
width = (30 * ureg.centimeter).to_base_units().magnitude 
height = (30 * ureg.centimeter).to_base_units().magnitude

point1 = [0., 0., 0.]
point2 = [length, width, height]

margin = (3 * ureg.centimeter).to_base_units().magnitude
nx = 0
ny = 2
h = (3 * ureg.centimeter).to_base_units().magnitude
msh_filename = "test_mesh.msh"
xdmf_filenames = ["concrete_mesh.xdmf", "rebar_mesh.xdmf"]


create_concrete_slab(
    point1, point2, nx, ny, margin, h, msh_filename, xdmf_filenames, where="lower"
)

concrete_mesh, rebar_mesh = read_xdmf(xdmf_filenames)

P1 = dfx.fem.VectorFunctionSpace(concrete_mesh, ("CG", 1))

rebar = ElasticTrussRebar(concrete_mesh, rebar_mesh, P1, parameters_steel)


def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma(v, parameters):
    e = eps(v)
    lam = (
        parameters["E"]
        * parameters["nu"]
        / (1 + parameters["nu"])
        / (1 - 2 * parameters["nu"])
    )
    mu = parameters["E"] / 2 / (1 + parameters["nu"])
    return lam * ufl.tr(e) * ufl.Identity(3) + 2.0 * mu * e


u = dfx.fem.Function(P1, name="Displacement")

u_ = ufl.TrialFunction(P1)
v_ = ufl.TestFunction(P1)
a = ufl.inner(sigma(u_, parameters_concrete), eps(v_)) * ufl.dx

# boundary forces
fdim = concrete_mesh.topology.dim - 1
marker = 42

facet_indices = dfx.mesh.locate_entities(
    concrete_mesh,
    fdim,
    lambda x: np.logical_and(
        np.isclose(x[0], length / 2, atol=h), np.isclose(x[2], height, atol=h)
    ),
)
facet_markers = np.full(len(facet_indices), marker).astype(np.int32)
facet_tag = dfx.mesh.meshtags(concrete_mesh, fdim, facet_indices, facet_markers)

concrete_mesh.topology.create_connectivity(
    concrete_mesh.topology.dim - 1, concrete_mesh.topology.dim
)
ds = ufl.Measure("ds", domain=concrete_mesh, subdomain_data=facet_tag)


external_force_form = force * ufl.dot(ufl.FacetNormal(concrete_mesh), v_) * ds(42)
internal_force_form = ufl.inner(eps(v_), sigma(u, parameters_concrete)) * ufl.dx
residual =   -internal_force_form

# right side
def left(x):
    return np.logical_and(np.isclose(x[0], margin), np.isclose(x[2], 0.0))


def right(x):
    return np.logical_and(np.isclose(x[0], length - margin), np.isclose(x[2], 0.0))


# left side
boundary_entities_left = dfx.mesh.locate_entities_boundary(
    concrete_mesh, concrete_mesh.topology.dim - 2, left
)
boundary_dofs_left = [
    dfx.fem.locate_dofs_topological(
        P1.sub(i), concrete_mesh.topology.dim - 2, boundary_entities_left
    )
    for i in range(3)
]

boundary_entities_right = dfx.mesh.locate_entities_boundary(
    concrete_mesh, concrete_mesh.topology.dim - 2, right
)
boundary_dofs_right = [
    dfx.fem.locate_dofs_topological(
        P1.sub(i), concrete_mesh.topology.dim - 2, boundary_entities_right
    )
    for i in range(3)
]

bc_z = dfx.fem.dirichletbc(
    PETSc.ScalarType(0),
    np.concatenate((boundary_dofs_left[2], boundary_dofs_right[2])),
    P1.sub(2),
)
bc_x = dfx.fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs_left[0], P1.sub(0))
bc_y = dfx.fem.dirichletbc(
    PETSc.ScalarType(0),
    np.concatenate((boundary_dofs_left[1], boundary_dofs_right[1])),
    P1.sub(1),
)

boundary_entities_upper = dfx.mesh.locate_entities_boundary(
    concrete_mesh, concrete_mesh.topology.dim - 2, 
    lambda x: np.logical_and(
        np.isclose(x[0], length / 2, atol=h), np.isclose(x[2], height, atol=h)
    ),
)
boundary_dofs_upper = dfx.fem.locate_dofs_topological(P1.sub(2), concrete_mesh.topology.dim - 2, boundary_entities_upper)
bc_upper = dfx.fem.dirichletbc(PETSc.ScalarType(-0.005), boundary_dofs_upper, P1.sub(2))

bcs = [bc_x, bc_y, bc_z, bc_upper]

problem = NonlinearReinforcementProblem(residual, a, u, rebar, bcs)
solver = dfx.nls.petsc.NewtonSolver(concrete_mesh.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

tup = solver.solve(u)
print(tup)
with dfx.io.XDMFFile( concrete_mesh.comm, "displacements.xdmf", "w") as f:
    f.write_mesh(concrete_mesh)
    f.write_function(u)
