from reinforcement.mesh import create_concrete_slab, read_xdmf
from reinforcement.rebar import ElasticTrussRebar
import dolfinx as dfx
import ufl
import numpy as np
from petsc4py import PETSc


parameters_steel = {"E": 42.0, "nu": 0.3, "A": 0.02}
parameters_concrete = {"E": 21.0, "nu": 0.3}

force = 0.8

length = 2
width = 0.3
height = 0.3
point1 = [0, 0, 0]
point2 = [length, width, height]

margin = 0.03
nx = 1
ny = 2
h = 0.05
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
        np.isclose(x[0], length / 2, atol=h), np.isclose(x[1], height, atol=h)
    ),
)
facet_markers = np.full(len(facet_indices), marker).astype(np.int32)
facet_tag = dfx.mesh.meshtags(concrete_mesh, fdim, facet_indices, facet_markers)

concrete_mesh.topology.create_connectivity(
    concrete_mesh.topology.dim - 1, concrete_mesh.topology.dim
)
ds = ufl.Measure("ds", domain=concrete_mesh, subdomain_data=facet_tag)


force_form = force * ufl.dot(ufl.FacetNormal(concrete_mesh), u_) * ds(42)

# dirichlet bcs

# right side
def left(x):
    return np.logical_and(np.isclose(x[0], margin), np.isclose(x[1], 0.0))


def right(x):
    return np.logical_and(np.isclose(x[0], length - margin), np.isclose(x[1], 0.0))


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

bcs = [bc_x, bc_y, bc_z]

u = dfx.fem.Function(P1, name="Displacement")
b = dfx.fem.petsc.assemble_vector(dfx.fem.form(force_form))

A = dfx.fem.petsc.assemble_matrix(dfx.fem.form(a), bcs=bcs)
A.assemble()
dfx.fem.apply_lifting(b.array, [dfx.fem.form(a)], [bcs])

# apply reinforcement
rebar.apply_to_stiffness(A, u.vector)
rebar.apply_to_forces(b, u.vector)
# solve the system
solver = PETSc.KSP().create(concrete_mesh.comm)
solver.setOperators(A)
solver.solve(b, u.vector)
