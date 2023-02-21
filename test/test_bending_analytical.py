from reinforcement.mesh import create_concrete_slab, read_xdmf
from reinforcement.rebar import ElasticTrussRebar
import dolfinx as dfx
import ufl
import numpy as np
from petsc4py import PETSc
from pint import UnitRegistry
from analytical_solution import analytical_solution

ureg = UnitRegistry()

parameters_steel = {
    "E": (210. * ureg.gigapascal).to_base_units().magnitude,
    "nu": 0.3,
    "A": (np.pi * (0.75 * ureg.centimeters)**2).to_base_units().magnitude,
    "rho":(7850 * ureg.kilogram/ureg.meter**3).to_base_units().magnitude,
    "amount":2,
    }
parameters_concrete = {
    "E": (25 * ureg.gigapascal).to_base_units().magnitude,
    "nu": 0.3,
    "rho": (2.4*ureg.gram/ureg.centimeter**3).to_base_units().magnitude,
    }

class NonlinearReinforcementProblem(dfx.fem.petsc.NonlinearProblem):
    """
    This class demonstrates how the reinforcement could be used in a nonlinear problem.
    """
    def __init__(self, R, dR, u, rebar, bcs = [], form_compiler_params={}, jit_params={}):
        super().__init__(R, u, bcs, dR,form_compiler_params, jit_params)
        self.rebar = rebar
    
    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        super().F(x,b)
        self.rebar.apply_to_forces(b, x)
        # The implementation in a real nonlinear case might look a little different

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        super().J(x,A)
        self.rebar.apply_to_stiffness(A, x)
        # The implementation in a real nonlinear case might look a little different
    

length = (2 * ureg.meter).to_base_units().magnitude
width = (30 * ureg.centimeter).to_base_units().magnitude 
height = (30 * ureg.centimeter).to_base_units().magnitude

force = (50 * ureg.kilonewton).to_base_units().magnitude
pressure = force / (length*width)

point1 = [0., 0., 0.]
point2 = [length, width, height]

margin = (3 * ureg.centimeter).to_base_units().magnitude
z_rebar = (5*ureg.centimeter).to_base_units().magnitude


def rebar_problem(n):
    nx = 0
    ny = n
    h = (2 * ureg.centimeter).to_base_units().magnitude
    msh_filename = "test_mesh.msh"
    xdmf_filenames = ["concrete_mesh.xdmf", "rebar_mesh.xdmf"]


    create_concrete_slab(
        point1, point2, nx, ny, margin, h, msh_filename, xdmf_filenames, z=[z_rebar]
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
        lambda x: np.isclose(x[2], height, atol=h)
    )
    facet_markers = np.full(len(facet_indices), marker).astype(np.int32)
    facet_tag = dfx.mesh.meshtags(concrete_mesh, fdim, facet_indices, facet_markers)

    concrete_mesh.topology.create_connectivity(
        concrete_mesh.topology.dim - 1, concrete_mesh.topology.dim
    )
    ds = ufl.Measure("ds", domain=concrete_mesh, subdomain_data=facet_tag)


    external_force_form = - pressure * ufl.dot(ufl.FacetNormal(concrete_mesh), v_) * ds(42)
    internal_force_form = ufl.inner(eps(v_), sigma(u, parameters_concrete)) * ufl.dx
    residual =   -(external_force_form-internal_force_form)

    # right side
    def left(x):
        return np.logical_and(np.isclose(x[0], 0.), np.isclose(x[2], 0.0))


    def right(x):
        return np.logical_and(np.isclose(x[0], length), np.isclose(x[2], 0.0))


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

    problem = NonlinearReinforcementProblem(residual, a, u, rebar, bcs)

    return problem, concrete_mesh, u

class DisplacementAtDofSensor:
    
    def __init__(self, u, lambda_function):
        self.u = u
        nodes = dfx.fem.locate_dofs_geometrical(u.function_space, lambda_function)
        self.dofs = np.arange(3) + 3 * nodes.reshape(-1,1)
        self.x = u.function_space.mesh.geometry.x[nodes]
    
    def __call__(self):
        return self.u.vector.array[self.dofs]

def test_rebar():
    for i in range(2,11):
        print(i)
        problem, mesh, u = rebar_problem(i)
        sensor = DisplacementAtDofSensor(u, lambda x : np.logical_and(np.isclose(x[1],0.15),np.isclose(x[2],0.)))
        solver = dfx.nls.petsc.NewtonSolver(mesh.comm, problem)
        # Set Newton solver options
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.convergence_criterion = "residual"
        _ = solver.solve(u)

        parameters_steel["amount"] = i 
        solution_analytical = analytical_solution(force/length, length, height, width, z_rebar, parameters_steel, parameters_concrete)

        solution_fem = sensor()
        solution_ana = solution_analytical.evaluate(sensor.x[:,0])
        max_error= np.max(np.abs(-solution_fem[:,2]-solution_ana))

        max_rel_error = max_error/np.max(np.abs(solution_ana))
        assert max_rel_error < 1e-2

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg", force=True)
    print(__name__)
    problem, mesh, u = rebar_problem(2)
    solution_analytical = analytical_solution(force/length, length, height, width, z_rebar, parameters_steel, parameters_concrete)
    solver = dfx.nls.petsc.NewtonSolver(mesh.comm, problem)
    sensor = DisplacementAtDofSensor(u, lambda x : np.logical_and(np.isclose(x[1],0.15),np.isclose(x[2],0.)))
    
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "residual"
    solver.report = True
    dfx.log.set_log_level(dfx.log.LogLevel.INFO)
    _ = solver.solve(u)
    solution_fem = sensor()
    plt.plot(sensor.x[:,0],-solution_fem[:,2], label="FEM")
    plt.plot(sensor.x[:,0],solution_analytical.evaluate(sensor.x[:,0]), label ="Analytical")
    plt.legend()
    #plt.yscale("log")
    plt.savefig("fem_solution.png")

    with dfx.io.XDMFFile(mesh.comm, "displacements.xdmf", "w") as f:
        f.write_mesh(mesh)
        f.write_function(u)
