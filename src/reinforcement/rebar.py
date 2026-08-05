from abc import ABC, abstractmethod

import dolfinx as dfx
import dolfinx.fem.petsc
import numpy as np
import ufl
from petsc4py import PETSc

from .maps import build_vertex_subspace_map


class RebarInterface(ABC):
    """
    An interface for trusses. Builds the rebar function space and the dof
    map between the rebar submesh and the concrete mesh.
    """
    def __init__(
        self,
        concrete_mesh : dfx.mesh.Mesh,
        rebar_mesh : dfx.mesh.Mesh,
        function_space : dfx.fem.FunctionSpace,
        parameters: dict,
        vertex_map: np.ndarray,
    ):
        """Initialize rebar class

        Args:
            concrete_mesh: The 3D mesh of the concrete structure.
            rebar_mesh: The line mesh of the steel rebar, built as a submesh
                of concrete_mesh's edges (see reinforcement.mesh.read_msh).
            function_space: The function space object of the concrete structure.
            parameters: A dictinary containing all needed parameters of the steel. Contains: 'A', 'E', 'rho'.
            vertex_map: rebar_mesh's vertex map, as returned by read_msh.

        """
        self.concrete_mesh = concrete_mesh
        self.rebar_mesh = rebar_mesh
        self.function_space = function_space
        self.parameters = parameters

        self.rebar_function_space = dfx.fem.functionspace(rebar_mesh, ("Lagrange", 1, (3,)))
        self.space_map = build_vertex_subspace_map(
            vertex_map, function_space, self.rebar_function_space
        )
        self.parent_dofs = _block_expand_to_rebar_dof_order(
            self.space_map, function_space.dofmap.index_map_bs
        )

    @abstractmethod
    def apply_to_forces(self, f_int, u):
        pass

    @abstractmethod
    def apply_to_stiffness(self, K, u):
        pass


def _block_expand_to_rebar_dof_order(space_map, block_size: int) -> np.ndarray:
    """
    Build the array of concrete (blocked) dof indices corresponding, in
    order, to every blocked dof of the rebar function space -- i.e. entry k
    is the concrete dof that rebar dof k should be added into.

    space_map.sub[vertex] gives the rebar dof for a given rebar_mesh vertex,
    so its inverse gives, for each rebar dof, the vertex it belongs to; from
    there space_map.parent gives the corresponding concrete dof.
    """
    num_dofs = len(space_map.sub)
    vertex_of_rebar_dof = np.empty(num_dofs, dtype=np.int32)
    vertex_of_rebar_dof[space_map.sub] = np.arange(num_dofs, dtype=np.int32)
    parent_dof_of_rebar_dof = space_map.parent[vertex_of_rebar_dof]

    return (
        block_size * parent_dof_of_rebar_dof[:, None] + np.arange(block_size)[None, :]
    ).ravel().astype(np.int32)


class ElasticTrussRebar(RebarInterface):
    """
    This class can insert purely elastic rebar stiffnesses into the concrete matrix and the internal forces vector.

    The truss's axial strain/stress are expressed as a UFL weak form on the
    rebar submesh (a 1D mesh embedded in 3D): the axial strain is the
    directional derivative of the displacement along the cell's own tangent
    direction, obtained from the mesh's Jacobian. This is evaluated (and its
    stiffness assembled) on the small rebar-local function space, then
    scattered into the caller's global concrete matrix/vector via the dof
    map from RebarInterface.

    """
    def __init__(
        self,
        concrete_mesh : dfx.mesh.Mesh,
        rebar_mesh : dfx.mesh.Mesh,
        function_space : dfx.fem.FunctionSpace,
        parameters: dict,
        vertex_map: np.ndarray,
    ):
        """Initialize rebar class

        Args:
            concrete_mesh: The 3D mesh of the concrete structure.
            rebar_mesh: The line mesh of the steel rebar, built as a submesh
                of concrete_mesh's edges (see reinforcement.mesh.read_msh).
            function_space: The function space object of the concrete structure.
            parameters: A dictinary containing all needed parameters of the steel. Contains: 'A', 'E', 'rho'.
            vertex_map: rebar_mesh's vertex map, as returned by read_msh.

        """
        super().__init__(concrete_mesh, rebar_mesh, function_space, parameters, vertex_map)

        V = self.rebar_function_space
        u_ = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        # explicit domain: a zero coefficient (e.g. rho defaulting to 0)
        # collapses the integrand to a symbolic zero, which would otherwise
        # leave ufl.dx unable to infer the integration domain from it.
        dx = ufl.Measure("dx", domain=rebar_mesh)

        tangent = ufl.Jacobian(rebar_mesh)[:, 0]
        tangent = tangent / ufl.sqrt(ufl.dot(tangent, tangent))

        def axial_strain(w):
            return ufl.dot(tangent, ufl.dot(ufl.grad(w), tangent))

        E = parameters["E"]
        A = parameters["A"]

        self._u_concrete = dfx.fem.Function(function_space)
        self._u_rebar = dfx.fem.Function(V)

        self._stiffness_form = dfx.fem.form(E * A * axial_strain(u_) * axial_strain(v) * dx)
        self._force_form = dfx.fem.form(
            E * A * axial_strain(self._u_rebar) * axial_strain(v) * dx
        )
        self._ones = dfx.fem.Constant(rebar_mesh, PETSc.ScalarType((1.0, 1.0, 1.0)))
        self._lumped_mass_form = dfx.fem.form(
            parameters.get("rho", 0.0) * A * ufl.inner(self._ones, v) * dx
        )

    def apply_to_diagonal_mass(self, M : PETSc.Vec):
        """
        Adds nodal masses to a diagonal mass matrix.

        Args:
            M: The diagonal from of the mass matrix as a PETSc vector

        """
        # row-sum ("HRZ") lumping: for a partition-of-unity basis, the row
        # sum of the consistent mass matrix equals inner(1, v) integrated
        # against the same weighting, avoiding an explicit matrix assembly.
        m_local = dfx.fem.petsc.assemble_vector(self._lumped_mass_form)
        M.setValues(self.parent_dofs, m_local.array, addv=PETSc.InsertMode.ADD)
        M.assemble()

    def apply_to_stiffness(self, K : PETSc.Mat, u : PETSc.Vec):
        """
        Adds truss stiffness to global stiffness matrix,

        Args:
            K: The global stiffness matrix.
            u: The global displacements.
        """
        K_local = dfx.fem.petsc.assemble_matrix(self._stiffness_form)
        K_local.assemble()

        # scatter only the local matrix's actual (sparse) nonzero structure:
        # densifying and inserting the full block would submit mostly
        # explicit zeros (rebar dofs are only weakly connected through a
        # handful of neighbouring segments), which bloats the global
        # matrix's sparsity pattern and makes the solve far more expensive.
        K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        num_local_rows = K_local.getSize()[0]
        for row in range(num_local_rows):
            cols, vals = K_local.getRow(row)
            if len(cols) == 0:
                continue
            K.setValues(
                [self.parent_dofs[row]], self.parent_dofs[cols], vals, addv=PETSc.InsertMode.ADD
            )
        K.assemble()

    def apply_to_forces(self, f_int : PETSc.Vec, u : PETSc.Vec, sign : float = 1.0):
        """
        Adds truss nodal internal forces to global forces vector.

        Args:
            f_int: The global forces vector.
            u: The global displacements.
            sign: Sign of the added values.
        """
        assert sign in [1.0, -1.0]
        self._u_concrete.x.array[:] = u.array
        self.space_map.map_to_sub(self._u_concrete, self._u_rebar)

        f_local = dfx.fem.petsc.assemble_vector(self._force_form)
        f_int.setValues(self.parent_dofs, sign * f_local.array, addv=PETSc.InsertMode.ADD)
        f_int.assemble()
