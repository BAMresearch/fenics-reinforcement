from abc import ABC, abstractmethod

import dolfinx as dfx
import numpy as np
from petsc4py import PETSc

from .maps import build_vertex_subspace_map


class RebarInterface(ABC):
    """
    An interface for trusses. Contains methods to assign dofs.
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
        self.dof_array = np.array([], dtype=np.int32)
        self._assign_dofs(vertex_map)

    def _assign_dofs(self, vertex_map: np.ndarray):
        rebar_function_space = dfx.fem.functionspace(self.rebar_mesh, ("Lagrange", 1, (3,)))
        space_map = build_vertex_subspace_map(vertex_map, self.function_space, rebar_function_space)

        fdim = self.rebar_mesh.topology.dim
        self.rebar_mesh.topology.create_connectivity(fdim, 0)
        cell_to_vertex = self.rebar_mesh.topology.connectivity(fdim, 0)
        num_lines_local = self.rebar_mesh.topology.index_map(fdim).size_local

        block_size = self.function_space.dofmap.index_map_bs
        dofs = []
        for cell in range(num_lines_local):
            for rebar_vertex in cell_to_vertex.links(cell):
                concrete_dof = space_map.parent[rebar_vertex]
                dofs.extend(block_size * concrete_dof + np.arange(block_size))
        self.dof_array = np.array(dofs, dtype=np.int32).reshape(-1, 3)

    @abstractmethod
    def apply_to_forces(self, f_int, u):
        pass

    @abstractmethod
    def apply_to_stiffness(self, K, u):
        pass


class ElasticTrussRebar(RebarInterface):
    """
    This class can insert purely elastic rebar stiffnesses into the concrete matrix and the internal forces vector.
    Equations from http://what-when-how.com/the-finite-element-method/fem-for-trusses-finite-element-method-part-1/

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

    def apply_to_diagonal_mass(self, M : PETSc.Vec):
        """
        Adds nodal masses to a diagonal mass matrix.
        
        Args:
            M: The diagonal from of the mass matrix as a PETSc vector 

        """
        points = self.function_space.tabulate_dof_coordinates().flatten()
        diagonal_mass_1d = np.array([[1.0, 0.0], [0.0, 1.0]])
        T = np.zeros((2, 6))
        for dofs in self.dof_array.reshape(-1, 6):
            delta_x = points[dofs[3:]] - points[dofs[:3]]
            l_axial = np.linalg.norm(delta_x, 2)
            matrix_entries = delta_x / l_axial

            T[0, :3] = matrix_entries
            T[1, 3:] = matrix_entries

            mass_local = np.diag(
                T.T
                @ (
                    self.parameters["rho"]
                    * self.parameters["A"]
                    * l_axial
                    * diagonal_mass_1d
                )
                @ T
            )

            M.setValues(dofs, mass_local, addv=PETSc.InsertMode.ADD)
            M.assemble()

    def apply_to_stiffness(self, K : PETSc.Mat, u : PETSc.Vec):
        """
        Adds truss stiffness to global stiffness matrix,
        
        Args:
            K: The global stiffness matrix.
            u: The global displacements.
        """
        points = self.function_space.tabulate_dof_coordinates().flatten()
        K_1d = np.array([[1.0, -1.0], [-1.0, 1.0]])
        T = np.zeros((2, 6))
        for dofs in self.dof_array.reshape(-1, 6):
            delta_x = points[dofs[3:]] - points[dofs[:3]]
            l_axial = np.linalg.norm(delta_x, 2)

            matrix_entries = delta_x / l_axial
            T[0, :3] = matrix_entries
            T[1, 3:] = matrix_entries
            AEL = self.parameters["A"] * self.parameters["E"] / l_axial
            K_local = T.T @ (AEL * K_1d) @ T

            K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            K.setValues(dofs, dofs, K_local.flat, addv=PETSc.InsertMode.ADD)
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
        points = self.function_space.tabulate_dof_coordinates().flatten()
        f_1d = np.array([-1.0, 1.0])
        T = np.zeros((2, 6))
        for dofs in self.dof_array.reshape(-1, 6):
            delta_x = points[dofs[3:]] - points[dofs[:3]]
            delta_u = u.array[dofs[3:]] - u.array[dofs[:3]]
            l_axial = np.linalg.norm(delta_x, 2)
            matrix_entries = delta_x / l_axial

            u_axial = np.inner(matrix_entries, delta_u)

            eps_axial = u_axial / l_axial

            T[0, :3] = matrix_entries
            T[1, 3:] = matrix_entries

            sigma_axial = self.parameters["E"] * eps_axial

            f_local = sign * T.T @ (self.parameters["A"] * sigma_axial * f_1d)

            f_int.setValues(dofs, f_local, addv=PETSc.InsertMode.ADD)
            f_int.assemble()
