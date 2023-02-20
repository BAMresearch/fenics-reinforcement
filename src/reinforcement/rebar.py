from abc import ABC, ABCMeta, abstractmethod
import dolfinx as dfx
import numpy as np
from petsc4py import PETSc
import sys

np.set_printoptions(threshold=sys.maxsize)


class RebarInterface(ABC):
    def __init__(self, concrete_mesh, rebar_mesh, function_space, parameters: dict):
        self.concrete_mesh = concrete_mesh
        self.rebar_mesh = rebar_mesh
        self.function_space = function_space
        self.parameters = parameters
        self.dof_array = np.array([], dtype=np.int32)
        self._assign_dofs()

    def _assign_dofs(self, tol=1e-6):
        # first all reinforcement dofs and their coordinates are found and saved in geometry_entities and points respectively
        fdim = self.rebar_mesh.topology.dim
        self.rebar_mesh.topology.create_connectivity(fdim, 0)
        num_lines_local = self.rebar_mesh.topology.index_map(fdim).size_local
        geometry_entities = dfx.cpp.mesh.entities_to_geometry(
            self.rebar_mesh, fdim, np.arange(num_lines_local, dtype=np.int32), False
        )
        dofs = []
        for line in geometry_entities:
            start = self.rebar_mesh.geometry.x[line][0]
            end = self.rebar_mesh.geometry.x[line][1]

            dofs_start = self._locate_concrete_dofs(start, tol)
            dofs_end = self._locate_concrete_dofs(end, tol)

            dofs.extend(dofs_start)
            dofs.extend(dofs_end)
        self.dof_array = np.array(dofs, dtype=np.int32).reshape(-1, 3)

    def _locate_concrete_dofs(self, point, tol):
        x, y, z = point

        def rebar_nodes(var):
            return np.logical_and(
                np.logical_and(np.abs(var[1] - y) < tol, np.abs(var[0] - x) < tol),
                np.abs(var[2] - z) < tol,
            )

        dofs_toappend = dfx.fem.locate_dofs_geometrical(
            self.function_space, rebar_nodes
        )
        try:
            assert len(dofs_toappend) == 1
        except AssertionError:
            raise Exception(
                f"{len(dofs_toappend)} dofs found at ({x},{y},{z}), expected 1. Try adjusting the tolerance"
            )

        return dofs_toappend * 3.0 + np.arange(3, dtype=np.float64)

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

    Remark
    ------
    The implementation is not very efficient
    """

    def apply_to_diagonal_mass(self, M):
        """
        This method will assume a diagonal Mass matrix. The added entries will be calculated according
        to Gauß-Lobatto Quadrature
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

    def apply_to_stiffness(self, K, u):
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

    def apply_to_forces(self, f_int, u, sign=1.0):
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
