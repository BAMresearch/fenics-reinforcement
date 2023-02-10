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
        self.dofs = []
        #self._assign_dofs()
    
    def _assign_dofs(self, tol = 1e-6):
        # first all reinforcement dofs and their coordinates are found and saved in geometry_entities and points respectively
        fdim = self.rebar_mesh.topology.dim
        self.rebar_mesh.topology.create_connectivity(fdim, 0)
        num_lines_local = self.rebar_mesh.topology.index_map(fdim).size_local
        geometry_entities = dfx.cpp.mesh.entities_to_geometry(self.rebar_mesh, fdim, np.arange(num_lines_local, dtype=np.int32), False)
        
        for line in geometry_entities:
            start = self.rebar_mesh.geometry.x[line][0]
            end = self.rebar_mesh.geometry.x[line][1]
            
            dofs_start = self._locate_concrete_dofs(start, tol)
            dofs_end = self._locate_concrete_dofs(end, tol)

            self.dofs.extend(dofs_start)
            self.dofs.extend(dofs_end)

    def _locate_concrete_dofs(self, point, tol):
        x,y,z=point
        def rebar_nodes(var):
            return np.logical_and(np.logical_and(np.abs(var[1]-y) < tol, np.abs(var[0]-x)<tol), np.abs(var[2]-z)<tol)
        dofs_toappend = dfx.fem.locate_dofs_geometrical(self.function_space, rebar_nodes)
        try:
            assert len(dofs_toappend) == 1
        except AssertionError:
            raise Exception(f"{len(dofs_toappend)} dofs found at ({x},{y},{z}), expected 1. Try adjusting the tolerance")
        
        return dofs_toappend * 3. + np.arange(3, dtype=np.float64)
    
    @abstractmethod
    def apply_to_forces(self, f_int, u):
        pass
    
    @abstractmethod
    def apply_to_stiffness(self, K, u):
        pass
    
class ElasticTrussRebar(RebarInterface):
        
    def apply_to_forces(self, f_int, u):
        print("sad")
        
    def apply_to_stiffness(self, K, u):
        print("fnsjf")
    