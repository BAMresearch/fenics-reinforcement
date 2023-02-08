from abc import ABC, ABCMeta, abstractmethod
import dolfinx as dfx
import numpy as np
from petsc4py import PETSc
import sys

np.set_printoptions(threshold=sys.maxsize)

class RebarInterface(ABC):
    def __init__(self, concrete_mesh, rebar_mesh, parameters: dict):
        self.concrete_mesh = concrete_mesh
        self.rebar_mesh = rebar_mesh
        self.parameters = parameters
        self.dofs = []
    
    #@abstractmethod
    def apply_to_forces(self, f_int, u):
        pass
    
    #@abstractmethod
    def apply_to_stiffness(self, K, u):
        pass
    
class elastic_truss_rebar(RebarInterface):
    def __init__(self, concrete_mesh, rebar_mesh, parameters):
        super().__init__(concrete_mesh, rebar_mesh, parameters)
        
    def apply_to_forces(self, f_int, u):
        print("sad")
        
    
    def apply_to_sitffness(self, K, u):
        print("fnsjf")
        
    def assign_dofs(self):
        
        # first all reinforcement dofs and their coordinates are found and saved in geometry_entities and points respectively
        fdim = self.rebar_mesh.topology.dim
        self.rebar_mesh.topology.create_connectivity(fdim, 0)
        num_facets_owned_by_proc = self.rebar_mesh.topology.index_map(fdim).size_local
        geometry_entities = dfx.cpp.mesh.entities_to_geometry(self.rebar_mesh, fdim, np.arange(num_facets_owned_by_proc, dtype=np.int32), False)
        points = self.rebar_mesh.geometry.x
        
        #print(points)
        print("geometry entitites", geometry_entities)
        # loop through every single reinforcement line
        for entity in geometry_entities:
            start = points[entity][0]
            end = points[entity][1]
            y_values = np.array([start[1],end[1]]) 
            x_values = np.array([start[0],end[0]])  
            z_values = np.array([start[2],end[2]])
            
        
            
            for (x,y,z) in zip(x_values,y_values,z_values):  
                

                
                
                def rebar_nodes(var):
                    
                    # find which nodes correspond to the pair of (x,y) values 
                    tol = 1e-6
                    #print("var_x",var[0])
                    #print("x",x)
                    logical = np.logical_and(np.logical_and(np.abs(var[1]-y) < tol, np.abs(var[0]-x)<tol), np.abs(var[2]-z<tol))
                    return logical
                
                #print(dfx.fem.locate_dofs_geometrical(self.parameters["VectorFunctionSpace_concrete"], rebar_nodes))
                
                # save all dofs that need to be reinforced here
                dofs_toappend = dfx.fem.locate_dofs_geometrical(self.parameters["VectorFunctionSpace_concrete"], rebar_nodes)[0]
                #print(dofs_toappend)
                
                for k in range(3): # TODO we had k=[0,1] for 2D, is it correct to go to [0,1,2] for 3D?=
                    self.dofs.append(dofs_toappend*3+k) # TODO we had *2 for 2D, is it correct to go *3 for 3D?
                    
                    
                
                    
