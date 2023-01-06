from abc import ABCMeta, abstractmethod

class RebarInterface(metaclass=ABCMeta):
    def __init__(self, concrete_mesh, rebar_mesh, parameters: dict):
        self.concrete_mesh = concrete_mesh
        self.rebar_mesh = rebar_mesh
        self.parameters = parameters
    
    @abstractmethod
    def apply_to_forces(self, f_int, u):
        pass
    
    @abstractmethod
    def apply_to_stiffness(self, K, u):
        pass
    
