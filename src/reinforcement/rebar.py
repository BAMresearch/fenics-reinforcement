class RebarInterface:
    def __init__(self, concrete_mesh, rebar_mesh, parameters: dict):
        self.concrete_mesh = concrete_mesh
        self.rebar_mesh = rebar_mesh
        self.parameters = parameters
    
    def apply_to_forces(self, f_int, u):
        raise NotImplementedError("apply_to_forces not implemented yet")

    def apply_to_stiffness(self, K, u):
        raise NotImplementedError("apply_to_stiffness not implemented yet")
    
