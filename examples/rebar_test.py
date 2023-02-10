#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:59:55 2023

@author: mcappone
"""
import dolfinx as dfx
from mpi4py import MPI
from reinforcement import RebarInterface,elastic_truss_rebar,read_xdmf






concrete_mesh, rebar_mesh = read_xdmf(["concrete_mesh.xdmf","rebar_mesh.xdmf"])
V1 = dfx.fem.VectorFunctionSpace(
    concrete_mesh, ("Lagrange", 1)
)  

parameters = {
  "E": "10",
  "VectorFunctionSpace_concrete": V1
}


e = elastic_truss_rebar(concrete_mesh, rebar_mesh, parameters)
e.assign_dofs()
