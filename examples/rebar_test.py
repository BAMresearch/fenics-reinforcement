#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:59:55 2023

@author: mcappone
"""
import dolfinx as dfx
from mpi4py import MPI
from reinforcement import RebarInterface,ElasticTrussRebar,read_xdmf

concrete_mesh, rebar_mesh = read_xdmf(["concrete_mesh.xdmf","rebar_mesh.xdmf"])
V1 = dfx.fem.VectorFunctionSpace(
    concrete_mesh, ("Lagrange", 1)
)  

parameters = {
  "E": 10.,
  "A": 0.02,
}


e = ElasticTrussRebar(concrete_mesh, rebar_mesh, V1, parameters)
e._assign_dofs(1e-4)
