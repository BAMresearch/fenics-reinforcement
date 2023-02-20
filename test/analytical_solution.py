#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:32:23 2023

@author: mcappone
"""
import numpy as np

class analytical_solution:
    def __init__(self,q0,B,H,L,zr,params_rebar,params_concrete):
        '''
        Parameters
        ----------
        q0: float  
            Distributed force (applied to the top of the slab over the whole width).
        B: float
           Width of the cuboid.
        H: float
           Height of the cuboid.
        L: float
           Length of the cuboid.
        zr: float 
            Distance from z=0 to the rebars.
        params_rebar: Dict
            Contains the following information on the reinforcement bars:
            "Youngs_Modulus_steel": float, "density_steel": float, "amount": int - amount of rebars, "area":float - area of one rebar.
        params_concrete: Dict
            Contains the following information of the concrete:
            "Youngs_Modulus_concrete": float, "density_concrete": float

        Returns
        -------

        '''
        E_s = params_rebar["E"]
        rho_s = params_rebar["rho"]
        i = params_rebar["amount"]
        A_r = params_rebar["A"]
        E_c = params_concrete["E"]
        rho_c = params_concrete["rho"]
        V_c = B*H*L
        V_s = i*A_r*L
        zg = (rho_c*V_c*H/2 + rho_s*V_s*zr) / (rho_c*V_c + rho_s*V_s)
        I_c = B*H**3/12 + (H/2 - zg)**2*B*H
        I_s = i*A_r*(A_r/(4*np.pi) + (zr-zg)**2)
        self.EI = E_c*I_c + E_s*I_s
        self.q0 = q0
        self.B = B
        
        
    
    def evaluate(self,y0):
        '''
        Parameters
        ----------
        y0 : float
            position at which displacement is to be evaluated.

        Returns
        -------
        w_y : float
            displacement in vertical direction at horizontal position y0.

        '''
        w_y = 1/(24*self.EI)*self.q0*(y0**4 -2*self.B*y0**3+self.B**3*y0)
        return w_y



