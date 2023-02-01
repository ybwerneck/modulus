from modulus.models.fully_connected import FullyConnectedArch

from sympy import Symbol, Eq
from sympy import Symbol, Function, Number
from modulus.eq.pde import PDE
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_1d import Point1D
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseBoundaryConstraint,
)
from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
from modulus.eq.pde import PDE
from modulus.geometry import Parameterization
a=0.001
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 20:30:47 2023

@author: yanbw
"""

import numpy as np
import matplotlib.pyplot as plt
a=0.8
b=0.8
thau=12
def generateExactSolution(t,dt,x0):
    
    n=int(t/dt)
    Sol=np.zeros((n,2))
    Sol[0]=x0
    T=0
    k=0
    while(k<n-1):
        x,w=Sol[k]
    
        Sol[k+1][0]=(x*(x-0.2)*(1-x)-w)*dt +  x
        ##Sol[k+1][0]=(x - (x**3)/3)*dt +  x
        Sol[k+1][1]=0.05*(x*0.2-0.8*w)*dt +w
        T=T+dt
        k=k+1
    return Sol


class SpringMass(PDE):
    name = "FHN"

    def __init__(self):

      
     

        t = Symbol("t")
        input_variables = {"t": t}

        x1 = Function("x1")(*input_variables)
        w = Function("w")(*input_variables)


        self.equations = {}
        self.equations["ode_x1"] = x1*(x1-0.2)*(1-x1) -w -x1.diff(t)
        self.equations["ode_w"] = 0.05*(x1*0.2-0.8*w)-w.diff(t)
                                    
                                    
        
@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    t_max,u0=10,[0.3,0]
    # make list of nodes to unroll graph on
    sm = SpringMass()
    sm.pprint()
    sm_net = FullyConnectedArch(
        input_keys=[Key("t")],
        output_keys=[Key("x1"),Key("w")],
    )
    nodes = sm.make_nodes() + [
        sm_net.make_node(name="network")
    ]

   
    # add constraints to solver


    t_symbol = Symbol("t")
    u_symbol = Symbol("x1")
    w_symbol = Symbol("w")
    k_symbol= Symbol("K")
    time_range = {t_symbol: (0, t_max)}
    # make geometry
    geo = Point1D(0)
    # make domain
    domain = Domain()

    # initial conditions
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"x1": u0[0],"w":u0[1]},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={
            "x1": 1.0,
            "w": 1.0


        },
        parameterization={t_symbol: 0},
    )
    domain.add_constraint(IC, name="IC")

    
    
    # solve over given time period
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_x1": 0.0,"ode_w":0.0},
        batch_size=cfg.batch_size.interior,
        parameterization={**time_range},
    )
    domain.add_constraint(interior, "interior")
    
 
    domain.add_constraint(interior, "interior")
    # add validation data
  
 

    T=np.empty([0])
    K=np.empty([0])
    SOLs=np.empty([0])
    SOLw=np.empty([0])
    krange=[1]
    
    deltaT = 0.01
    
    
    for KR in krange:
        t = np.arange(0, t_max, deltaT)
        t = np.expand_dims(t, axis=-1)
        T=np.append(T,t)
        K = np.append(K,np.full_like (t,KR))
        R=generateExactSolution(t_max,deltaT,u0)
        SOLs=np.append(SOLs,np.array(R.T[0]))
        SOLw=np.append(SOLw,np.array(R.T[1]))
    

    
    
    
    
    
    
    
    t=np.expand_dims(T,axis=-1)
    k=np.expand_dims(K,axis=-1)    

    sol=SOLs
    solw=SOLw
    invar_numpy = {"t": t,}
    outvar_numpy = {
        "x1": sol,
        "w":solw
    }
    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=2048
    )
    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
