import torch
import numpy as np

import modulus
from modulus.models.deeponet import DeepONetArch
from modulus.domain.constraint.continuous import DeepONetConstraint


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
from modulus.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
from modulus.dataset import *
from modulus.domain.constraint import SupervisedGridConstraint
HYDRA_FULL_ERROR=1

a=0.001
def generateExactSolution(t,dt,x0,rate):
    
    n2=int(t/dt)
    n = int(t/(dt*rate))
    Sol=np.zeros(n)
    Sol[0]=x0
    Sol2=np.zeros(n2)
    Sol2[0]=x0
    T=0
    k=0
    while(k<n2-1):
        x=Sol2[k]
        Sol2[k+1]=x*(x-0.5)*(1-x)*dt +  x
        if ((k+1)%rate == 0):
            T=T+1
            Sol[T] = Sol2[k+1]         
        k=k+1
    return Sol


class SpringMass(PDE):
    name = "SpringMass"

    def __init__(self):

      
     

        t = Symbol("t")
        input_variables = {"t": t}

        x1 = Function("x1")(*input_variables)



        self.equations = {}
        self.equations["ode_x1"] = x1*(1-x1)*(x1-0.5) -x1.diff(t)
                                    
                                    
        
@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    x0=0.6
    c1=1/x0 -1
    t_max = 10.0
    
     
    t_symbol = Symbol("t")
    x = Symbol("x")
   
    
    # make list of nodes to unroll graph on
    sm = SpringMass()
    sm_net = instantiate_arch(
        input_keys=[Key("t",scale=(0,t_max))],
        output_keys=[Key("x1",scale=(0,1))],
        cfg=cfg.arch.fully_connected,
    )
    nodes = sm.make_nodes() + [
        sm_net.make_node(name="spring_mass_network")
    ]

   
    # add constraints to solver
    # make geometry
    geo = Point1D(0)
    print(geo)
  
    time_range = {t_symbol: (0, t_max)}

    # make domain
    domain = Domain()

    # initial conditions
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"x1": x0},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={
            "x1": 1.0,


        },
        parameterization={t_symbol: 0},
    )
   # domain.add_constraint(IC, name="IC")

    
    
    # solve over given time period
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_x1": 0.0},
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
        
    )
   # domain.add_constraint(interior, "interior")

    
    ##Data driven constraing
    rate = 100
    deltaT = 0.01
    t = np.linspace(0, t_max, int((t_max/(deltaT)) /rate) )
    t = np.expand_dims(t, axis=-1)
    t = np.expand_dims(t, axis=-1)
    t = np.expand_dims(t, axis=-1)
    print(np.shape(t))
    Sol=generateExactSolution(t_max,deltaT,x0,rate)
    Sol = np.expand_dims(Sol, axis=-1)
    Sol = np.expand_dims(Sol, axis=-1)
    Sol = np.expand_dims(Sol, axis=-1)
    print(np.shape(t),"training set")
    invar_numpy = {"t": t}
    outvar_numpy = {
        "x1": Sol
    }
    trainset=DictGridDataset(invar_numpy,outvar_numpy)
    
    data = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={"t":t},
        outvar={"x1":Sol},
        batch_size=10,
    )
    domain.add_constraint(data, "data")
    
    
    
    
    
    tmax=10
    # add validation data
    rate = 10
    deltaT = 0.01
    t = np.linspace(0, t_max, int((t_max/(deltaT)) /rate) )
    t = np.expand_dims(t, axis=-1)
    Sol=generateExactSolution(t_max,deltaT,x0,rate)
    invar_numpy = {"t": t}
    plt.plot(Sol,"o")
    plt.show()
    print(np.shape(t),"val set")


    outvar_numpy = {
        "x1": Sol
    }
    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=124,
    )
    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
