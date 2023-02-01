import numpy as np
from sympy import Symbol, sin

import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_1d import Line1D
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.eq.pde import PDE
from modulus.models.fully_connected import FullyConnectedArch

from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
from sympy import Symbol, Function, Number
from modulus.eq.pde import PDE

import numpy as np
from sympy import Symbol, sin

import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_1d import Line1D
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node


def generateExactSolution(t,dt,x0,rate,P):
    
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
        Sol2[k+1]=x*(x-0.2)*(1-x)*dt +  x
        if ((k+1)%rate == 0):
            T=T+1
            Sol[T] = Sol2[k+1]         
        k=k+1
    return Sol
class WaveEquation1D(PDE):
   

    name = "WaveEquation1D"

    def __init__(self, c=1.0):
        
        
        t = Symbol("x")
        input_variables = {"x": t}

        x1 = Function("u")(*input_variables)

        self.equations = {}
        self.equations["ode"] = x1*(1-x1)*(x1-0.2) -x1.diff(t)
        

        



u0=0.6

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    we = WaveEquation1D(c=1.0)
    wave_net =FullyConnectedArch(
            input_keys=[Key("x"), Key("K")],
            output_keys=[Key("u")],
        ) 

    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network")]

    # add constraints to solver
    # make geometry
    x, k_symbol = Symbol("x"), Symbol("K")
    t_max=L = 10.0
    geo = Line1D(0, L)
    k_range = {k_symbol: (0, 1)}

    # make domain
    domain = Domain()

    # initial condition
   # IC = PointwiseInteriorConstraint(
   #     nodes=nodes,
    #    geometry=geo,
   #     outvar={"u": sin(x), "u__t": sin(x)},
    #    batch_size=cfg.batch_size.IC,
    #    lambda_weighting={"u": 1.0, "u__t": 1.0},
     #   parameterization={x: (0.0,0.2)},
    #)
   # domain.add_constraint(IC, "IC")

    # boundary condition
    
    #BC = PointwiseBoundaryConstraint(
    #    nodes=nodes,
    #    geometry=geo,
    #    outvar={"u": 0},
    #    batch_size=cfg.batch_size.BC,
    #    parameterization=time_range,
    #)
    #domain.add_constraint(BC, "BC")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=k_range,
    )
    domain.add_constraint(interior, "interior")

    T=np.empty([0])
    K=np.empty([0])
    SOLs=np.empty([0])
    krange= [(0.3 + 0.01*i*0.1) for i in range(1,100)]
    #krange2= [(0.4 + 0.01*i*0.1) for i in range(1,100)]
    #krange3=[(0.5 + 0.01*i*0.5) for i in range(1,100)]
    #krange= np.append(krange, krange2)
    #krange= np.append(krange,krange3)
    
    deltaT = 0.01
    rate = 100
    t = np.linspace(0, t_max, int((t_max/(deltaT)) /rate) )
    t = np.expand_dims(t, axis=-1)
    for KR in krange:
        
        
        T=np.append(T,t)
        K = np.append(K,np.full_like (t,KR))
        SOLs=np.append(SOLs,np.array(generateExactSolution(t_max,deltaT,KR,rate,KR)))
    
    
    t=np.expand_dims(T,axis=-1)
    k=np.expand_dims(K,axis=-1)
    sol=SOLs
    
    invar_numpy = {"x": t,"K":k}
    outvar_numpy = {
        "u": sol
    }
    
    
    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=1024
    )
    domain.add_validator(validator)
    
    
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()



if __name__ == "__main__":
    run()
