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
from modulus.models.deeponet import DeepONetArch
from modulus.domain.constraint.continuous import DeepONetConstraint


a=0.001


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
        Sol2[k+1]=P*x*(x)*(1-x)*dt +  x
        if ((k+1)%rate == 0):
            T=T+1
            Sol[T] = Sol2[k+1]         
        k=k+1
    return Sol

class SpringMass(PDE):
    name = "FHN"

    def __init__(self):

      
     

        t,k = Symbol("t"),Symbol("K")
        input_variables = {"t": t,"K":k}

        x1 = Function("x1")(*input_variables)



        self.equations = {}
        self.equations["ode_x1"] = k*x1*(1-x1)*(x1) -x1.diff(t)

                                    
                                    
        
@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    u0=0.7
    # make list of nodes to unroll graph on
    sm = SpringMass()
    sm.pprint()
    sm_net = FullyConnectedArch(
        input_keys=[Key("t"), Key("K")],
        output_keys=[Key("x1")],
    )
    nodes = sm.make_nodes() + [
        sm_net.make_node(name="network")
    ]

   
    # add constraints to solver
    # make geometry
    geo = Point1D(0)
    t_max = 10.0
    t_symbol = Symbol("t")
    u_symbol = Symbol("x1")
    k_symbol= Symbol("K")
    time_range = {t_symbol: (0, t_max)}

    # make domain
    domain = Domain()

    # initial conditions
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"x1": u0},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={
            "x1": 1.0,


        },
        parameterization={t_symbol: 0,**{k_symbol: (0.1, 0.4)}},
    )
    domain.add_constraint(IC, name="IC")

    
    
    # solve over given time period
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_x1": 0.0},
        batch_size=cfg.batch_size.interior,
        parameterization={**time_range,**{k_symbol: (0.1, 0.4)}},
        lambda_weighting={
            "ode_x1": 10.0,


        },
    )
    domain.add_constraint(interior, "interior")
    
    ##Data driven constraing
    T=np.empty([0])
    K=np.empty([0])
    SOLs=np.empty([0])
    krange= [(0.07 + 0.097*i*0.3*10) for i in range(1,10)]

    deltaT = 0.01
    rate = 10
    t = np.linspace(0, t_max, int((t_max/(deltaT)) /rate) )
    t = np.expand_dims(t, axis=-1)
    
    for KR in krange:
        
        
        T=np.append(T,t)
        K = np.append(K,np.full_like (t,KR))
        SOLs=np.append(SOLs,np.array(generateExactSolution(t_max,deltaT,u0,rate,KR)))
    
    
    
    t = np.expand_dims(T, axis=-1)
    t = np.expand_dims(t, axis=-1)
    t = np.expand_dims(t, axis=-1)

    k = np.expand_dims(K, axis=-1)
    k = np.expand_dims(k, axis=-1)
    k = np.expand_dims(k, axis=-1)
    
    Sol = np.expand_dims(SOLs, axis=-1)
    Sol = np.expand_dims(Sol, axis=-1)
    Sol = np.expand_dims(Sol, axis=-1)
    
    print(np.shape(t),"training set")
    
    
    invar_numpy = {"t": t,"k":k}
    outvar_numpy = {
        "x1": Sol
    }

    
    data = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={"t":t,"K":k},
        outvar={"x1":Sol},
        batch_size=100,
    )
    domain.add_constraint(data, "data")
    
    
    # add validation data
  
 
    T=np.empty([0])
    K=np.empty([0])
    SOLs=np.empty([0])
    krange= [(0.1 + 0.01*i*0.3) for i in range(1,100)]
    
    deltaT = 0.01
    rate = 10
    t = np.linspace(0, t_max, int((t_max/(deltaT)) /rate) )
    t = np.expand_dims(t, axis=-1)
    for KR in krange:
        
        
        T=np.append(T,t)
        K = np.append(K,np.full_like (t,KR))
        SOLs=np.append(SOLs,np.array(generateExactSolution(t_max,deltaT,u0,rate,KR)))
    
    
    t=np.expand_dims(T,axis=-1)
    k=np.expand_dims(K,axis=-1)
    sol=SOLs
    
    invar_numpy = {"t": t,"K":k}
    outvar_numpy = {
        "x1": sol
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
