from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.fourier_net import FourierNetArch
from modulus.models.siren import SirenArch
from modulus.models.modified_fourier_net import ModifiedFourierNetArch
from modulus.models.dgm import DGMArch

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
    PointwiseBoundaryConstraint,
)
from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
from modulus.eq.pde import PDE
from modulus.geometry import Parameterization
from sympy import Symbol, Eq, Abs, tanh, Or, And

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


class SpringMass(PDE):
    name = "SpringMass"

    def __init__(self):

        t = Symbol("t")
        input_variables = {"t": t}

        x1 = Function("x1")(*input_variables)

        self.equations = {}
        self.equations["ode_x1"] = x1*(1-x1)*(x1-0.2) -x1.diff(t)
        
@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    
    # make list of nodes to unroll graph on
    sm = SpringMass()
    sm.pprint()
    #sm_net = FullyConnectedArch(
    #    input_keys=[Key("t"), Key("K")],
    #    output_keys=[Key("x1")],
    #)
    #nodes = sm.make_nodes() + [
    #    sm_net.make_node(name="network")
    #]

    
    if cfg.custom.arch == "FullyConnectedArch":
        flow_net = FullyConnectedArch(
            input_keys=[Key("t"), Key("K")],
            output_keys=[Key("x1")],
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "FourierNetArch":
        flow_net = FourierNetArch(
            input_keys=[Key("t"), Key("K")],
            output_keys=[Key("x1")],
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "SirenArch":
        flow_net = SirenArch(
            input_keys=[Key("t"), Key("K")],
            output_keys=[Key("x1")],
            #normalization={"x": (-2.5, 2.5), "y": (-2.5, 2.5), "z": (-2.5, 2.5)},
        )
    elif cfg.custom.arch == "ModifiedFourierNetArch":
        flow_net = ModifiedFourierNetArch(
            input_keys=[Key("t"), Key("K")],
            output_keys=[Key("x1")],
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "DGMArch":
        flow_net = DGMArch(
            input_keys=[Key("t"), Key("K")],
            output_keys=[Key("x1")],
            layer_size=128,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    else:
        sys.exit(
            "Network not configured for this script. Please include the network in the script"
        )
    
    
    
    nodes = sm.make_nodes() + [
        flow_net.make_node(name="network")
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
        outvar={"x1": k_symbol},
        batch_size=cfg.batch_size.IC,
        parameterization={**{t_symbol:0},**{k_symbol: (0.3, 0.4)}},
    )
    domain.add_constraint(IC, name="IC")

    
    
    # solve over given time period
    interior1 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_x1": 0.0},
        batch_size=cfg.batch_size.interior1,
        parameterization={**time_range,**{k_symbol: (0.3, 0.4)}},
        #criteria=And(t_symbol > 0, t_symbol < 3),
        lambda_weighting={
            "ode_x1": 1.0,
        }
        
    )
    domain.add_constraint(interior1, "interior1")
    
    
    interior2 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_x1": 0.0},
        batch_size=cfg.batch_size.interior2,
        parameterization={**time_range,**{k_symbol: (0.3, 0.4)}},
        criteria=And(t_symbol > 3, t_symbol < 10),
        #quasirandom=cfg.custom.quasirandom,
        
    )
    #domain.add_constraint(interior2, "interior2")
    
    
    
    # add validation data
  
 
   
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
