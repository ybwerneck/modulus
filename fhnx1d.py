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
from modulus.utils.io.plotter import ValidatorPlotter

from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
import matplotlib.pyplot as plt
from modulus.domain.validator.discrete import GridValidator
from modulus.dataset.discrete import DictGridDataset
from modulus.domain.constraint.continuous import DeepONetConstraint

class plotter(ValidatorPlotter):
    "Default plotter class for validator"

    def __call__(self, invar, true_outvar, pred_outvar):
        "Default function for plotting validator data"

        # interpolate 2D data onto grid
        print(len(invar))
        print(np.shape((pred_outvar["u"].flatten())))
        print(len(pred_outvar))
        
        
        invar["K"]=invar["K"]*50
        #print(invar["K"])
        true_outvar["u"]=np.expand_dims(true_outvar["u"], axis=1) 
        
        extent, true_outvar, pred_outvar = self._interpolate_2D(
                200, invar, true_outvar, pred_outvar
            )
        ndim=2
        # make plots
        dims = list(invar.keys())
        fs = []
        print("kk")
        for k in pred_outvar:
            f = plt.figure(figsize=(3 * 5, 4), dpi=100)
            for i, (o, tag) in enumerate(
                zip(
                    [true_outvar[k], pred_outvar[k], ((true_outvar[k] - pred_outvar[k])**2)**(0.5) ],
                    ["true", "pred", "diff"],
                )
            ):
           #     print("extent",extent)
                plt.subplot(1, 3, 1 + i)
                if ndim == 1:
                    plt.plot(invar[dims[0]][:, 0], o[:, 0])
                    plt.xlabel(dims[0])
                elif ndim == 2:
                    plt.imshow(o.T, origin="lower", extent=extent)
                    
                    plt.xlabel(dims[0])
                    plt.ylabel(dims[1])
                    if(tag=="diff"):
                       
                        plt.clim(0,0.3)
                    plt.colorbar()
                plt.title(f"{k}_{tag}")
            plt.tight_layout()
            fs.append((f, k))

        return fs




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
    k_range = {k_symbol: (0.3, 0.5)}

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

    
    ##DD constraint
    
    T=np.empty([0])
    K=np.empty([0])
    SOLs=np.empty([0])
    krange= [(0.3 + 0.01*i*0.5) for i in range(1,10)]

    deltaT = 0.01
    rate = 100
    t = np.linspace(0, t_max, int((t_max/(deltaT)) /rate) )
    t = np.expand_dims(t, axis=-1)
    
    for KR in krange:
        
        
        T=np.append(T,t)
        K = np.append(K,np.full_like (t,KR))
        SOLs=np.append(SOLs,np.array(generateExactSolution(t_max,deltaT,KR,rate,KR)))
    
    
    
    t = np.expand_dims(T, axis=-1)
  


    k = np.expand_dims(K, axis=-1)

    
    Sol = np.expand_dims(SOLs, axis=-1)

    print(np.shape(t),"training set")
    
    
    invar_numpy = {"t": t,"k":k}
    outvar_numpy = {
        "x1": Sol
    }

    
    data = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={"x":t,"K":k},
        outvar={"u":Sol},
        batch_size=20,
    )
    domain.add_constraint(data, "data")
    
    
    
    
    ##validator
    deltaT = 0.01
    rate = 10
    t = np.linspace(0, t_max, int((t_max/(deltaT)) /rate) )
    t = np.expand_dims(t, axis=-1)
    for KR in krange:
        
        
        T=np.append(T,t)
        K = np.append(K,np.full_like (t,KR))
        SOLs=np.append(SOLs,np.array(generateExactSolution(t_max,deltaT,KR,rate,0)))
    
    
    t=np.expand_dims(T,axis=-1)
    k=np.expand_dims(K,axis=-1)
    sol=SOLs
    
    invar_numpy = {"x": t,"K":k}
    outvar_numpy = {
        "u": sol
    }
    
  
    
    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=1024,plotter= plotter()
    )
    domain.add_validator(validator)
    
    
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()



if __name__ == "__main__":
    run()
