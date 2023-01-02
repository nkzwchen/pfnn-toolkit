"""PFNN solve diffusion equation demo"""

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from sympy import symbols, ln
import numpy as np

from equation import DiffusionEquation, DiffusionEquationLossNet
from model import pfnnmodel
from data import gendata
from train import PfnnSolver
from util import ArgParse, environment_init


def get_equation():
    """set diffusion equation"""
    x1, x2 = symbols("x1 x2")
    u = ln(10 * (x1 + x2)**2 + (x1 - x2)**2 + 0.5)

    a_11 = (x1 + x2)**2 + 1
    a_12 = (-(x1**2) + x2**2)
    a_21 = (-(x1**2) + x2**2)
    a_22 = (x1 - x2)**2 + 1

    return DiffusionEquation([[a_11, a_12], [a_21, a_22]], u, [x1, x2])


def neumann_filter(x: np.ndarray) -> np.ndarray:
    """filter to select points in neumann boundary"""
    return (abs(x[..., 0] - (-1)) > 1e-8)


def dirichlet_filter(x: np.ndarray) -> np.ndarray:
    """filter to select points in dirichlet_filter"""
    return (abs(x[..., 0] - (-1)) < 1e-8)


if __name__ == "__main__":

    # init environment
    args = ArgParse()
    environment_init(args)

    #sample data
    equation = get_equation()
    lenfac = pfnnmodel.LenFac(Tensor(args.bound, mstype.float32).reshape(2, 2), 1, dirichlet_sides=['x_inf'])
    InSet, BdSet, TeSet = gendata.GenerateSet(args, lenfac, equation, dirichlet_filter, neumann_filter)

    #construct model
    netg = pfnnmodel.NetG()
    netf = pfnnmodel.NetF()
    loss_net_f = DiffusionEquationLossNet(netf)

    optimg = nn.Adam(netg.trainable_params(), learning_rate=args.g_lr)
    optimf = nn.Adam(netf.trainable_params(), learning_rate=args.f_lr)

    #train model to solve diffusion equation
    solver = PfnnSolver(args=args,
                        InSet=InSet,
                        BdSet=BdSet,
                        TeSet=TeSet,
                        net_f=netf,
                        net_g=netg,
                        loss_net_f=loss_net_f,
                        lenfac=lenfac,
                        optim_f=optimf,
                        optim_g=optimg)
    solver.solve()
