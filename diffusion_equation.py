"""Run PFNN"""
import argparse
import os
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from src import pfnnmodel
from data import gendata
from train import PfnnSolver
from sympy import symbols, ln
from equation import DiffusionEquation, DiffusionEquationLossNet
from mindspore.communication.management import init, get_rank
from mindspore import context
from mindspore.context import ParallelMode

def get_equation():
    x1, x2 = symbols("x1 x2")
    u = ln(10 * (x1 + x2) ** 2 + (x1 - x2) ** 2 + 0.5)

    a_11 = (x1 + x2) ** 2 + 1
    a_12 = (-(x1 ** 2) + x2 ** 2)
    a_21 = (-(x1 ** 2) + x2 ** 2)
    a_22 = (x1 - x2) ** 2 + 1
    
    return DiffusionEquation([[a_11, a_12], [a_21, a_22]], u, [x1, x2])

def ArgParse():
    """Get Args"""
    parser = argparse.ArgumentParser(
        description="Penalty-Free Neural Network Method")
    parser.add_argument("--bound", type=float, default=[-1.0, 1.0, -1.0, 1.0],
                        help="lower and upper bound of the domain")
    parser.add_argument("--inset_nx", type=int, default=[60, 60],
                        help="size of the inner set")
    parser.add_argument("--bdset_nx", type=int, default=[60, 60],
                        help="size of the boundary set")
    parser.add_argument("--teset_nx", type=int, default=[101, 101],
                        help="size of the test set")
    parser.add_argument("--g_epochs", type=int, default=6000,
                        help="number of epochs to train neural network g")
    parser.add_argument("--f_epochs", type=int, default=6000,
                        help="number of epochs to train neural network f")
    parser.add_argument("--g_lr", type=float, default=0.01,
                        help="learning rate to train neural network g")
    parser.add_argument("--f_lr", type=float, default=0.01,
                        help="learning rate to train neural network f")
    parser.add_argument("--parallel_mode", type=str, default="SINGLE", choices=["SINGLE", "DATA_PARALLEL", "AUTO_PARALLEL"],
                        help="parallel mode to train model")
    parser.add_argument("--path", type=str, default="./optimal_state/",
                        help="the basic folder of g_path and f_path")
    parser.add_argument("--g_path", type=str, default="optimal_state_g_pfnn.ckpt",
                        help="the path that will put checkpoint of netg")
    parser.add_argument("--f_path", type=str, default="optimal_state_f_pfnn.ckpt",
                        help="the path that will put checkpoint of netf")
    _args = parser.parse_args()
    return _args

def neumann_filter(x):
    return (abs(x[..., 0] - (-1)) > 1e-8)

def dirichlet_filter(x):
    return (abs(x[..., 0] - (-1)) < 1e-8)

if __name__ == "__main__":
    args = ArgParse()
    
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if args.parallel_mode == "AUTO_PARALLEL":
       context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    init('nccl')
    device_id = int(get_rank())
    context.set_context(device_id=device_id) # set device_id
    
    equation = get_equation()

    errors = None

    if not os.path.exists(args.path):
        os.mkdir(args.path)

    args.g_path = args.path + args.g_path
    args.f_path = args.path + args.f_path
        
    lenfac = pfnnmodel.LenFac(
        Tensor(args.bound, mstype.float32).reshape(2, 2), 1, dirichlet_sides=['x_inf'])
    InSet, BdSet, TeSet = gendata.GenerateSet(args, lenfac, equation, dirichlet_filter, neumann_filter)   
   
    netg = pfnnmodel.NetG()
    netf = pfnnmodel.NetF()
    loss_net_f = DiffusionEquationLossNet(netf)
    
    optimg = nn.Adam(netg.trainable_params(), learning_rate=args.g_lr)
    optimf = nn.Adam(netf.trainable_params(), learning_rate=args.f_lr)
    
    solver = PfnnSolver(args = args,
                        InSet = InSet,
                        BdSet = BdSet,
                        TeSet = TeSet,
                        net_f = netf,
                        net_g = netg,
                        loss_net_f = loss_net_f,
                        lenfac = lenfac,
                        optim_f = optimf,
                        optim_g = optimg)
    solver.solve()

