# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"],
                        help="use cpu or gpu to train")

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

