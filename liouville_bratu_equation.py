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
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from sympy import symbols
import numpy as np

from model import pfnnmodel
from data import gendata
from train import PfnnSolver
from equation import LiouvilleBratuEquation, LiouvilleBratuEquationLossNet
from util import ArgParse, environment_init


LAMBDA = 0.3
def get_equation():
    x1, x2 = symbols("x1 x2")
    u = ((x1 ** 3 - 3 * x1) * (x2 ** 3 - 3 * x2) - 3)
    
    return LiouvilleBratuEquation(LAMBDA , u, [x1, x2])

def neumann_filter(x):
    return (np.zeros_like(x[..., 0]) > 1e-8)

def dirichlet_filter(x):
    return (np.zeros_like(x[..., 0]) < 1e-8)

if __name__ == "__main__":

    # init environment
    args = ArgParse()
    environment_init(args)
    
    equation = get_equation()
    errors = None
        
    lenfac = pfnnmodel.LenFac(
        Tensor(args.bound, mstype.float32).reshape(2, 2), 1, dirichlet_sides=['x_inf', 'x_sup', 'y_inf', 'y_sup'])
    InSet, BdSet, TeSet = gendata.GenerateSet(args, lenfac, equation, dirichlet_filter, neumann_filter)   
   
    netg = pfnnmodel.NetG()
    netf = pfnnmodel.NetF()
    loss_net_f = LiouvilleBratuEquationLossNet(netf, LAMBDA)
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

