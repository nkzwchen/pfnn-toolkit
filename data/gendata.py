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
"""Data for PFNN"""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

class InnerSet():
    """Inner Set"""
    def __init__(self, bounds, nx, lenfac, equation):
        self.dim = 2
        self.bounds = bounds
        self.lenfac = lenfac
        self.area = (self.bounds[0, 1] - self.bounds[0, 0]) * \
                    (self.bounds[1, 1] - self.bounds[1, 0])

        self.gp_num = 2
        self.gp_wei = [1.0, 1.0]
        self.gp_pos = [(1 - 0.5773502692) / 2, (1 + 0.5773502692) / 2]

        self.nx = [int(nx[0] / self.gp_num), int(nx[1] / self.gp_num)]
        self.hx = [(self.bounds[0, 1] - self.bounds[0, 0]) / self.nx[0],
                   (self.bounds[1, 1] - self.bounds[1, 0]) / self.nx[1]]

        self.size = self.nx[0] * self.gp_num * self.nx[1] * self.gp_num
        self.x = np.zeros([self.size, self.dim], dtype=np.float32)
        self.a = None
        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.gp_num):
                    for l in range(self.gp_num):
                        self.x[m, 0] = self.bounds[0, 0] + \
                            (i + self.gp_pos[k]) * self.hx[0]
                        self.x[m, 1] = self.bounds[1, 0] + \
                            (j + self.gp_pos[l]) * self.hx[1]
                        m = m + 1
        
        if hasattr(equation, 'A_handle'):
            self.a = equation.A_handle(self.x)
        else:
            self.a = None
        self.c = equation.right_handle(self.x)
        self.ua = equation.ground_handle(self.x)

        grad_ = ops.composite.GradOperation(get_all=True)

        self.l = self.lenfac(Tensor(self.x, mstype.float32)).asnumpy()
        self.lx = grad_(self.lenfac)(Tensor(self.x, mstype.float32))[
        0].asnumpy()



class BoundarySet():
    """Boundary Set"""
    def __init__(self, bounds, nx, lenfac, equation, dirichlet_filter, neumann_filter):
        self.dim = 2
        self.bounds = bounds
        self.lenfac = lenfac
        self.length = 2 * (bounds[0, 1] - bounds[0, 0]) + \
                      2 * (bounds[1, 1] - bounds[1, 0])

        self.gp_num = 2
        self.gp_wei = [1.0, 1.0]
        self.gp_pos = [(1 - 0.5773502692) / 2, (1 + 0.5773502692) / 2]
        self.nx = [int(nx[0] / self.gp_num), int(nx[1] / self.gp_num)]
        self.hx = [(self.bounds[0, 1] - self.bounds[0, 0]) / self.nx[0],
                   (self.bounds[1, 1] - self.bounds[1, 0]) / self.nx[1]]

        self.d_x = None
        self.d_a = None

        self.n_x = None
        self.n_n = None
        self.n_a = None
        self.n_l = None
        self.has_neumann_boundary = False

        x = np.zeros([2 * (self.nx[0] + self.nx[1]) * self.gp_num, self.dim], dtype=np.float32)
        n = np.zeros([2 * (self.nx[0] + self.nx[1]) * self.gp_num, self.dim], dtype=np.float32)
        m = 0

        for j in range(self.nx[1]):
            for l in range(self.gp_num):
                x[m, 0] = self.bounds[0, 0]
                x[m, 1] = self.bounds[1, 0] + \
                    (j + self.gp_pos[l]) * self.hx[1]
                n[m, 0] = -1.0
                n[m, 1] = 0.0
                m = m + 1
 
        for i in range(self.nx[0]):
            for k in range(self.gp_num):
                x[m, 0] = self.bounds[0, 0] + \
                    (i + self.gp_pos[k]) * self.hx[0]
                x[m, 1] = self.bounds[1, 0]
                n[m, 0] = 0.0
                n[m, 1] = -1.0
                m = m + 1
        for j in range(self.nx[1]):
            for l in range(self.gp_num):
                x[m, 0] = self.bounds[0, 1]
                x[m, 1] = self.bounds[1, 0] + \
                    (j + self.gp_pos[l]) * self.hx[1]
                n[m, 0] = 1.0
                n[m, 1] = 0.0
                m = m + 1
        for i in range(self.nx[0]):
            for k in range(self.gp_num):
                x[m, 0] = self.bounds[0, 0] + \
                    (i + self.gp_pos[k]) * self.hx[0]
                x[m, 1] = self.bounds[1, 1]
                n[m, 0] = 0.0
                n[m, 1] = 1.0
                m = m + 1

        d_index = dirichlet_filter(x)
        n_index = neumann_filter(x)

        assert d_index.sum() != 0, "Missing dirichlet boundary!"
        self.d_x = x[d_index]
        if hasattr(equation, 'A_handle'):
            self.d_a = equation.A_handle(self.d_x)
        else:
            self.d_a = None
        self.d_r = equation.ground_handle(self.d_x)
        self.d_length = self.length * (d_index.sum() / x.shape[0])
            
        if n_index.sum() != 0:
            self.has_neumann_boundary = True
            self.n_length = self.length * (n_index.sum() / x.shape[0])
            self.n_x = x[n_index]
            self.n_n = n[n_index]
            if hasattr(equation, 'A_handle'):
                self.n_a = equation.A_handle(self.n_x)
            else:
                self.n_a = None
            self.n_r = equation.rn_handle(self.n_x, self.n_n)
            self.n_l = self.lenfac(Tensor(self.n_x, mstype.float32)).asnumpy()
        

class TestSet():
    """Test Set"""
    def __init__(self, bounds, nx, equation):
        self.dim = 2
        self.bounds = bounds
        self.nx = nx
        self.hx = [(self.bounds[0, 1] - self.bounds[0, 0]) / (self.nx[0] - 1),
                   (self.bounds[1, 1] - self.bounds[1, 0]) / (self.nx[1] - 1)]

        self.size = self.nx[0] * self.nx[1]
        self.x = np.zeros([self.size, self.dim], dtype=np.float32)
        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                self.x[m, 0] = self.bounds[0, 0] + i * self.hx[0]
                self.x[m, 1] = self.bounds[1, 0] + j * self.hx[1]
                m = m + 1
        
        self.ua = equation.ground_handle(self.x)


def GenerateSet(args, lenfac, equation, dirichlet_filter, neumann_filter):
    """
    Generate Set
    """
    bound = np.array(args.bound).reshape(2, 2)
    return InnerSet(bound, args.inset_nx, lenfac, equation),\
        BoundarySet(bound, args.bdset_nx, lenfac, equation, dirichlet_filter, neumann_filter), \
        TestSet(bound, args.teset_nx, equation)

