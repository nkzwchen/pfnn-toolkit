from sympy import *
import mindspore as md
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

import equation


class LiouvilleBratuEquationLossNet(nn.Cell):
    """NetLoss"""
    def __init__(self, net, Lambda):
        super(LiouvilleBratuEquationLossNet, self).__init__()
        self.matmul = nn.MatMul()
        self.grad = ops.composite.GradOperation()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.net = net
        self.Lambda = Lambda
        self.exp = md.ops.Exp()

    def get_variable(self, InSet, BdSet, g_fun_val):
        """Get Parameters for NetLoss"""
        assert BdSet.has_neumann_boundary == False, "LiouvilleBratu equation only support dirichlet boundary"
        self.InSet_size = InSet.size
        self.InSet_dim = InSet.dim
        self.InSet_area = InSet.area

        InSet_g = g_fun_val[0]
        InSet_gx = g_fun_val[1]
        self.InSet_g = Parameter(Tensor(InSet_g, mstype.float32), name="InSet_g", requires_grad=False)
        self.InSet_l = Parameter(Tensor(InSet.l, mstype.float32), name="InSet_l", requires_grad=False)
        self.InSet_gx = Parameter(Tensor(InSet_gx, mstype.float32), name="InSet_gx", requires_grad=False)
        self.InSet_lx = Parameter(Tensor(InSet.lx, mstype.float32), name="InSet_lx", requires_grad=False)

        self.InSet_c = Parameter(Tensor(InSet.c, mstype.float32), name="InSet_c", requires_grad=False)

    def construct(self, InSet_x):
        """forward"""
        InSet_f = self.net(InSet_x)
        InSet_fx = self.grad(self.net)(InSet_x)
        InSet_u = self.InSet_g + self.InSet_l * InSet_f
        InSet_ux = self.InSet_gx + self.InSet_lx * InSet_f + self.InSet_l * InSet_fx

        InSet_loss = 0.5 * self.InSet_area * self.sum(self.mean((InSet_ux ** 2), 0)) - \
               self.Lambda * self.InSet_area * self.mean(self.exp(InSet_u)) + \
               self.InSet_area * self.mean(self.InSet_c * InSet_u)

        return InSet_loss


class LiouvilleBratuEquation():
    def __init__(self, Lambda, u, x):
        self.Lambda = Lambda
        x1 = x[0]
        x2 = x[1]

        r = diff(u, x1, 2) + diff(u, x2, 2) + self.Lambda * exp(u)
        rhs_function = equation.Function(input=[x1, x2], output=[r])
        u_function = equation.Function(input=[x1, x2], output=[u])
        self.right_handle = self._get_right_handle(rhs_function)
        self.ground_handle = self._get_ground_handle(u_function)

    def _get_right_handle(self, rhs_function):
        return rhs_function

    def _get_ground_handle(self, u_function):
        return u_function
