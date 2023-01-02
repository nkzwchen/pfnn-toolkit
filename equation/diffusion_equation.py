from sympy import *
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

import equation


class DiffusionEquationLossNet(nn.Cell):
    """NetLoss"""
    def __init__(self, net):
        super(DiffusionEquationLossNet, self).__init__()
        self.matmul = nn.MatMul()
        self.grad = ops.composite.GradOperation()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.net = net
        self.has_neumann_boundary = False

    def get_variable(self, InSet, BdSet, g_fun_val):
        """Get Parameters for NetLoss"""
        self.InSet_size = InSet.size
        self.InSet_dim = InSet.dim
        self.InSet_area = InSet.area

        InSet_g = g_fun_val[0]
        InSet_gx = g_fun_val[1]
        self.InSet_g = Parameter(Tensor(InSet_g, mstype.float32), name="InSet_g", requires_grad=False)
        self.InSet_l = Parameter(Tensor(InSet.l, mstype.float32), name="InSet_l", requires_grad=False)
        self.InSet_gx = Parameter(Tensor(InSet_gx, mstype.float32), name="InSet_gx", requires_grad=False)
        self.InSet_lx = Parameter(Tensor(InSet.lx, mstype.float32), name="InSet_lx", requires_grad=False)
        self.InSet_a = Parameter(Tensor(InSet.a, mstype.float32), name="InSet_a", requires_grad=False)
        self.InSet_c = Parameter(Tensor(InSet.c, mstype.float32), name="InSet_c", requires_grad=False)
        if BdSet.has_neumann_boundary:
            self.has_neumann_boundary = True
            self.BdSet_nlength = BdSet.n_length
            self.BdSet_nr = Parameter(Tensor(BdSet.n_r, mstype.float32), name="BdSet_nr", requires_grad=False)
            self.BdSet_nl = Parameter(Tensor(BdSet.n_l, mstype.float32), name="BdSet_nl", requires_grad=False)
            BdSet_ng = g_fun_val[2]
            self.BdSet_ng = Parameter(Tensor(BdSet_ng, mstype.float32), name="BdSet_ng", requires_grad=False)

    def construct(self, InSet_x, BdSet_x=None):
        """forward"""
        InSet_f = self.net(InSet_x)
        InSet_fx = self.grad(self.net)(InSet_x)
        InSet_u = self.InSet_g + self.InSet_l * InSet_f
        InSet_ux = self.InSet_gx + self.InSet_lx * InSet_f + self.InSet_l * InSet_fx
        InSet_aux = self.matmul(self.InSet_a, InSet_ux.reshape((self.InSet_size, self.InSet_dim, 1)))
        InSet_aux = InSet_aux.reshape(self.InSet_size, self.InSet_dim)
        InSet_loss = 0.5 * self.InSet_area * self.sum(self.mean((InSet_aux * InSet_ux), 0)) + \
               self.InSet_area * self.mean(self.InSet_c * InSet_u)
        if self.has_neumann_boundary:
            BdSet_nu = self.BdSet_ng + self.BdSet_nl * self.net(BdSet_x)
            BdSet_loss = self.BdSet_nlength * self.mean(self.BdSet_nr * BdSet_nu)
            return InSet_loss - BdSet_loss
        else:
            return InSet_loss


class DiffusionEquation():
    def __init__(self, A, u, x):

        a_11 = A[0][0]
        a_12 = A[0][1]
        a_21 = A[1][0]
        a_22 = A[1][1]
        x1 = x[0]
        x2 = x[1]

        u_x1 = diff(u, x1)
        u_x2 = diff(u, x2)

        r1 = diff(a_11, x1) * u_x1 + \
             a_11 * diff(u, x1, 2) + \
             diff(a_12, x1) * u_x2 + \
             a_12 * diff(u_x1, x2)

        r2 = diff(a_21, x2) * u_x1 + \
             a_21 * diff(u_x1, x2) + \
             diff(a_22, x2) * u_x2 + \
             a_22 * diff(u, x2, 2)
        r = r1 + r2

        rr_1 = a_11 * u_x1 + a_12 * u_x2
        rr_2 = a_21 * u_x1 + a_22 * u_x2

        rhs_function = equation.Function(input=[x1, x2], output=[r])
        a_function = equation.Function(input=[x1, x2], output=[a_11, a_12, a_21, a_22])
        u_function = equation.Function(input=[x1, x2], output=[u])
        rr_function = equation.Function(input=[x1, x2], output=[rr_1, rr_2])
        self.right_handle = self._get_right_handle(rhs_function)
        self.A_handle = self._get_A_handle(a_function)
        self.ground_handle = self._get_ground_handle(u_function)
        self.rn_handle = self._get_rn_handle(rr_function)

    def _get_right_handle(self, rhs_function):
        return rhs_function

    def _get_A_handle(self, a_function):
        def A(x):
            ret = a_function(x)
            return ret.reshape(-1, 2, 2)

        return A

    def _get_ground_handle(self, u_function):
        return u_function

    def _get_rn_handle(self, rr_function):

        rr = rr_function

        def rnhs(x, n):
            return (rr(x) * n).sum(-1, keepdims=True)

        return rnhs
