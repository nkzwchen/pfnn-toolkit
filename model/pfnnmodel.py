"""
Define the network of PFNN
A penalty-free neural network method for solving a class of
second-order boundary-value problems on complex geometries
"""
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.common.initializer import Normal


class LenFac(nn.Cell):
    """
    Caclulate the length

    Args:
        bounds: Boundary of area
    """
    def __init__(self, bounds, mu, dirichlet_sides):
        super(LenFac, self).__init__()
        self.dirichlet_sides = dirichlet_sides
        self.bounds = bounds
        self.hx = self.bounds[0, 1] - self.bounds[0, 0]
        self.mu = mu

    def cal_l(self, x):
        """caclulate function"""

        l_list = []
        if 'x_inf' in self.dirichlet_sides:
            l_list.append(1.0 - (1.0 - (x[..., 0:1] - self.bounds[0, 0]) / self.hx)**self.mu)
        if 'y_inf' in self.dirichlet_sides:
            l_list.append(1.0 - (1.0 - (x[..., 1:2] - self.bounds[1, 0]) / self.hx)**self.mu)
        if 'x_sup' in self.dirichlet_sides:
            l_list.append(1.0 - (1.0 - (self.bounds[0, 1] - x[..., 0:1]) / self.hx)**self.mu)
        if 'y_sup' in self.dirichlet_sides:
            l_list.append(1.0 - (1.0 - (self.bounds[1, 1] - x[..., 1:2]) / self.hx)**self.mu)
        ret = None
        for l in l_list:
            if ret == None:
                ret = l
            else:
                ret *= l
        return ret

    def construct(self, x):
        """forward"""
        return self.cal_l(x)


class NetG(nn.Cell):
    """NetG"""
    def __init__(self):
        super(NetG, self).__init__()
        self.sin = ops.Sin()
        self.fc0 = nn.Dense(2, 10, weight_init=Normal(0.2), bias_init=Normal(0.2))
        self.fc1 = nn.Dense(10, 10, weight_init=Normal(0.2), bias_init=Normal(0.2))
        self.fc2 = nn.Dense(10, 1, weight_init=Normal(0.2), bias_init=Normal(0.2))
        self.w_tensor = Tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], mstype.float32)
        self.w = Parameter(self.w_tensor, name="w", requires_grad=False)
        self.matmul = nn.MatMul()

    def network_without_label(self, x):
        """caclulate without label"""
        z = self.matmul(x, self.w)
        h = self.sin(self.fc0(x))
        x = self.sin(self.fc1(h)) + z
        return self.fc2(x)

    def network_with_label(self, x, label):
        """caclulate with label"""
        x = self.network_without_label(x)
        return ((x - label)**2).mean()

    def construct(self, x, label=None):
        """forward"""
        if label is None:
            return self.network_without_label(x)
        return self.network_with_label(x, label)


class NetF(nn.Cell):
    """NetF"""
    def __init__(self):
        super(NetF, self).__init__()
        self.sin = ops.Sin()
        self.fc0 = nn.Dense(2, 10, weight_init=Normal(0.2), bias_init=Normal(0.2))
        self.fc1 = nn.Dense(10, 10, weight_init=Normal(0.2), bias_init=Normal(0.2))
        self.fc2 = nn.Dense(10, 10, weight_init=Normal(0.2), bias_init=Normal(0.2))
        self.fc3 = nn.Dense(10, 10, weight_init=Normal(0.2), bias_init=Normal(0.2))
        self.fc4 = nn.Dense(10, 10, weight_init=Normal(0.2), bias_init=Normal(0.2))
        self.fc5 = nn.Dense(10, 10, weight_init=Normal(0.2), bias_init=Normal(0.2))
        self.fc6 = nn.Dense(10, 1, weight_init=Normal(0.2), bias_init=Normal(0.2))

        self.w_tensor = Tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], mstype.float32)
        self.w = Parameter(self.w_tensor, name="w", requires_grad=False)
        self.matmul = nn.MatMul()

    def construct(self, x):
        """forward"""
        z = self.matmul(x, self.w)
        h = self.sin(self.fc0(x))
        x = self.sin(self.fc1(h)) + z
        h = self.sin(self.fc2(x))
        x = self.sin(self.fc3(h)) + x
        h = self.sin(self.fc4(x))
        x = self.sin(self.fc5(h)) + x
        return self.fc6(x)
