import sympy
import numpy as np


class Function(object):
    def __init__(self, input: list, output: list):
        self._has_lambdified = False
        self.input = input
        self.output = output
        self.input_dim = len(input)
        self.output_dim = len(output)
        self._func = []
        self._lambdify()

    def _lambdify(self, method='numpy'):

        for u in self.output:
            self._func.append(sympy.lambdify(self.input, u))
        self._has_lambdified = True

    def __call__(self, x):
        assert (self._has_lambdified)

        shape = list(x.shape)
        assert shape[-1] == len(self.input)

        x = x.reshape(-1, shape[-1])

        output_shape = list(x.shape)
        output_shape[-1] = len(self._func)
        u = np.empty(output_shape)

        for idx, f in enumerate(self._func):
            xx = {}
            for vidx, v in enumerate(self.input):
                xx[str(v)] = x[:, vidx]
            u[:, idx] = f(**xx)

        shape[-1] = len(self._func)
        return u.reshape(shape)