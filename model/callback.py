"""Train NetG and NetF/NetLoss"""

from mpi4py import MPI
from mindspore.train.callback import Callback
from mindspore import ops
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import save_checkpoint


class SaveCallbackNETG(Callback):
    """
    SavedCall for NetG to print loss and save checkpoint

    Args:
        net(NetG): The instantiation of NetG
        path(str): The path to save the checkpoint of NetG
    """
    def __init__(self, net, path):
        super(SaveCallbackNETG, self).__init__()
        self.loss = 1e5
        self.net = net
        self.path = path
        self.print = ops.Print()

    def step_end(self, run_context):
        """print info and save checkpoint per 100 steps"""
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % 100 == 0:
            comm = MPI.COMM_WORLD
            loss = cb_params.net_outputs.asnumpy()
            loss = loss.sum()
            loss = comm.reduce(loss.item(), root=0)
            rank = comm.Get_rank()

            if rank == 0:
                if bool(loss < self.loss):
                    self.loss = loss
                    save_checkpoint(self.net, self.path)
                self.print(f"NETG epoch : {cb_params.cur_epoch_num}, loss : {loss}")


class SaveCallbackNETLoss(Callback):
    """
    SavedCall for NetG to print loss and save checkpoint

    Args:
        net(NetG): The instantiation of NetF
        path(str): The path to save the checkpoint of NetF
        x(np.array): valid dataset
        ua(np.array): Label of valid dataset
    """
    def __init__(self, net, path, x, l, g, ua):
        super(SaveCallbackNETLoss, self).__init__()
        self.loss = 1e5
        self.error = 1e5
        self.net = net
        self.path = path
        self.l = l
        self.x = x
        self.g = g
        self.ua = ua
        self.print = ops.Print()

    def step_end(self, run_context):
        """print info and save checkpoint per 100 steps"""
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % 100 == 0:
            u = (Tensor(self.g, mstype.float32) +
                 Tensor(self.l, mstype.float32) * self.net(Tensor(self.x, mstype.float32))).asnumpy()
            ground = (self.ua**2).sum()
            error = ((u - self.ua)**2).sum()

            comm = MPI.COMM_WORLD
            ground = comm.reduce(ground, root=0, op=MPI.SUM)
            error = comm.reduce(error, root=0, op=MPI.SUM)
            loss = comm.reduce(cb_params.net_outputs.asnumpy().sum().item(), root=0, op=MPI.SUM)

            rank = comm.Get_rank()
            if rank == 0:
                error = ((error) / (ground + 1e-8))**0.5

                if bool(error < self.error):
                    self.loss = error
                    save_checkpoint(self.net, self.path)

                self.print(f"NETF epoch : {cb_params.cur_epoch_num}, loss : {loss}, error : {error}")
