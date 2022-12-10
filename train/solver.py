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
import time

from mpi4py import MPI
from mindspore import context, Tensor
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from src import callback
from data import dataset

class PfnnSolver():
    def __init__(self, **cfg):
        self.args = cfg['args']

        self.InSet = cfg['InSet']
        self.BdSet = cfg['BdSet']
        self.TeSet = cfg['TeSet']

        self.net_f = cfg['net_f']
        self.net_g = cfg['net_g']
        self.loss_net_f = cfg['loss_net_f']

        self.lenfac = cfg['lenfac']

        self.optim_f = cfg['optim_f']
        self.optim_g = cfg['optim_g']

        self.g_epochs = self.args.g_epochs
        self.f_epochs = self.args.f_epochs

        self.g_path = self.args.g_path
        self.f_path = self.args.f_path
        self.dataset_g, self.dataset_f = dataset.GenerateDataSet(self.InSet, self.BdSet)


    def solve(self):
        if self.args.device == "gpu":
           # context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            print("START TRAINING")
        start_gnet_time = time.time()
        self._train_g()
        elapsed_gnet = time.time() - start_gnet_time
        start_fnet_time = time.time()
        self._train_f()
        elapsed_fnet = time.time() - start_fnet_time
        if rank == 0:
            print("Train NetG total time: %.2f, train NetG one step time: %.5f" %
                (elapsed_gnet, elapsed_gnet/self.g_epochs))
            print("Train NetF total time: %.2f, train NetF one step time: %.5f" %
                    (elapsed_fnet, elapsed_fnet/self.f_epochs))

        errors = self._calerror()
        errors = comm.reduce(errors.item(), root=0, op=MPI.SUM)
        if rank == 0:        
            print("test_error = %.3e\n" % (errors))
    
    def _train_g(self):
        """
        The process of preprocess and process to train NetG
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            print("START TRAIN NEURAL NETWORK G")
        model = Model(network=self.net_g, loss_fn=None, optimizer=self.optim_g)
        model.train(self.g_epochs, self.dataset_g, callbacks=[
                callback.SaveCallbackNETG(self.net_g, self.g_path)], dataset_sink_mode=True)
    
    def _train_f(self):

        g_func_val = self._get_func_val()
        
        self.loss_net_f.get_variable(self.InSet, self.BdSet, g_func_val)

        print("START TRAIN NEURAL NETWORK F")
        model = Model(network=self.loss_net_f,
                    loss_fn=None, optimizer=self.optim_f)
        model.train(self.f_epochs, self.dataset_f, callbacks=[callback.SaveCallbackNETLoss(
            self.net_f, self.f_path, self.InSet.x, self.InSet.l, g_func_val[0], self.InSet.ua)], dataset_sink_mode=True)
    
    def _get_func_val(self):
        grad_ = ops.composite.GradOperation(get_all=True)
        InSet_g = self.net_g(Tensor(self.InSet.x, mstype.float32))
        InSet_gx = grad_(self.net_g)(Tensor(self.InSet.x, mstype.float32))[0]
        if self.BdSet.has_neumann_boundary:
            BdSet_ng = self.net_g(Tensor(self.BdSet.n_x, mstype.float32))
        else:
            BdSet_ng = None
        return [InSet_g, InSet_gx, BdSet_ng]
    
    def _calerror(self):
        """
        The eval function
        Args:
            netg: Instantiation of NetG
            netf: Instantiation of NetF
            lenfac: Instantiation of LenFac
            TeSet: Test Dataset
        Return:
            error: Test error
        """
        x = Tensor(self.TeSet.x, mstype.float32)
        TeSet_u = (self.net_g (x) + self.lenfac (Tensor(x)) * self.net_f (x)).asnumpy()
        Teerror = (((TeSet_u - self.TeSet.ua)**2).sum() /
                (self.TeSet.ua ** 2).sum()) ** 0.5
        return Teerror

