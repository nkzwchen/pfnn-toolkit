"""Generate Dataset"""
import mindspore.dataset as ds
from mindspore.communication.management import get_rank, get_group_size

import numpy as np

class DataSetNetG():
    """
    Generate Training Dataset to train NetG
    """
    def __init__(self, data, label, dtype=np.float32):
        self.data = data.astype(dtype)
        self.label = label.astype(dtype)

    def __getitem__(self, index):
        return (self.data, self.label)

    def __len__(self):
        return 1


class DataSetNetLoss():
    """
    Generator Training Dataset to train NetF
    """
    def __init__(self, x, nx=None, has_neumann_boundary=False, dtype=np.float32):
        self.x = x.astype(dtype)
        self.has_neumann_boundary = has_neumann_boundary
         
        if self.has_neumann_boundary is True:
            self.nx = nx.astype(dtype)
        else:
            self.nx = None
            
    def __getitem__(self, index):
        if self.has_neumann_boundary:
            return (self.x, self.nx)
        else:
           return (self.x)

    def __len__(self):
        return 1


def GenerateDataSet(inset, bdset):
    """
    Generator Dataset for training

    Args:
        inset: Inner Set
        bdset: Boundary Set
    """
    rank_id = get_rank()
    rank_size = get_group_size()
    datasetnetg = DataSetNetG(bdset.d_x, bdset.d_r)
    DS_NETG = ds.GeneratorDataset(
        datasetnetg, ["data", "label"], shuffle=False, num_shards=rank_size, shard_id=rank_id)
    
    if bdset.has_neumann_boundary:
        datasetnetloss = DataSetNetLoss(inset.x, bdset.n_x, has_neumann_boundary=True)
        DS_NETL = ds.GeneratorDataset(
        datasetnetloss, ["x_inset", "x_bdset"], shuffle=False, num_shards=rank_size, shard_id=rank_id)
    else:
        datasetnetloss = DataSetNetLoss(inset.x)
        DS_NETL = ds.GeneratorDataset(
        datasetnetloss, ["x_inset"], shuffle=False, num_shards=rank_size, shard_id=rank_id)

    return DS_NETG, DS_NETL
