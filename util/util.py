import os

import argparse
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import init, get_rank

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
    parser.add_argument("--parallel_mode", type=str, default="SINGLE", choices=["SINGLE", "DATA_PARALLEL"], help="parallel mode to train model")
    parser.add_argument("--path", type=str, default="./optimal_state/",
                        help="the basic folder of g_path and f_path")
    parser.add_argument("--g_path", type=str, default="optimal_state_g_pfnn.ckpt",
                        help="the path that will put checkpoint of netg")
    parser.add_argument("--f_path", type=str, default="optimal_state_f_pfnn.ckpt",
                        help="the path that will put checkpoint of netf")
    _args = parser.parse_args()
    return _args

def environment_init(args: argparse.ArgumentParser):
    """init environment"""
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    
    if args.parallel_mode == "DATA_PARELLEL":
        # context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
       
    init('nccl')
    device_id = int(get_rank())
    context.set_context(device_id=device_id) # set device_id
    
    if not os.path.exists(args.path):
        os.mkdir(args.path)

    args.g_path = args.path + args.g_path
    args.f_path = args.path + args.f_path
