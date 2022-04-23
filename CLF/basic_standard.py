import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os

parser = argparse.ArgumentParser('./main.py', description='Run individual continual learning experiment.')
parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")


def run(args, verbose=False):
    print("CLR:", args)
    print(torch.__version__)


if __name__ == '__main__':
    # -load input-arguments
    args = parser.parse_args()
    # -run experiment
    run(args, verbose=True)
