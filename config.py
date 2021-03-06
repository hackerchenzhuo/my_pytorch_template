import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse

class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', '..', 'data', ''))

        # TODO: add some static variable  (The frequency of change is low)

    def get_args(self):
        parser = argparse.ArgumentParser()
        # base
        parser.add_argument('--gpu', default=0, type=int)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--epoch', default=100, type=int)
        parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])
        parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])

        # torthlight
        parser.add_argument("--no_tensorboard", default=False, action="store_true")
        parser.add_argument("--exp_name", default="test", type=str, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--exp_id", default="001", type=str, help="Experiment ID")
        parser.add_argument("--random_seed", default=1104, type=int)

        # TODO: add some dynamic variable 


        args = parser.parse_args()
        return args

    def update_train_configs(self, args):
        self.gpu = args.gpu
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.save_model = args.save_model
        self.only_test = args.only_test
        self.no_tensorboard = args.no_tensorboard
        self.exp_name = args.exp_name
        self.dump_path = args.dump_path
        self.exp_id = args.exp_id
        self.random_seed = args.random_seed

        # add some constraint for parameters
        # e.g. cannot save and test at the same time
        assert not (self.save_model and self.only_test)

        # TODO: update some dynamic variable
