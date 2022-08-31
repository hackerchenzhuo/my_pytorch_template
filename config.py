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
        parser.add_argument("--exp_name", default="huawei_exp", type=str, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--exp_id", default="001", type=str, help="Experiment ID")
        parser.add_argument("--random_seed", default=42, type=int)

        parser.add_argument("--data_path", default="huawei", type=str, help="Experiment path")
        
        # TODO: add some dynamic variable
        parser.add_argument("--model_name", default="", type=str, help="model name")
        
        
        # 训练阶段
        parser.add_argument('--train_strategy', default=1, type=int)
        parser.add_argument("--scheduler", default="linear", type=str, choices=["linear", "cos"])
        parser.add_argument("--optim", default="adam", type=str)
        parser.add_argument('--lr', type=float, default=3e-5)
        parser.add_argument('--margin', default=9.0, type=float, help='The fixed margin in loss function. ')
        parser.add_argument('--emb_dim', default=1000, type=int, help='The embedding dimension in KGE model.')
        parser.add_argument('--adv_temp', default=1.0, type=float, help='The temperature of sampling in self-adversarial negative sampling.')
        parser.add_argument("--contrastive_loss", default=0, type=int, choices=[0, 1])
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        parser.add_argument('--scheduler_steps', type=int, default=None,
                    help='total number of step for the scheduler, if None then scheduler_total_step = total_step')

        self.cfg = parser.parse_args()
        

    def update_train_configs(self):
        # add some constraint for parameters
        # e.g. cannot save and test at the same time
        assert not (self.cfg.save_model and self.cfg.only_test)

        # TODO: update some dynamic variable
        self.cfg.data_root = self.data_root
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)

        return self.cfg
