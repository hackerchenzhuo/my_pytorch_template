import os
import os.path as osp
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import pdb
import pprint
import json
import pickle

from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import Dataset


class Runner:
    def __init__(self, args, writer):
        self.datapath = edict()
        self.datapath.log_dir = get_dump_path(args)
        self.datapath.model_dir = os.path.join(self.datapath.log_dir, 'model')

        # TODO: init code
        self.args = args
        self.data_init()
        self.writer = writer

    def data_init(self):
        data_name = ""
        # with open(self.args.data_path, 'r') as f:
        #     train_examples = json.load(f)
        # with open(osp.join(self.args.data_path, data_name), "rb") as f:
        #     train_examples = pickle.load(f)
        train_examples = torch.load(osp.join(self.args.data_path, data_name))
        pdb.set_trace()
        train_dataset = Dataset(train_examples, self.args)
        # train_sampler = RandomSampler(train_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            # sampler=train_sampler,
            batch_size=self.args.batch_size,
            # drop_last=True,
            num_workers=6,
            shuffle=True
        )

    def run(self):
        # TODO:  for i in epoch:...
        self.train()
        # writer.add_scalars("name",{"dic":val}, epoch)
        # TODO: if ...
        self.eval()
        # https://zhuanlan.zhihu.com/p/382950853
        # writer.add_scalars("loss",{"dic":val}, epoch)

        # TODO: save or load

    # one time train
    def train(self):

        pass

    # one time eval
    def eval(self):
        pass

    def _load_model(self, model):
        # TODO: path
        save_path = ""
        model.load_state_dict(torch.load(save_path))
        print(f"loading model done!")

    def _save_model(self, model):

        model_name = type(model).__name__
        # TODO: path

        save_path = "..."
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"saving model done!")
        return save_path


if __name__ == '__main__':
    cfg = cfg()
    args = cfg.get_args()
    cfg.update_train_configs(args)
    set_seed(cfg.random_seed)

    pprint.pprint(args)

    logger = initialize_exp(cfg)
    logger_path = get_dump_path(cfg)
    writer = None
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    comment = f'bath_size={cfg.batch_size} exp_id={cfg.exp_id}'

    if not cfg.no_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', time_stamp), comment=comment)

    # print("print c to continue...")

    torch.cuda.set_device(cfg.gpu)
    runner = Runner(cfg, writer)
    runner.run()

    if not cfg.no_tensorboard:
        writer.close()
    logger.info("done!")
