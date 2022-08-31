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
from src.utils import set_optim, Loss_log


class Runner:
    def __init__(self, args, writer, logger):
        self.datapath = edict()
        self.datapath.log_dir = get_dump_path(args)
        self.datapath.model_dir = os.path.join(self.datapath.log_dir, 'model')

        # TODO: init code
        self.args = args
        self.data_init()
        self.writer = writer
        self.writer = writer
        self.logger = logger
        self.optim_init(self.args)

    def optim_init(self, opt):
        step_per_epoch = len(self.train_dataloader)
        opt.warmup_steps = int(step_per_epoch * opt.epoch * 0.06)
        opt.total_steps = int(step_per_epoch * opt.epoch)
        self.logger.info(f"warmup_steps: {opt.warmup_steps}")
        self.logger.info(f"total_steps: {opt.total_steps}")
        self.logger.info(f"weight_decay: {opt.weight_decay}")
        self.optimizer, self.scheduler = set_optim(self.args, self.model)


    def data_init(self):
        data_name = ""
        # with open(self.args.data_path, 'r') as f:
        #     train_examples = json.load(f)
        # with open(osp.join(self.args.data_path, data_name), "rb") as f:
        #     train_examples = pickle.load(f)
        train_examples = torch.load(osp.join(self.args.data_path, data_name))
        pdb.set_trace()
        self.train_dataset = Dataset(train_examples, self.args)
        # train_sampler = RandomSampler(train_dataset)

        self.train_dataloader = DataLoader(
            train_dataset,
            # sampler=train_sampler,
            batch_size=self.args.batch_size,
            # drop_last=True,
            num_workers=6,
            shuffle=True
        )

    def run(self):
        self.loss_log = Loss_log()
        with tqdm(total=self.args.epoch) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            for i in range(self.args.epoch):
                _tqdm.set_description(f'Train | epoch {i} Loss {self.loss_log.get_loss():.5f} Acc {self.loss_log.get_acc()*100:.3f}%')

                # TODO:  for i in epoch:...
                self.train()
                # writer.add_scalars("name",{"dic":val}, epoch)
                # TODO: if ...
                self.eval()
                # https://zhuanlan.zhihu.com/p/382950853
                # writer.add_scalars("loss",{"dic":val}, epoch)

                _tqdm.update(1)

        # TODO: save or load
        self.logger.info(f"min loss {self.loss_log.get_min_loss()}")

        # TODO: save or load
        self._save_model()

    # one time train
    def train(self):
        self.model.train()
        curr_loss = 0.
        curr_loss_dic = {'cls_loss': 0., 'reg_loss': 0., 'orth_loss': 0., 'con_loss': 0.}
        for batch in self.train_dataloader:
            # ... = batch
            # loss = self.model(...)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            curr_loss += train_loss.item()
            curr_loss_dic['cls_loss'] += loss_dic['cls_loss']
            curr_loss_dic['reg_loss'] += loss_dic['reg_loss']
            curr_loss_dic['orth_loss'] += loss_dic['orth_loss']
            if self.args.contrastive_loss:
                curr_loss_dic['con_loss'] += loss_dic['con_loss']

            loss_log.update(curr_loss)
        
        torch.cuda.empty_cache()
        return curr_loss, curr_loss_dic

    # one time eval
    def eval(self):
        self.model.eval()


        torch.cuda.empty_cache()
        pass

    # one time eval
    def test(self):
        pass

    def _load_model(self, model):
        # TODO: path
        save_path = osp.join(self.args.data_path, 'save')
        save_path = osp.join(save_path, f'{self.args.model_name}.pkl')
        self.model.load_state_dict(torch.load(save_path))
        self.model.cuda()
        print(f"loading model done!")

    def _save_model(self, model):

        model_name = type(model).__name__
        # TODO: path

        save_path = osp.join(self.args.data_path, 'save')
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), osp.join(save_path, f'seed_{self.args.random_seed}_{model_name}.pkl'))

        print(f"saving model done!")
        return save_path


if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    
    cfgs = cfg.update_train_configs()
    pdb.set_trace()
    set_seed(cfgs.random_seed)

    pprint.pprint(cfgs)

    logger = initialize_exp(cfgs)
    logger_path = get_dump_path(cfgs)
    writer = None
    cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'

    if not cfgs.no_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    # print("print c to continue...")

    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger)
    runner.run()
    if cfg.only_test:
        runner.test()
    else:
        runner.run()

    if not cfgs.no_tensorboard and not cfg.only_test:
        writer.close()
    logger.info("done!")
