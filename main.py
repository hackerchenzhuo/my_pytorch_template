import os
import os.path as osp
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import pdb
import pprint
import json
import pickle
from collections import defaultdict

from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import load_data, Collator_base,
from src.utils import set_optim, Loss_log

# 并行相关
from src.distributed_utils import init_distributed_mode, dist_pdb, is_main_process, reduce_value, cleanup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

class Runner:
    def __init__(self, args, writer, writer=None, logger=None, rank=0):
        self.datapath = edict()
        self.datapath.log_dir = get_dump_path(args)
        self.datapath.model_dir = os.path.join(self.datapath.log_dir, 'model')
        self.rank = rank
        # TODO: data init code
        self.args = args
        self.data_init()
        self.writer = writer
        self.logger = logger
        self.scaler = GradScaler()
        # TODO: model init code
        self.model_list = []
        self.model_choise()
        
        set_seed(args.random_seed)
        if self.args.only_test:
            # TODO: 测试的时候dataloader如何..
            self.dataloader_init()
            pass
        else:
            # TODO: 训练的时候dataloader如何..
            self.dataloader_init()
            if self.args.dist:
                # 并行训练需要权值共享
                # 初始化self.model_list
                self.model_sync()
            else:
                # 初始化self.model_list
                
            self.optim_init(self.args)


    def model_sync(self):
        folder = osp.join(self.args.data_path, "tmp")
        if not os.path.exists(folder):
            # 依次创建所有不存在的目录
            os.makedirs(folder)
        checkpoint_path = osp.join(folder, "initial_weights.pt")
        if self.rank == 0:
            torch.save(self.model.state_dict(), checkpoint_path)
        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        self.model = self._model_sync(self.model, checkpoint_path)
        
    def _model_sync(self, model, checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.args.device))
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.args.device)
        # 这个地方是GPU还是device
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)
        self.model_list.append(model)
        model = model.module
        return model
    
    def model_choise(self):
        assert self.args.model_name in ["EVA", "MSNEA", "MEAformer"]
        # TODO: 选择模型
        # 载入模型
        
        pass
    
    # 支持人工传入accumulation_step和total_step以限定优化区间
    def optim_init(self, opt, total_step=None, accumulation_step=None):
        step_per_epoch = len(self.train_dataloader)
        opt.warmup_steps = int(step_per_epoch * opt.epoch * 0.1)
        opt.total_steps = int(step_per_epoch * opt.epoch) if total_step is None else int(total_step)
        
        if self.rank == 0 and total_step is None:
            self.logger.info(f"warmup_steps: {opt.warmup_steps}")
            self.logger.info(f"total_steps: {opt.total_steps}")
            self.logger.info(f"weight_decay: {opt.weight_decay}")
        # TODO: freeze_part 添加
        freeze_part = []
        
        self.optimizer, self.scheduler = set_optim(opt, self.model_list, freeze_part, accumulation_step)


    def data_init(self):
        data_name = ""
        # load_data 函数设计
        # train_examples = torch.load(osp.join(self.args.data_path, data_name))
        # pdb.set_trace()
        self.train_set, self.test_set, self.eval_set= load_data(self.logger, self.args)
        if self.args.dist and not self.args.only_test:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set)
            self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.eval_set)


    def dataloader_init(self):
        bs = self.args.batch_size
        # TODO: Collator Collator_base,
        
        if self.args.dist and not self.args.only_test:
            # 分布式情况
            self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
            # TODO: self.dataloader _dataloader_dist 训练测试验证
        
        else:
            # 普通情况
            # TODO: self.dataloader _dataloader 训练测试验证
    
    def _dataloader_dist(self, train_set, train_sampler, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=self.args.workers,
            persistent_workers=True, #True
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=True,
            shuffle=(self.args.only_test == 0),
            drop_last=(self.args.only_test == 0),
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader
    
    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.loss_weight = [1, 1]
        self.step = 0
        
        with tqdm(total=self.args.epoch) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            for i in range(self.args.epoch):
                # _tqdm.set_description(f'Train | epoch {i} Loss {self.loss_log.get_loss():.5f} Acc {self.loss_log.get_acc()*100:.3f}%')
                if self.args.dist and not self.args.only_test:
                    self.train_sampler.set_epoch(i)
                # -------------------------------
                self.train(_tqdm)
                # 每个epoch统计一次loss
                loss_log.update(self.curr_loss)
                self.update_loss_log()
                # writer.add_scalars("name",{"dic":val}, epoch)
                if i % eval_epoch == 0 and i > 0:
                    self.eval(_tqdm)
                
                # https://zhuanlan.zhihu.com/p/382950853
                # writer.add_scalars("loss",{"dic":val}, epoch)

                _tqdm.update(1)

        # TODO: save or load
        if self.rank == 0:
            self.logger.info(f"min loss {self.loss_log.get_min_loss()}")
            if not self.args.only_test and self.args.save_model:
            # TODO: save or load
            self._save_model(self.model)

    # one time train
    def train(self, _tqdm):
        self.model.train()
        curr_loss = 0.
        self.loss_log.acc_init()
        accumulation_steps = self.args.accumulation_steps
        torch.cuda.empty_cache()
        
        for batch in self.train_dataloader:
            # with autocast():
            # ... = batch
            # TODO: loss
            # output 包含 loss_dic, loss_weight
            loss, output = self.model(...)
            loss = loss / accumulation_steps
            # loss.backward()
            self.scaler.scale(loss).backward()
            if self.args.dist:
                loss = reduce_value(loss, average=True)
            self.step += 1
            
            # -------- 模型统计 --------
            if not self.args.dist or is_main_process():
                curr_loss += loss.item()
                _tqdm.set_description(f'Train | step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                self.output_statistic(loss, output)
                
        
            # -------- 梯度累计与模型更新 --------
            if self.step % accumulation_steps == 0 and self.step > 0:
                # 更新优化器
                self.scaler.unscale_(self.optimizer)
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    # pdb.set_trace()
                    self.scheduler.step()
                
                if not self.args.dist or is_main_process():
                    # pdb.set_trace()
                    self.lr = self.scheduler.get_last_lr()[-1]
                    self.writer.add_scalars("lr", {"lr": self.lr}, self.step)
                # 模型update
                for model in self.model_list:
                    model.zero_grad(set_to_none=True)
            
            if self.args.dist:
                torch.cuda.synchronize(self.args.device)

        return curr_loss

    def output_statistic(self, loss, output):
        # 统计模型的各种输出
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        
        if 'loss_weight' in output and output['loss_weight'] is not None:
            self.loss_weight = output['loss_weight']
            
            
    def update_loss_log(self):
        # 把统计的模型各种输出存下来
        # https://zhuanlan.zhihu.com/p/382950853
        #  "mask_loss": self.curr_loss_dic['mask_loss'], "ke_loss": self.curr_loss_dic['ke_loss']
        vis_dict = {"train_loss": self.curr_loss}
        vis_dict.update(self.curr_loss_dic)
        self.writer.add_scalars("loss", vis_dict, self.step)
        # TODO: 统计每个loss的weight
        if self.loss_weight is not None:
            loss_weight_dic = {}
            loss_weight_dic["mask"] = 1 / (self.loss_weight[0]**2)
            loss_weight_dic["kpi"] = 1 / (self.loss_weight[1]**2)
            self.writer.add_scalars("loss_weight", loss_weight_dic, self.step)
            # vis_kpi_dic = {"recover": 1 / (self.kpi_loss_weight[0]**2), "classifier": 1 / (self.kpi_loss_weight[1]**2)}
            # if self.args.contrastive_loss and len(self.kpi_loss_weight) > 2:
            #     vis_kpi_dic.update({"contrastive": 1 / (self.kpi_loss_weight[2]**2)})
            # self.writer.add_scalars("kpi_loss_weight", vis_kpi_dic, self.step)
        
        # init log loss
        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.

    # one time eval
    def eval(self, _tqdm):
        self.model.eval()


        torch.cuda.empty_cache()
        pass

    # one time eval
    def test(self):
        pass

    
    def _load_model(self, model, model_name=None):
        # TODO: path
        if model_name is None:
            model_name = self.args.model_name_save
        save_path = osp.join(self.args.data_path, self.args.model_name, 'save')
        save_path = osp.join(save_path, f'{model_name}.pkl')
        if (len(model_name) == 0 or not os.path.exists(save_path)) and self.rank == 0:
            if len(model_name) > 0
                self.logger.info(f"{model_name}.pkl not exist!!")
            else:   
                self.logger.info("Random init...")
            return model
        
        
        if 'Dist' in self.args.model_name:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(save_name), map_location=self.args.device).items()})
        else:
            model.load_state_dict(torch.load(save_name, map_location=self.args.device))
        
        model.cuda()
        if self.rank == 0:
            self.logger.info(f"loading model [{model_name}.pkl] done!")

        return model

    def _save_model(self, model):

        model_name = self.args.model_name
        # TODO: path

        save_path = osp.join(self.args.data_path, model_name, 'save')
        os.makedirs(save_path, exist_ok=True)
        
        prefix = ""
        if self.args.dist:
            prefix = f"dist_{prefix}"
        save_path = osp.join(save_path, f'{self.args.exp_id}_{prefix}.pkl')
        
        if model is None:
            return
        if self.args.save_model:
            torch.save(model.state_dict(), save_path)
            
            self.logger.info(f"saving [{save_path}] done!")

        return save_path

        

if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()
    set_seed(cfgs.random_seed)
    # -----  Init ----------
    if cfgs.dist and not cfgs.only_test:
        init_distributed_mode(args=cfgs)
    else:
        # 下面这条语句在并行的时候可能内存泄漏，导致无法停止
        torch.multiprocessing.set_sharing_strategy('file_system')
    rank = cfgs.rank
    # pprint.pprint(cfgs)
    
    writer, logger = None, None
    if rank == 0:
        logger = initialize_exp(cfgs)
        logger_path = get_dump_path(cfgs)
        cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
        if not cfgs.no_tensorboard and not cfgs.only_test:
            writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    cfgs.device = torch.device(cfgs.device)
    
    # print("print c to continue...")
    # -----  Begin ----------
    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger, rank)
    if cfgs.only_test:
        runner.test()
    else:
        runner.run()

    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test and rank == 0:
        writer.close()
        logger.info("done!")
        
    if cfgs.dist and not cfgs.only_test:
        dist.barrier()
        dist.destroy_process_group()
        
        
