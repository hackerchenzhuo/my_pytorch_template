
import os
import errno
import torch
import sys
import logging
import json
from pathlib import Path
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch.distributed as dist
import csv
import os.path as osp
import time
import re
import pdb
from torch import nn



def set_optim(opt, model_list, freeze_part=[], accumulation_step=None):
    named_parameters = []
    for model in model_list:
        model_para_train, freeze_layer = [], []
        model_para = list(model.named_parameters())
        for n, p in model_para:
            if not any(nd in n for nd in freeze_part):
                model_para_train.append((n, p))
            else:
                p.requires_grad = False
                freeze_layer.append((n, p))
                
        named_parameters.extend(model_para_train) 
    
    parameters = [
        {'params': [p for n, p in named_parameters], "lr": opt.lr, 'weight_decay': opt.weight_decay}
    ]
    
    if opt.optim == 'adamw':
        # optimizer = optim.AdamW(model.parameters(), lr=opt.lr, eps=opt.adam_epsilon)
        optimizer = optim.AdamW(parameters, lr=opt.lr, eps=opt.adam_epsilon)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=opt.lr)
    
    # 梯度累计
    if accumulation_step is None:
        accumulation_step = opt.accumulation_steps
    # schedule 设定
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        scheduler_steps = opt.total_steps
        # scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps, min_ratio=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_steps/accumulation_step), num_training_steps=int(opt.total_steps/accumulation_step))
    elif opt.scheduler == 'cos':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_steps/accumulation_step), num_training_steps=int(opt.total_steps/accumulation_step))
    
    return optimizer, scheduler


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return 1.0


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        # self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(max(1, self.warmup_steps)) + self.min_ratio

        # if self.fixed_lr:
        #     return 1.0

        return max(0.0,
                   1.0 + (self.min_ratio - 1) * (step - self.warmup_steps) / float(max(1.0, self.scheduler_steps - self.warmup_steps)),
                   )


class Loss_log():
    def __init__(self):
        self.loss = [999999.]
        self.acc = [0.]
        self.flag = 0
        self.token_right_num = []
        self.token_all_num = []
        # 默认不使用top_k acc
        self.use_top_k_acc = 0

    def acc_init(self, topn=[1]):
        self.loss = []
        self.token_right_num = []
        self.token_all_num = []
        self.topn = topn
        self.use_top_k_acc = 1
        self.top_k_word_right = {}
        for n in topn:
            self.top_k_word_right[n] = []
    
    def get_token_acc(self):
        # 返回list
        if len(self.token_all_num) == 0:
            return 0.
        elif self.use_top_k_acc == 1:
            res = []
            for n in self.topn:
                res.append(round((sum(self.top_k_word_right[n]) / sum(self.token_all_num)) * 100 , 3))
            return res
        else:
            return [sum(self.token_right_num)/sum(self.token_all_num)]
    
    def update_token(self, token_num, token_right):
        # 输入是list文件
        self.token_all_num.append(token_num)
        if isinstance(token_right, list):
            for i, n in enumerate(self.topn):
                self.top_k_word_right[n].append(token_right[i])
        self.token_right_num.append(token_right)
        
        
        
    def update(self, case):
        self.loss.append(case)

    def update_acc(self, case):
        self.acc.append(case)

    def get_loss(self):
        return self.loss[-1]

    def get_acc(self):
        return self.acc[-1]

    def get_min_loss(self):
        return min(self.loss)

    def get_loss(self):
        if len(self.loss) == 0:
            return 500.
        return mean(self.loss)
    
    def early_stop(self):
        # min_loss = min(self.loss)
        if self.loss[-1] > min(self.loss):
            self.flag += 1
        else:
            self.flag = 0

        if self.flag > 1000:
            return True
        else:
            return False

    def torch_accuracy(output, target, topk=(1,)):
        '''
        param output, target: should be torch Variable
        '''
        # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
        # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
        # print(type(output))

        topn = max(topk)
        batch_size = output.size(0)

        _, pred = output.topk(topn, 1, True, True) # 返回(values,indices）其中indices就是预测类别的值，0为第一类
        pred = pred.t() # torch.t()转置，既可得到每一行为batch最好的一个预测序列

        is_correct = pred.eq(target.view(1, -1).expand_as(pred))

        ans = []
        ans_num = []
        for i in topk:
            # is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
            is_correct_i = is_correct[:i].contiguous().view(-1).float().sum(0, keepdim=True)
            ans_num.append(int(is_correct_i.item()))
            ans.append(is_correct_i.mul_(100.0 / batch_size))

        return ans, ans_num
