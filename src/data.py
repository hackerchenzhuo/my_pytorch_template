import torch
import random
import json
import numpy as np
import pdb
import torch.distributed as dist
import os.path as osp
from transformers import BertTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 cfg):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        return example
        # pdb.set_trace()
#         return {
#             'index': index,
#             'question': question,
#             'caption': caption,
#             'target': target,
#             'answer': answer,
#             'fact': fact,
#             'score': scores
#         }
