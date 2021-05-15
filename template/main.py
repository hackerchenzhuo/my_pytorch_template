import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb

from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path, Top_K_Metric

class Runner:
    def __init__(self, args):
        self.log_dir = get_dump_path(args)
        self.model_dir = os.path.join(self.log_dir, 'model')

        # TODO: init code

        self.args = args

    def run(self):
        # TODO:  for i in epoch:...
        self.train()

        # TODO: if ...
        self.eval()

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
        # TODO: path
        model_name = type(model).__name__
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
    logger = initialize_exp(cfg)
    logger_path = get_dump_path(cfg)

    if not cfg.no_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    # print("print c to continue...")

    torch.cuda.set_device(cfg.gpu)
    runner = Runner(cfg)
    runner.run()

    if not cfg.no_tensorboard:
        writer.close()
    logger.info("done!")