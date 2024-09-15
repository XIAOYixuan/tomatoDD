# encoding: utf-8
# Author: Yixuan
# 
# How to schedule the training process
import os
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from tomato.utils import logger, utils
from tomato.criteria.base import BaseCriterion
from tomato.models.base import BaseModel 

class BaseTrainingStrategy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def save_output(self, task, name, stat_dict: dict):
        pass


    @abstractmethod
    def load_checkpoint(self, task, tag: str, model_dir: str):
        pass

    
    @abstractmethod
    def set_params(self, args):
        # TODO: maybe a new strategy class for optimizer and scheduler?
        # set the parameters for the optimizer and scheduler
        pass

    
    @abstractmethod
    def set_optimizer_scheduler(self, task):
        pass

    
    @abstractmethod
    def step_optimizer(self, task):
        """ The func is also used in train(), needs to be implemented by each task instance
        """
        pass


    @abstractmethod
    def zerograd_optimizer(self, task):
        """ The func is also used in train(), needs to be implemented by each task instance
        """
        pass

    @abstractmethod
    def step_scheduler(self, task, epoch_idx):
        """ The func is also used in train(), needs to be implemented by each task instance
        """
        pass

    
    @abstractmethod
    def forward_for_validation(self, task, data_loader):
        pass


    @abstractmethod
    def forward_one_batch(self, task, batch):
        """ THe function is used in train() and forward_for_validation()
        For train(), only loss and logging_output
        is needed
        For forward_for_validation(), the Task Instance determines the output
        In general, at least the loss and logging_output are
        needed if the task instance decides to use the 
        base class's train() function. For the other two, 
        it's more flexible since they are required to be
        implemented by each task instance
        """
        pass

class BaseTask(ABC):

    def __init__(self, train_strategy: BaseTrainingStrategy):
        self.train_strategy = train_strategy

    def setup(self, cfg: str, 
                 exp: str, 
                 data, 
                 model: BaseModel, 
                 criterion: BaseCriterion, 
                 is_infer = False) -> None:
        self.is_infer = is_infer
        # assign variables
        self.train_args =  utils.config2arg(cfg, "train") # type: argparse.Namespace
        self.data_args = utils.config2arg(cfg, "data") # argparse.Namespace
        self.exp = exp
        # NOTICE: we are hardcoding the data here!
        if len(data) == 2:
            self.dataset, self.data_loader = data
        elif len(data) == 3:
            # TODO: should this be impl in a multi ds base class
            logger.info("Using multiple datasets")
            self.dataset_keys, self.dataset, self.data_loader = data
        self._set_dataloaders()
        self.model = model
        self.criterion = criterion

        # prep tensorboard 
        self.save_dir = f"output/ckpts/{exp}"
        self.writer = SummaryWriter(f"output/tensorboard/{exp}")
        self.ckpt_dir = f"output/ckpts/{exp}"
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # init other variables
        self.optimizer = None
        self.scheduler = None

        utils.check_key(self.train_args, "validate_interval")
        utils.check_key(self.train_args, "log_interval")
        utils.check_key(self.train_args, "lr")

        if not is_infer:
            self._save_config(Path(self.ckpt_dir)/"train.yaml")
        #TODO: steps per epoch is only used in cos scheduler, 
        # it should be moved to the scheduler
        #self.steps_per_epoch = len(self.data_loader["train"])

    def  _save_config(self, save_path):
        config = {
            "data": vars(self.data_args),
            "model": vars(self.model.args),
            "criterion": vars(self.criterion.args),
            "train": vars(self.train_args),
        }
        logger.info(f"Saving config to output dir {save_path}...")
        output_path = Path(save_path)
        with open(output_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        # combine model_args, criterion_args, train_args, data_args
        # and output to 

    def load_checkpoint(self, tag: str, model_dir:str):
        if model_dir is None:
            logger.info("model_dir is None, load from the default dir")
            self.train_strategy.load_checkpoint(self, tag, self.ckpt_dir)
        else:
            logger.info("model_dir is given, load from the given dir")
            self.train_strategy.load_checkpoint(self, tag, model_dir)

    
    def _log_result(self, fig_dir, 
                   avg_logs, avg_count, 
                   other_logs, 
                   step):
        """ items in the avg_logs are averaged over the batch
        """
        for key, item in avg_logs.items():
            item /= avg_count 
            self.writer.add_scalar(f"{fig_dir}/{key}", item, step)
        for key, item in other_logs.items():
            self.writer.add_scalar(f"{fig_dir}/{key}", item, step)

    
    def _update_logging_output(self, logging_output, batch_logging_output):
        for key in batch_logging_output:
            logging_output[key] += batch_logging_output[key]       
    
    
    def _resume_from_last_checkpoint(self):
        # TODO: best way to compute steps per epoch
        self.steps_per_epoch = len(self.cur_train_dataloader)
        total_steps = self.train_args.start_steps
        start_epoch = total_steps // self.steps_per_epoch
        start_steps = total_steps % self.steps_per_epoch

        if start_steps > 0:
            logger.info(f"Resume training from epoch {start_epoch}, step {start_steps}")
            # fast forward dataloader
            train_loader = self.data_loader["train"]
            train_loader.dataset.fast_forward_mode = True
            for _ in range(start_steps):
                next(iter(train_loader))
            train_loader.dataset.fast_forward_mode = False
        else:
            logger.info(f"Start training from epoch {start_epoch}")
        return start_epoch, start_steps


    @abstractmethod
    def _set_dataloaders(self):
        pass


    @abstractmethod
    def validate(self, *args, **kwargs):
        pass


    @abstractmethod
    def validate_and_save(self, *args, **kwargs):
        pass


    @abstractmethod    
    def train(self):
        """ The main training loop. 
        It should:
        1. set model to train mode
        2. set optimizer
        3. init stats such as indices for tensorboard
        4. reload checkpoint if needed
        5. train
        """
        raise NotImplementedError