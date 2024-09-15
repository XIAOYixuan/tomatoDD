# encoding: utf-8
# Author: Yixuan
# 
# 

import os
import argparse
import random
from datetime import datetime
import torch
from tqdm import tqdm
from pathlib import Path

from tomato.utils import logger
import tomato.utils
from tomato.utils import utils
from .base import BaseTask, BaseTrainingStrategy
import tomato.train_util as train_util
from tomato.criteria import BuiltInCriterion
from torch.utils.data import DataLoader

import numpy as np
from torch.profiler import profile, ProfilerActivity
import contextlib

@contextlib.contextmanager
def pytorch_profiler_context(filename="profiler_results"):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=True) as p:
        yield p
    p.export_chrome_trace(f"{filename}.json")  # Export results for visualization

from abc import ABC, abstractmethod
class FADTraining(BaseTrainingStrategy):

    @abstractmethod
    def compute_eer(self, labels, scores):
        pass

    
    @abstractmethod
    def get_prediction(self, score_tensor):
        pass


class XentTraining(FADTraining):

    def save_output(self, 
                    task: BaseTask,
                    name: str, stat_dict: dict):
        logger.info(f"Saving to {task.ckpt_dir}")
        mdl_path = os.path.join(task.ckpt_dir, f"{name}.mdl")
        torch.save(task.model.state_dict(), mdl_path)
        stat_path = os.path.join(task.ckpt_dir, f"{name}.stat")
        with open(stat_path, "w") as f:
            for stat_name, stat_value in stat_dict.items():
                f.write(f"{stat_name}: {stat_value}\n")


    def set_params(self, args):
        logger.info(f"Setting params for {self.__class__.__name__}")
        utils.check_key(args, "start_steps")
        
        utils.check_key(args, "max_steps")
        utils.check_key(args, "max_epoch")

        # parameters for Adam
        utils.check_key(args, "lr")
        utils.check_key(args, "beta1")
        utils.check_key(args, "beta2")
        utils.check_key(args, "weight_decay")

        # parameters for cosince scheduler
        # self.check_key("lr_min")
        # parameters for StepLR
        utils.check_key(args, "lr_gamma")
        utils.check_key(args, "lr_step_size")
    

    def set_optimizer_scheduler(self, task: BaseTask):
        task.model_optimizer = torch.optim.Adam(task.model.parameters(),
                                                lr=task.train_args.lr,
                                                betas=(task.train_args.beta1, task.train_args.beta2),
                                                weight_decay=task.train_args.weight_decay)
        # the following is a bit too complicated 
        """
        self.steps_per_epoch = len(self.cur_train_dataloader)
        total_steps = self.train_args.max_epoch * self.steps_per_epoch
        self.model_scheduler = train_util.CosineAnnealingLR(self.train_args, 
                                                            total_steps, 
                                                            self.model_optimizer)

        """
        from tomato.train_util import StepLR
        task.model_scheduler = StepLR(task.train_args, 
                                      task.model_optimizer)


    def step_optimizer(self, task: BaseTask):
        task.model_optimizer.step()
    

    def zerograd_optimizer(self, task: BaseTask):
        task.model_optimizer.zero_grad()


    def step_scheduler(self, task: BaseTask, epoch_idx):
        task.model_scheduler.step(epoch_idx)


    def forward_for_validation(self, task: BaseTask, data_loader):
        logging_output = task.criterion.zero_logging_output()
        uttids, all_scores, all_labels = [], [], []
        with torch.no_grad():
            pbar = tqdm(data_loader, total=len(data_loader))
            #logger.info("Forwarding for validation----------")
            #logger.info(f"type(dataloader): {type(data_loader)}")
            for batch in pbar:
                _, output_dict, batch_logging_output = self.forward_one_batch(task, batch)
                task._update_logging_output(logging_output, batch_logging_output)
                outputs = output_dict["feats_out"] # [bz, 2]
                #outputs = outputs[:, 1]
                uttids.extend(batch["uttids"])
                all_scores.append(outputs.data.cpu())
                all_labels.append(batch["labels"].data.cpu())
        return all_labels, all_scores, uttids, logging_output

    
    def forward_one_batch(self, task: BaseTask, batch):
        output_dict = task.model(batch) # [bz, 2]

        loss = task.criterion(batch, output_dict)
        batch_logging_output = {
            "loss": loss.item()
        }
        return loss, output_dict, batch_logging_output
    
    def forward_one_batch_lwf_neg(self, task: BaseTask, batch):
        output_dict = task.model(batch)
        loss = task.criterion(batch, output_dict)

        prev_model = task.prev_model
        with torch.no_grad():
            prev_output_dict = prev_model(batch)
        
        out = output_dict["feats_out"]
        prev_out = prev_output_dict["feats_out"]

        labels = batch["labels"]
        fake_out = out[labels == 1]
        fake_prev_out = prev_out[labels == 1]
        if fake_out.shape[0] != 0:
            log_p = torch.log_softmax(fake_out / 2.0, dim=1)
            q = torch.log_softmax(fake_prev_out / 2.0, dim=1)
            loss += torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        
        batch_logging_output = {
            "loss": loss.item()
        }
        return loss, output_dict, batch_logging_output

    def compute_eer(self, labels, scores):
        # if the score is 1-d, use it directly
        if scores.shape[1] == 1:
            return train_util.compute_eer(labels, scores[:, 0])
        else:
            return train_util.compute_eer(labels, scores[:, 1])


    def load_checkpoint(self, task: BaseTask, tag: str, model_dir: str):
        model_path = os.path.join(model_dir, f"{tag}.mdl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        logger.info(f"Loading the {tag} model from {model_dir}/{tag}.mdl")
        task.model.load_checkpoint(model_path)


    def get_prediction(self, score_tensor):
        if len(score_tensor.shape) == 1:
            # sigmoid of numpy array
            sigmoid_score = 1 / (1 + np.exp(score_tensor))
            return sigmoid_score > 0.5
        else:
            # score_tensor: [bz, 2]
            return np.argmax(score_tensor, axis=1)
    

class OCSoftmaxTraining(FADTraining):


    def save_output(self, task: BaseTask, name: str, stat_dict: dict):
        logger.info(f"Saving to {task.ckpt_dir}")
        mdl_path = os.path.join(task.ckpt_dir, f"{name}.mdl")
        torch.save(task.model.state_dict(), mdl_path)
        stat_path = os.path.join(task.ckpt_dir, f"{name}.stat")
        with open(stat_path, "w") as f:
            for stat_name, stat_value in stat_dict.items():
                f.write(f"{stat_name}: {stat_value}\n")

        # bc criterion is also trainable
        criterion_path = os.path.join(task.ckpt_dir, f"{name}.criterion.mdl")
        torch.save(task.criterion.state_dict(), criterion_path)


    def set_params(self, args):
        logger.info(f"Setting params for {self.__class__.__name__}")
        utils.check_key(args, "start_steps")
        
        utils.check_key(args, "max_steps")
        utils.check_key(args, "max_epoch")

        # parameters for Adam
        utils.check_key(args, "lr")
        utils.check_key(args, "beta1")
        utils.check_key(args, "beta2")
        utils.check_key(args, "weight_decay")

        # parameters for StepLR
        utils.check_key(args, "lr_gamma")
        utils.check_key(args, "lr_step_size")


    def set_optimizer_scheduler(self, task: BaseTask):
        
        task.model_optimizer = torch.optim.Adam(
            task.model.parameters(),
            lr = task.train_args.lr,
            betas = (task.train_args.beta1, task.train_args.beta2),
            eps = 1e-8, 
            weight_decay = task.train_args.weight_decay)
        
        task.criterion_optimizer = torch.optim.SGD(
            task.criterion.parameters(),
            lr = task.train_args.lr,
        )

        from tomato.train_util import StepLR
        task.model_scheduler = StepLR(task.train_args, task.model_optimizer)
        task.criterion_scheduler = StepLR(task.train_args, task.criterion_optimizer)


    def step_optimizer(self, task: BaseTask):
        task.model_optimizer.step()
        task.criterion_optimizer.step()
    

    def zerograd_optimizer(self, task: BaseTask):
        task.model_optimizer.zero_grad()
        task.criterion_optimizer.zero_grad()


    def step_scheduler(self, task: BaseTask, epoch_idx: int):
        task.model_scheduler.step(epoch_idx)
        task.criterion_scheduler.step(epoch_idx)


    def forward_for_validation(self, task: BaseTask, data_loader):
        """ Forward the whole dataset for evaluation.
        """
        logging_output = task.criterion.zero_logging_output()
        uttids, all_scores, labels = [], [], []
        with torch.no_grad():
            pbar = tqdm(data_loader, total=len(data_loader))
            for batch in pbar:
                loss, scores, batch_logging_output = self.forward_one_batch(task, batch)
                task._update_logging_output(logging_output, batch_logging_output)
                uttids.extend(batch["uttids"])
                all_scores.append(scores.data.cpu())
                labels.append(batch["labels"].data.cpu())

        return labels, all_scores,  uttids, logging_output
    
    
    def forward_one_batch(self, task: BaseTask, batch):
        out_dict = task.model(batch)
        loss, output_score, cur_logging_output = task.criterion(batch, out_dict)
        #TODO: make output_score the last return value
        return loss, output_score, cur_logging_output

    
    def compute_eer(self, labels, scores):
        """
         reverse the scores, because the target class, aka the fake audios,
         would have a smaller scores based on cos similarity score
         but compute_eer would expect a higher score for the target class
         return: eer, thresh
        """
        return train_util.compute_eer(labels, -scores)
    

    def load_checkpoint(self, task, tag: str, model_dir: str):
        mdl_path = os.path.join(model_dir, f"{tag}.mdl")
        logger.info(f"Loading the {tag} base model from {mdl_path}")
        task.model.load_checkpoint(mdl_path)

        criterion_path = os.path.join(model_dir, f"{tag}.criterion.mdl")
        logger.info(f"Loading the {tag} base model from {criterion_path}")
        task.criterion.load_checkpoint(criterion_path)
    

    def get_prediction(self, score_tensor):
        # TODO: naitve method now, if cos similarity score is > 0, then it's a target class (real audio)
        pred = score_tensor < 0
        return pred


class FADBaseTask(BaseTask):
    """ A FADBaseTask is a binary prediction task.
    Other FAD Task might include a continual-learning based FAD task, or 
    attacker attribution task.
    """

    def setup(self, cfg: str, exp: str, data, 
              model, criterion, is_infer=False) -> None:
        super().setup(cfg, exp, data, model, criterion, is_infer)

        utils.check_key(self.train_args, "start_steps")
        utils.check_key(self.train_args, "max_steps")
        utils.check_key(self.train_args, "max_epoch")
        
        # train_args type: argparse.Namespace
        # ensure that the key start_steps in train_args
        self.train_strategy.set_params(self.train_args)

        # for LwF, load the previous best model
        # lfcc_aug_both = Path("/mount/arbeitsdaten54/projekte/deepfake/fad/robustFAD/Tomato/output/ckpts/lfcc_aug_both/")
        #self.train_strategy.load_checkpoint(self, "best", lfcc_aug_both)
        # cloned_model = type(self.model)(self.model.args)  
        # Copy the parameters
        # cloned_model.load_state_dict(self.model.state_dict())
        # self.prev_model = cloned_model

    
    def _set_dataloaders(self):
        logger.info(f"Setting dataloader for {self.__class__.__name__}")
        if self.is_infer:
            self.cur_test_dataloader = self.data_loader["test"]
        else:
            self.cur_train_dataloader = self.data_loader["train"]
            self.cur_dev_dataloader = self.data_loader["dev"]


    def _getall_scores_labels(self, data_loader):
        """ Forward the whold dataset for evaluation.
        Output tensor will be transferred to cpu
        """
        all_labels, all_scores, uttids, logging_output  = self.train_strategy.forward_for_validation(self, data_loader)
        all_labels = torch.cat(all_labels, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = all_labels.data.cpu().numpy()
        all_scores = all_scores.data.cpu().numpy()
        return uttids, all_labels, all_scores, logging_output

    def _compute_eer_by_origin_ds(self, uttids, labels, scores, dataset):
        ds2labels = {}
        ds2scores = {}

        for idx, uttid in enumerate(uttids):
            info = dataset.get_info(uttid)
            origin_ds = info["origin_ds"]
            if origin_ds not in ds2labels:
                ds2labels[origin_ds] = []
                ds2scores[origin_ds] = []
            label = labels[idx]
            score = scores[idx]
            ds2labels[origin_ds].append(label)
            ds2scores[origin_ds].append(score)
        all_keys = ds2labels.keys()
        all_keys = sorted(all_keys)
        for ds in all_keys:
            labels = np.array(ds2labels[ds])
            scores = np.array(ds2scores[ds])
            eer, _ = self.train_strategy.compute_eer(labels, scores)
            logger.info(f"Validation EER of {ds}: {eer}")
            

    def validate(self, num_steps):
        """ Mainly used during training, to select the best model.
        It always use EER as the default metric. If ACC is needed, a few changes needs to be added to the logging_output
        """
        if self.cur_dev_dataloader is None:
            logger.info("No dev set, skip validation, no best model will be saved")
            # return inf
            return float("inf")
        logger.info("Validating...")
        self.model.eval()
        self.criterion.eval()
        
        dev_loader = self.cur_dev_dataloader 
        dev_dataset = dev_loader.dataset
        uttids, all_labels, all_scores, logging_output = self._getall_scores_labels(dev_loader)
        self._compute_eer_by_origin_ds(uttids, all_labels, all_scores, dev_dataset)
        #logger.info(f"all scores dim: {all_scores.shape}")

        # print("label", labels) # [0 1 1]
        # print("all_scores", all_scores) # negative vlaues

        eer, thresh = self.train_strategy.compute_eer(all_labels, all_scores)
        logger.info(f"Validation finished EER: {eer}")
        self._log_result("Validation", 
                        logging_output, len(dev_loader), 
                        {"eer": eer, "threshold": thresh}, 
                        num_steps)
        self.model.train()
        self.criterion.train()

        return eer

    
    def validate_and_save(self, best_err, cur_step):
        """ 
        Validate the model and save the output if the 
        error is lower than the best error
        """
        err = self.validate(cur_step)
        if err < best_err:
            best_err = err
            stat_dict = {
                "eer": best_err,
                "cur_step": cur_step
            }
            self.train_strategy.save_output(self, "best", stat_dict)
        return best_err


    def train_one_batch(self, 
                        batch_idx, batch,
                        cur_step, 
                        best_err, 
                        logging_output):
        cur_step += 1 # index start from 1
        should_stop = True 

        if cur_step > self.train_args.max_steps:
            best_err = self.validate_and_save(best_err, cur_step)
            return cur_step, should_stop, best_err, logging_output
        
        if cur_step % self.train_args.validate_interval == 0:
            # TODO; float comparison wrong
            best_err = self.validate_and_save(best_err, cur_step)

        if cur_step % self.train_args.log_interval == 0:
            self._log_result("Train", 
                            logging_output, batch_idx + 1,
                            {"model_lr": self.model_scheduler.get_last_lr()}, 
                            cur_step)
        #print("======= forward_one_batch ========") 
        self.train_strategy.zerograd_optimizer(self)
        loss, _, batch_logging_output = self.train_strategy.forward_one_batch(self, batch)
        #print("batch_idx", batch_idx)
        #print("batch_logging_output", batch_logging_output)
        self._update_logging_output(logging_output, batch_logging_output)
        #print("logging_output", logging_output)
        #print("------------- end forward_one_batch ---------------")
        loss.backward()

        self.train_strategy.step_optimizer(self)
        return cur_step, not should_stop, best_err, logging_output


    def train_one_epoch(self, 
                        best_err, 
                        cur_step):
        should_stop = False
        train_loader = self.cur_train_dataloader
        
        # retrieve the stats for tensorboard
        logging_output = self.criterion.zero_logging_output()

        # only works when accumulate_steps == 1
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch_idx, batch in enumerate(pbar):
            cur_step, should_stop, best_err, logging_output = self.train_one_batch(batch_idx, batch, 
                                 cur_step, 
                                 best_err, 
                                 logging_output)
            if should_stop:
                return cur_step, should_stop, best_err
        return cur_step, should_stop, best_err 


    def train(self):
        # Start profiling here
        self.model.train()
        self.criterion.train()

        self.train_strategy.set_optimizer_scheduler(self)
        start_epoch, cur_step = self._resume_from_last_checkpoint()

        if cur_step > 0:
            best_err = self.validate(cur_step)
        else:
            best_err = float("inf")
        for epoch_idx in range(start_epoch, self.train_args.max_epoch):
            logger.info(f"Epoch {epoch_idx}:")
            cur_step, should_stop, best_err = self.train_one_epoch(best_err, cur_step)
            # TODO: should we start to step scheduler before traiing, otherwise the initial lr is not update to date
            self.train_strategy.step_scheduler(self, epoch_idx)
            self.train_strategy.save_output(self, 
                                            "last", 
                                            {"cur_step": cur_step})
            if should_stop:
                logger.info("Early stop triggered")
                break
        
            best_err = self.validate_and_save(best_err, cur_step)
        logger.info("Training finished")
        self.writer.close() 


    def _get_infer_csv_name(self, args: argparse.Namespace, metrics: dict):
        """ Used in infer() to generate the csv name
        The csv name includes the tag_name(defined by the user, similar to exp name), ckpt_name, config_path, metrics
        """
        output_dir = args.output
        tag_name = args.tag

        ckpt_path = args.ckpt_dir + "=" + args.ckpt_tag
        config_path = args.config
        ckpt_name = ckpt_path.replace("/", "=")
        config_path = config_path.replace("/", "=")

        print("metrics", metrics)
        metric_str = "_".join([f"{key}{value:.2f}" for key, value in metrics.items()])
        csv_name = f"{output_dir}/{tag_name}_{ckpt_name}_{config_path}_{metric_str}.csv"
        return csv_name


    def infer(self, args: argparse.Namespace):
        output_dir = Path(args.output)
        cfg_dir = Path(args.output)/"config"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cfg_dir, exist_ok=True)
        self._save_config(cfg_dir / f"{args.tag}.yaml")

        self.model.eval()
        self.criterion.eval()
        
        # TODO: bug here, we also need logging_output
        logger.info(f"Infering on {args.split}")
        test_loader = self.data_loader[args.split]
        uttids, all_labels, all_scores, _  = self._getall_scores_labels(test_loader)
        cur_ds = test_loader.dataset
        self._compute_eer_by_origin_ds(uttids, all_labels, all_scores, cur_ds)

        eer, threshold = self.train_strategy.compute_eer(all_labels, all_scores)
        alldata = []
        if all_scores.ndim == 1:
            scores = all_scores
        elif all_scores.ndim == 2 and all_scores.shape[1] == 2:
            scores = all_scores[:, 1]
        elif all_scores.ndim == 2 and all_scores.shape[1] == 1:
            # TODO: check the output of all labels, it should be a 1-d tensor
            scores = all_scores[:, 0]
        else:
            raise ValueError(f"The shape of all_scores {all_scores} is not correct")
        
        for uttid, score, label in zip(uttids, scores, all_labels):
            alldata.append([uttid, score, label])
        # check if preds and all_labels have the same shape
        acc = np.mean((scores>threshold)== all_labels) 
        #print("all_scores[:1].shape", all_scores[:, 1].shape)
        acc *= 100
        eer *= 100
        output_dict = {
            "acc": acc, 
            "eer": eer
        }

        csv_path = self._get_infer_csv_name(args, output_dict)
        
        import pandas as pd
        datadf = pd.DataFrame(alldata, columns=["uttid", "prediction", "label"])
        datadf.to_csv(csv_path, index=False)
        return output_dict
    
    
class FADCLBaseTask(FADBaseTask):

    def __init__(self, train_strategy: FADTraining):
        super().__init__(train_strategy)

    def setup(self, cfg_path, exp, data, model, criterion, is_infer=False) -> None:
        super().setup(cfg_path, exp, data, model, criterion, is_infer)
        # we need to store multiple model during training, the base one
        # the continual one, use the prefix to ditinguish them
        
        # The following var is used for tensorboard writer
        # "task" refers to a task in continual learning
        # each task might have its own distribution, could be a dataset name
        # or the model that is used to train the task
        self.cur_task_tag = "base"
        # each cl result would be put under a new output dir 
        # which is under the base_ckpt_dir 
        self.base_ckpt_dir = self.ckpt_dir

    def _set_dataloaders(self):
        """ By default, it uses the base training't setting to 
        init cur train and dev dataloader 
        """
        if self.is_infer:
            return
        train_key = self.dataset_keys["base_train"][0]
        self.cur_train_dataloader = self.data_loader[train_key]
        if "base_dev" in self.dataset_keys:
            dev_key = self.dataset_keys["base_dev"][0]
            self.cur_dev_dataloader = self.data_loader[dev_key]
        else:
            self.cur_dev_dataloader = None


    def train(self):
        if not self.train_args.has_best_model:
            self.train_base_model()
        self.continual_learning()


    def train_base_model(self):
        # prep weights
        # TODO: Hard coded here, should be changed in the future
        weight_neg = (25380 - 2580)/25380
        weight_pos = 2580/25380
        self.criterion.change_weight(weight_pos, weight_neg)
        self.model.train()
        self.criterion.train()
        
        self.train_strategy.set_optimizer_scheduler(self) 
        start_epoch, cur_step = self._resume_from_last_checkpoint()

        if cur_step > 0:
            best_err = self.validate(cur_step)
        else:
            best_err = float("inf")
        for epoch_idx in range(start_epoch, self.train_args.max_epoch):
            logger.info(f"Epoch {epoch_idx}:")
            self.train_strategy.step_scheduler(self, epoch_idx)
            cur_step, should_stop, best_err = self.train_one_epoch(best_err, cur_step)
            self.train_strategy.save_output(self, 
                                            "last", 
                                            {"cur_step": cur_step})
            if should_stop:
                logger.info("Early stop triggered")
                break
        
            best_err = self.validate_and_save(best_err, cur_step)
        logger.info("Basemodel Training finished")


    def load_best_basemodel(self):
        self.train_strategy.load_checkpoint(self, "best", self.base_ckpt_dir)

    
    def _log_result(self, fig_dir, avg_logs, avg_count, other_logs, step):
        for key, item in avg_logs.items():
            item /= avg_count 
            self.writer.add_scalar(f"{self.cur_task_tag}/{fig_dir}_{key}", item, step)
        for key, item in other_logs.items():
            self.writer.add_scalar(f"{self.cur_task_tag}/{fig_dir}_{key}", item, step)


    def train_one_new_task(self, task_train_tag: str, task_dev_tag: str):
        cl_train = task_train_tag
        cl_dev = task_dev_tag
        logger.info(f"now training on {cl_train}")
        self.cur_task_tag = cl_train
        self.cur_train_dataloader = self.data_loader[cl_train]
        if cl_dev is None:
            self.cur_dev_dataloader = None
        else:
            self.cur_dev_dataloader = self.data_loader[cl_dev]

        self.ckpt_dir = os.path.join(self.base_ckpt_dir, cl_train)
        #if os.path.exists(self.ckpt_dir):
        #    logger.info(f"{cl_train} has been trained, skip")
        #    return
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.train_args.lr = self.train_args.finetune_lr
        self.train_strategy.set_optimizer_scheduler(self)
        # TODO: support resume from the previous best model
        #start_epoch, cur_step = self._resume_from_last_checkpoint()
        
        best_err = float("inf")
        cur_step = 0
        for epoch_idx in range(0, self.train_args.finetune_max_epoch):
            logger.info(f"Training on {cl_train}, Epoch {epoch_idx}:")
            self.train_strategy.step_scheduler(self, epoch_idx)
            cur_step, should_stop, best_err = self.train_one_epoch(best_err, cur_step)
            self.train_strategy.save_output(self, 
                                            f"last", {"cur_step": cur_step})
            self.train_strategy.save_output(self, 
                                            f"epoch{epoch_idx}", {"cur_step": cur_step})
            if should_stop:
                logger.info("Early stop triggered")
                break
        
            best_err = self.validate_and_save(best_err, cur_step)
        logger.info(f"Training for {cl_train} finished")
        # self.avg_infer(idx+1)


    def continual_learning(self):
        self.load_best_basemodel()
        #self.avg_infer(0)
        # bc we do upsample, always use 0.5 vs 0.5 ratio
        self.criterion.change_weight(0.5, 0.5)
        self.model.train()
        self.criterion.train()


        cl_trainsets = self.dataset_keys["cl_train"]
        # we don't save the best model for the cl task
        # becase each task lasts for at most five epoch, and we store
        # the model for each task
        # cl_devsets = self.dataset_keys["cl_dev"]
        # sort the keys to make sure the order is the same
        # cl_trainsets.sort()
        # cl_devsets.sort()

        # TODO: resume from one model
        #for idx, (cl_train, cl_dev) in enumerate(zip(cl_trainsets, cl_devsets)):
        #    self.train_one_new_task(cl_train, cl_dev)
        for idx, cl_train in enumerate(cl_trainsets):
            self.train_one_new_task(cl_train, None)


    def _infer(self, split):
        test_keys = self.dataset_keys[split]
        eers = []
        for test_dataset in test_keys:
            #if "ASV" in test_dataset:
            #    eers.append((test_dataset, float("nan"), float("nan"), None, None, None))
            #    continue
            logger.info("Infering on %s" % test_dataset)
            test_dataloader = self.data_loader[test_dataset]
            uttids, all_labels, all_scores, _ = self._getall_scores_labels(test_dataloader)
            eer, thresh = self.train_strategy.compute_eer(all_labels, all_scores)
            logger.info(f"EER of {test_dataset}: {eer:.2f} at threshold {thresh:.2f}")
            eers.append((test_dataset, eer, thresh, uttids, all_scores, all_labels))
            # break
        return eers


    def avg_infer(self, task_id):
        self.model.eval()
        self.criterion.eval()

        eers = self._infer()
        avg_eer = 0
        for test_dataset, eer, thresh, uttids, pred, tgt in eers:
            self.writer.add_scalar(f"AvgInfer/{test_dataset}", eer, task_id)
            avg_eer += eer
        avg_eer /= len(eers)
        self.writer.add_scalar(f"AvgInfer/average", eer, task_id)
        
        self.model.train()
        self.criterion.train()


    def infer(self, args: argparse.Namespace):
        # called by the infer.py
        output_dir = Path(args.output)
        cfg_dir = Path(output_dir)/"config"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        if not cfg_dir.exists():
            cfg_dir.mkdir(parents=True)

        self.model.eval()
        self.criterion.eval()

        eers = self._infer(args.split)

        avg_eer = 0
        
        # usually one output dir represents one exp
        # the tag is used to distinguish the different infer results
        # e.g., the performance on different datasets
        base_tag = args.tag 
        # save infer config
        summary = []
        for test_dataset, eer, thresh, uttids, scores, tgt in eers:
            preds = self.train_strategy.get_prediction(scores) 
            csv_data = []
            for uttid, pred, label in zip(uttids, preds, tgt):
                csv_data.append([uttid, pred, label])
            eer *= 100
            args.tag = base_tag + "_" + test_dataset
            self._save_config(Path(cfg_dir) / f"{args.tag}.yaml")
            csv_path = self._get_infer_csv_name(args, {"eer": eer, "thresh": thresh})
            import pandas as pd
            datadf = pd.DataFrame(csv_data, columns=["uttid", "prediction", "label"])
            datadf.to_csv(csv_path, index=False)
            summary.append([test_dataset, eer])
        
            avg_eer += eer

        avg_eer /= len(eers)
        summary_dir = os.path.join(output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        summary_csv = os.path.join(summary_dir, f"{base_tag}_summary.csv")
        sum_df = pd.DataFrame(summary, columns=["dataset", "eer"])
        sum_df.to_csv(summary_csv, index=False)
        # write the summary resutls to csv
        return {"avg_eer": avg_eer}



class IncreFADMemoryReplayBase(FADCLBaseTask):

    def setup(self, cfg_path, exp, data, model, criterion, is_infer=False) -> None:
        super().setup(cfg_path, exp, data, model, criterion, is_infer)
        self.replay_dataloader = []
        self.max_replay_batches = self.train_args.max_replay_batches
    
    def train_one_epoch(self, 
                        best_err, 
                        cur_step):
        should_stop = False
        train_loader = self.cur_train_dataloader
        
        # retrieve the stats for tensorboard
        logging_output = self.criterion.zero_logging_output()

        # only works when accumulate_steps == 1
        pbar = tqdm(train_loader, total=len(train_loader))
        final_batch_idx = None
        for batch_idx, batch in enumerate(pbar):
            final_batch_idx = batch_idx
            cur_step, should_stop, best_err, logging_output = self.train_one_batch(batch_idx, batch, 
                                 cur_step, 
                                 best_err, 
                                 logging_output)
            if should_stop:
                break

        # do replay
        for idx, dataloader in enumerate(self.replay_dataloader):
            logger.info(f"Memory replaying on {idx+1} previous set")
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= self.max_replay_batches:
                    break
                cur_step, should_stop, best_err, logging_output = self.train_one_batch(final_batch_idx + batch_idx, batch, 
                                 cur_step, 
                                 best_err, 
                                 logging_output)
        return cur_step, should_stop, best_err 
    

    def train_one_new_task(self, task_train_tag: str, task_dev_tag: str):
        super().train_one_new_task(task_train_tag, task_dev_tag)
        self.replay_dataloader.append(self.cur_train_dataloader)


    def continual_learning(self):
        self.replay_dataloader.append(self.cur_train_dataloader)
        return super().continual_learning()


class FADMemoryReplayBase(FADCLBaseTask):

    def setup(self, cfg_path, exp, data, model, criterion, is_infer=False) -> None:
        super().setup(cfg_path, exp, data, model, criterion, is_infer)
        self.replay_dataloader = []
        self.buffer_size = 256
        self.buffer = []
        self.cur_replay_buffer = None
    
    def train_one_epoch(self, 
                        best_err, 
                        cur_step):
        should_stop = False
        train_loader = self.cur_train_dataloader
        
        # retrieve the stats for tensorboard
        logging_output = self.criterion.zero_logging_output()

        # only works when accumulate_steps == 1
        pbar = tqdm(train_loader, total=len(train_loader))
        final_batch_idx = None
        for batch_idx, batch in enumerate(pbar):
            final_batch_idx = batch_idx
            cur_step, should_stop, best_err, logging_output = self.train_one_batch(batch_idx, batch, 
                                 cur_step, 
                                 best_err, 
                                 logging_output)
            if should_stop:
                break

        #print(f"==== len cur replay buffer: {len(self.cur_replay_buffer['uttids'])} ====")
        cur_step, should_stop, best_err, logging_output = self.train_one_batch(
            final_batch_idx + 1, 
            self.cur_replay_buffer, 
            cur_step, 
            best_err, 
            logging_output)
        return cur_step, should_stop, best_err 

    def pre_replay_sample(self):
        total_tasks = len(self.replay_dataloader)
        if total_tasks == 0:
            return None
        #print(f"====== total_tasks: {total_tasks} ======")
        sample_per_task = self.buffer_size // total_tasks

        cur_buffer = self._get_sample_from_latest(self.replay_dataloader[-1], sample_per_task)
        self._remove_sample_from_previous(sample_per_task)
        self.buffer.append(cur_buffer)
        return self._group_samples()
    
    def _get_sample_from_latest(self, dataloader, sample_per_task):
        # Get one batch from the dataloader
        remain_num = sample_per_task
        all_selected = {'uttids': [], 'feats': [], 'labels': [], 'origin_ds': [], 'speakers': [], 'attackers': []}
        
        for batch in dataloader:
            actual_sample = min(remain_num, len(batch['uttids']))
            # Select random samples from the batch
            selected_indices = random.sample(range(len(batch['uttids'])), actual_sample)
            remain_num -= actual_sample
            for key in all_selected.keys():
                all_selected[key].extend([batch[key][i] for i in selected_indices])
            if remain_num == 0:
                break
        return all_selected
    
    def _remove_sample_from_previous(self, sample_per_task):
        for idx, one_task_buffer in enumerate(self.buffer):
            cur_num_sample = len(one_task_buffer['uttids'])
            remove_num_sample = max(0, cur_num_sample - sample_per_task)
            if remove_num_sample == 0:
                continue
            # randomly select sample_per_task
            new_buffer = {'uttids': [], 'feats': [], 'labels': [], 'origin_ds': [], 'speakers': [], 'attackers': []}
            selected_indices = random.sample(range(cur_num_sample), sample_per_task)
            for key in new_buffer.keys():
                new_buffer[key] = [one_task_buffer[key][i] for i in selected_indices]
            #print("need to remove", remove_num_sample, "and select ", sample_per_task, "new buffer size is ", len(new_buffer['uttids']))
            self.buffer[idx] = new_buffer

    def _group_samples(self):
        all_selected = {'uttids': [], 'feats': [], 'labels': [], 'origin_ds': [], 'speakers': [], 'attackers': []}
        for one_task_buffer in self.buffer:
            #print(f"========== len of one_task_buffer: {len(one_task_buffer['uttids'])} ==========")
            for key in all_selected.keys():
                all_selected[key].extend(one_task_buffer[key])

        # Shuffle the combined buffer to mix samples from different tasks
        combined_lengths = len(all_selected['uttids'])
        indices = list(range(combined_lengths))
        random.shuffle(indices)

        shuffled_batch = {key: [all_selected[key][i] for i in indices] for key in all_selected.keys()}
        shuffled_batch["feats"] = torch.stack(shuffled_batch["feats"])
        shuffled_batch["labels"] = torch.stack(shuffled_batch["labels"])
        #print(shuffled_batch)
        #print(f"len of shuffled_batch: {len(shuffled_batch['uttids'])}")
        return shuffled_batch

    def train_one_new_task(self, task_train_tag: str, task_dev_tag: str):
        super().train_one_new_task(task_train_tag, task_dev_tag)
        self.replay_dataloader.append(self.cur_train_dataloader)
        self.cur_replay_buffer = self.pre_replay_sample()

    def continual_learning(self):
        self.replay_dataloader.append(self.cur_train_dataloader)
        self.cur_replay_buffer = self.pre_replay_sample()
        return super().continual_learning()

