# encoding: utf-8
# Author: Yixuan
# 
# 

import sys
import os
import torch
from tomato.task.base import BaseTask
from tomato.task import FADCLBaseTask
from tomato.task import XentTraining
from tomato.task.fad_task import FADTraining
from tomato.models.base import BaseModel

class LwFTraining(XentTraining):
    # TODO: hardcoded 
    def set_lwf(self):
        self.temperature = 2

    def forward_one_batch(self, task: BaseTask, batch):
        use_real_only = False 
        output_dict = task.model(batch) # [bz, 2]
        loss = task.criterion(batch, output_dict)
        
        if task.prev_model is None:
            batch_logging_output = {
                "loss": loss.item()
            }
            return loss, output_dict, batch_logging_output

        with torch.no_grad():
            prev_output_dict = task.prev_model(batch)
        
        out = output_dict["feats_out"]
        prev_out = prev_output_dict["feats_out"]

        if use_real_only:
            labels = batch["labels"]
            real_out = out[labels == 0]
            real_prev_out = out[labels == 0]
            log_p = torch.log_softmax(real_out / self.temperature, dim=1)
            q = torch.log_softmax(real_prev_out / self.temperature, dim=1)
        else:
            log_p = torch.log_softmax(out / self.temperature, dim=1)
            q = torch.log_softmax(prev_out / self.temperature, dim=1)

        loss += torch.nn.functional.kl_div(log_p, q, reduction='batchmean') 
        
        batch_logging_output = {
            "loss": loss.item()
        }
        return loss, output_dict, batch_logging_output


class FADLwFTask(FADCLBaseTask):

    def __init__(self, train_strategy: LwFTraining):
        super().__init__(train_strategy)
        self.prev_model = None
        self.train_strategy.set_lwf()

    def _clone_model(self, model: BaseModel):
        # Create a new instance of the same class as the original model
        cloned_model = type(model)(model.args)  
        # Copy the parameters
        cloned_model.load_state_dict(model.state_dict())
        return cloned_model

    def continual_learning(self):
        self.load_best_basemodel()
        # exp5, start from task 1
        self.prev_model = self._clone_model(self.model)
        self.criterion.change_weight(0.5, 0.5)
        self.model.train()
        self.criterion.train()
    
        cl_trainsets = self.dataset_keys["cl_train"]
        for idx, cl_train in enumerate(cl_trainsets):
            self.train_one_new_task(cl_train, None)
            self.prev_model = self._clone_model(self.model)  # Update prev_model after each task
