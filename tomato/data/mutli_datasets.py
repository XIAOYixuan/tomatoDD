# encoding: utf-8
# Author: Yixuan
# 
# Dataset for continuous learning
# TODO: The whole file needs to be refactored

import os
import argparse
import copy

import numpy as np

from pathlib import Path

from tomato.utils import logger
from abc import ABC, abstractmethod

class MutliDSBase(ABC):

    def __init__(self, args: argparse.Namespace, split: str, train_model: bool) -> None:
        self.split = split
        self.train_mode = train_model
        self.datasets = {}
        self.order_list = []
        self.args = args


    def get_datasets(self) -> dict:
        if len(self.datasets) == 0:
            raise ValueError("datasets is empty")
        return self.datasets


    def get_order_list(self) -> list:
        if len(self.order_list) == 0:
            raise ValueError("order_list is empty")
        uniq_order = set(self.order_list)
        uniq_keys = set(self.datasets.keys())
        # compare set, set content should be the same
        if uniq_keys != uniq_order:
            print("uniq_order: ", uniq_order)
            print("uniq_keys: ", uniq_keys)
            raise ValueError("order_list and datasets keys are not the same")
        return self.order_list


class GeneralMultiDS(MutliDSBase):

    def __init__(self, args: argparse.Namespace, split: str, train_model: bool) -> None:
        super().__init__(args, split, train_model)
        self.root = Path(args.data_path)

        data_list_path = args.data_list
        data_list = []
        with open(data_list_path, "r") as f:
            for line in f:
                data_list.append(line.strip())

        from .fad_trim import GeneralFAD
        for subset in data_list:
            data_path = self.root / subset 
            subset_args = copy.deepcopy(args)
            subset_args.data_path = self.root / data_path
            self.datasets[subset] = GeneralFAD(subset_args, split=split, train_mode=train_model)
        self.order_list = data_list


class MLAADMultiAttacker(MutliDSBase):

    def __init__(self, args: argparse.Namespace, split: str, train_mode: bool) -> None:
        """ Load MLAAD with multiple attackers, each attacker form a dataset
        key_id: mlaad_{attacker}
        """
        super().__init__(args, split, train_mode)

        assert split in ["train", "dev", "test"], f"split {split} not supported"

        from .fad_trim import MLAADMono

        attackers = self.load_attackers(args)
        for attacker in attackers:
            # deepcopy args
            attacker_args = copy.deepcopy(args)
            fake_path = os.path.join(attacker_args.fake_path, attacker)
            attacker_args.fake_path = fake_path
            cur_ds_key = f"mlaad_{attacker}"
            self.datasets[cur_ds_key] = MLAADMono(attacker_args, split=split, train_mode=train_mode)
            self.order_list.append(cur_ds_key)
        
        for attacker in attackers:
            logger.info(f"attacker: {attacker}")
        self.redistribute_real()

    
    def load_attackers(self, args):
        fake_path = args.fake_path
        fake_path = fake_path.replace("/", "_")

        # if order is given
        if hasattr(args, "attacker_order"):
            attacker_order = args.attacker_order

            attackers = []
            with open(attacker_order, "r") as f:
                for line in f:
                    attackers.append(line.strip())
        else:
            # make to compatible with the previous configs
            attackers = os.listdir(args.fake_path)

        # random shuffle
        def random_shuffle_order():
            tmp_path = Path("./tmp")
            if not tmp_path.exists():
                os.makedirs(tmp_path, exist_ok=True)

            attacker_tmp_path = tmp_path / fake_path
            if not attacker_tmp_path.exists():
                attackers = os.listdir(args.fake_path)
                # randomly shuffle the attackers
                np.random.shuffle(attackers)
                with open(attacker_tmp_path, "w") as f:
                    for attacker in attackers:
                        f.write(attacker + "\n")
            else:
                attackers = []
                with open(attacker_tmp_path, "r") as f:
                    for line in f:
                        attackers.append(line.strip())
        return attackers
    

    def remove_real(self, cur_ds):
        fake_ids = []
        real_ids = []
        for id in cur_ds.uttids:
            if cur_ds.uttid2info[id]["label"] == 1:
                fake_ids.append(id)
            else:
                real_ids.append(id)
        cur_ds.uttids = fake_ids
        return real_ids


    def redistribute_real(self):
        """
        Evenly separate the real data uttid to each attacker
        """
        if len(self.datasets) == 1:
            return
        # uttinfo should contains the whole real set
        # check if the real sets are the same
        def cmp(alist, blist):
            alist.sort()
            blist.sort()
            all_same = True
            for i in range(len(alist)):
                if alist[i] != blist[i]:
                    print(f"{i} th: alist: {alist[i]}, blist: {blist[i]}")
                    print(f"len(alist[i])={len(alist[i])}")
                    print(f"len(blist[i])={len(blist[i])}")
                    print(alist[i])
                    print(blist[i])
                    all_same = False
                    break
            return all_same
        
        real_split_file = None
        # make sure that they use the same real split file
        # bc MLAAD share the real data
        for ds_id in self.datasets:
            self.remove_real(self.datasets[ds_id])
            # load real audios from the real utt files
            real_split = self.datasets[ds_id].tmp_real_split
            if real_split_file is None:
                real_split_file = real_split
            assert real_split_file == real_split

        real_uttids = []
        with open(real_split_file, "r") as f:
            for line in f:
                real_uttids.append(line.strip())

        total_attackers = len(self.datasets)
        num_real = len(real_uttids)
        num_real_per_attacker = (num_real + total_attackers - 1) // total_attackers

        #print(f"total_real: {num_real}, num_real_per_attacker: {num_real_per_attacker}")
        for i, ds_id in enumerate(self.datasets):
            #print(f"before cur dataset: {ds_id}, len uttids: {len(self.datasets[ds_id].uttids)}")
            start = i * num_real_per_attacker
            end = (i+1) * num_real_per_attacker
            self.datasets[ds_id].uttids.extend(real_uttids[start:end])
            #print(f"cur dataset: {ds_id}, len uttids: {len(self.datasets[ds_id].uttids)}")
    

class MultiDS(MutliDSBase):
    """ The class can load multiple dataset classes
    currently it only loads asvspoof19 as the train set
    and mlaad as the cl test set

    in short, we'll have multiple train, dev, test sets
    #TODO: now it acts like a class factory, use different 
    subclasses to load different datasets, need to make it more general
    """

    def __init__(self, args: argparse.Namespace, split: str, train_mode: bool):
        super().__init__(args, split, train_mode)
        self.base_args = argparse.Namespace(**args.base)
        self.cl_args = argparse.Namespace(**args.cl)
        self.future_args = argparse.Namespace(**args.future)

        # TODO: make it more general
        if hasattr(args, "transformation"):
            # add the arg to the asvspoof19_args and mlaad_en_args
            self.base_args.transformation = args.transformation
            self.cl_args.transformation = args.transformation
            self.future_args.transformation = args.transformation
        if hasattr(args, "max_len"):
            self.base_args.max_len = args.max_len
            self.cl_args.max_len = args.max_len
            self.future_args.max_len = args.max_len

        self.datasets = {}

        if "base" in split:
            train_or_dev = "train" if "train" in split else "dev"
            self.load_base_traindev(train_or_dev)
        elif "cl" in split:
            train_or_dev = "train" if "train" in split else "dev"
            self.load_cl_traindev(train_or_dev)
        elif "future" in split:
            self.load_future()
        elif "test" in split:
            self.load_test()

        # TODO: weighted loss or upsample data? now let's just use the upsample data

        if "train" in split and "cl" in split:
            for key in self.datasets:
                self.datasets[key].uttids = self.datasets[key].upsample_data()

    def load_base_traindev(self, split: str):
        """
        split: str, [train, dev]
        """
        from .fad_trim import GeneralFAD
        base_ds = GeneralFAD(self.base_args, split=split, train_mode=self.train_mode)
        self.datasets[f"{split}_{base_ds.dataset_name}"] = base_ds
        self.order_list.append(f"{split}_{base_ds.dataset_name}")

    def load_cl_traindev(self, split: str):
        """
        split: str, [cl_train, cl_dev]
        """
        cl_ds = GeneralMultiDS(self.cl_args, split=split, train_model=self.train_mode)
        datasets = cl_ds.get_datasets()
        for key in datasets:
            self.datasets[f"{split}_{key}"] = datasets[key]
            self.order_list.append(f"{split}_{key}")


    def load_test(self):
        """
        we measure the average performance, so we need to load both datsets
        """
        split = "test"
        from .fad_trim import GeneralFAD 
        base_test = GeneralFAD(self.base_args, split=split, train_mode=False)
        self.datasets[f"test_{base_test.dataset_name}"] = base_test 
        self.order_list.append(f"test_{base_test.dataset_name}")

        cl_test = GeneralMultiDS(self.cl_args, split=split, train_model=False) 
        datasets = cl_test.get_datasets()
        for key in datasets:
            self.datasets[f"test_{key}"] = datasets[key]
            self.order_list.append(f"test_{key}")

    def load_future(self):
        split = "test"
        future_test = GeneralMultiDS(self.future_args, split=split, train_model=False)
        datasets = future_test.get_datasets()
        # TODO: maybe we shouldn't use prefix to distinguish the datasets
        for key in datasets: 
            self.datasets[f"future_{key}"] = datasets[key] 
            self.order_list.append(f"future_{key}")