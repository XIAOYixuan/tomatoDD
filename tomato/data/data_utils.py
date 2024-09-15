# encoding: utf-8
# Author: Yixuan
# 
#
from copy import deepcopy
from pathlib import Path

import numpy as np

#TODO: rename --> MetaDatasetIO
#TODO: add, need to uniq uttids
#TODO: need to re-split fake and real after adding
class MetaDatasetReader:

    def __init__(self):
        self.real_uttids = None
        self.fake_uttids = None

    @classmethod
    def from_path(cls, path, split=None):
        obj = cls()
        obj.path = Path(path)
        if split is not None:
            obj.read_splits([split])
        obj.uttids = list(obj.uttid2info.keys())
        cls.show_statistics(obj)
        return obj

    @classmethod
    def from_utt_info(cls, uttids, utt2info, utt2path):
        obj = cls()
        
        obj.uttids = uttids
        obj.uttid2info = utt2info
        obj.uttid2path = utt2path

        for uttid in obj.uttid2info:
            if "st" in obj.uttid2info[uttid] and "dur" in obj.uttid2info[uttid]:
                line = f"{uttid}\t{obj.uttid2info[uttid]['origin_ds']}\t{obj.uttid2info[uttid]['speaker']}\t{obj.uttid2info[uttid]['attacker']}\t{obj.uttid2info[uttid]['label']}\t{obj.uttid2info[uttid]['st']}\t{obj.uttid2info[uttid]['dur']}\n"
            else:
                line = f"{uttid}\t{obj.uttid2info[uttid]['origin_ds']}\t{obj.uttid2info[uttid]['speaker']}\t{obj.uttid2info[uttid]['attacker']}\t{obj.uttid2info[uttid]['label']}\n"
            obj.uttid2info[uttid]["line"] = line
        return obj

    #TODO: need to update the real and fake uttids
    # impl the + method
    def __add__(self, other):
        new_obj = deepcopy(self)
        new_obj.uttid2info.update(other.uttid2info)
        new_obj.uttid2path.update(other.uttid2path)
        new_obj.uttids = list(new_obj.uttid2info.keys())
        return new_obj
    
    def __len__(self):
        return len(self.uttids)

    def sample_one_class(self, split_dict, class_label):
        if class_label == "fake":
            this_uttids = deepcopy(self.fake_uttids)
        else:
            this_uttids = deepcopy(self.real_uttids)
        remain_uttids = len(this_uttids)

        split2uttids = {}
        for split, count in split_dict.items():
            if remain_uttids == 0:
                print(f"Warning: no more uttids for class {class_label} split {split}")
                continue
            cur_uttids = min(remain_uttids, count[class_label])
            chosen_uttids = np.random.choice(this_uttids, cur_uttids, replace=False)
            this_uttids = list(set(this_uttids) - set(chosen_uttids))
            split2uttids[split] = list(chosen_uttids)
            remain_uttids -= cur_uttids
            remain_uttids = max(0, remain_uttids)
        return split2uttids

    def get_path(self, uttids):
        utt2path = {}
        for uttid in uttids:
            utt2path[uttid] = self.uttid2path[uttid]
        return utt2path
    
    def get_info(self, uttids):
        utt2info = {}
        for uttid in uttids:
            utt2info[uttid] = self.uttid2info[uttid]
        return utt2info
    
    def query_path(self, uttid):
        return self.uttid2path[uttid]

    def split_dataset_num(self, split_dict):
        """ Split uttids based on the absolute number
        """
        fake_split2uttids = self.sample_one_class(split_dict, "fake")
        real_split2uttids = self.sample_one_class(split_dict, "real")
        self.split2uttids = {}
        self.split2uttids.update(fake_split2uttids)
        self.split2uttids.update(real_split2uttids)
        return self.split2uttids 

    def split_dataset(self, split_dict):
        """ Split uttids based on the split_dict ratio
        """
        # dict: has {key: ratio}
        # e.g. {"train": 0.8, "dev": 0.1, "test": 0.1}

        if self.fake_uttids is None:
            self._split_real_and_fake()
        assert sum(split_dict.values()) == 1.0, "Split ratios should sum to 1.0"
        total_fake = len(self.fake_uttids)
        total_real = len(self.real_uttids)
        split2count = {}
        for split, ratio in split_dict.items():
            split2count[split] = {
                    "fake": int(total_fake*ratio),
                    "real": int(total_real*ratio)
                }
        return self.split_dataset_num(split2count)

    def write_dataset(self, split, out_dir):
        out_dir = Path(out_dir)
        print(f"Writing the whole dataset as {split} to {out_dir}")
        with open(out_dir/f"{split}.tsv", "w") as ftsv:
            for uttid in self.uttids:
                ftsv.write(f"{uttid}\t{self.uttid2path[uttid]}\n")
    
        with open(out_dir/f"{split}.txt", "w") as ftxt:
            for uttid in self.uttids: 
                ftxt.write(self.uttid2info[uttid]["line"])
        print(f"Done writing")
    
    def write_split(self, split, out_dir):
        out_dir = Path(out_dir)
        print(f"Writing {split} to {out_dir}")
        with open(out_dir/f"{split}.tsv", "w") as ftsv:
            for uttid in self.split2uttids[split]:
                ftsv.write(f"{uttid}\t{self.uttid2path[uttid]}\n")
    
        with open(out_dir/f"{split}.txt", "w") as ftxt:
            for uttid in self.split2uttids[split]:
                ftxt.write(self.uttid2info[uttid]["line"])
        print(f"Done writing")

    def write_multiple_splits(self, splits, out_dir):
        for split in splits:
            self.write_split(split, out_dir)

    def read_splits(self, splits):
        self.uttid2path = {}
        self.uttid2info = {}
        for split in splits:
            utt2path = self._read_tsv(self.path/f"{split}.tsv")
            utt2info = self._read_txt(self.path/f"{split}.txt")
            self.uttid2path.update(utt2path)
            self.uttid2info.update(utt2info)

    def _split_real_and_fake(self):
        self.real_uttids = []
        self.fake_uttids = []
        uttids = list(self.uttid2info.keys())
        for uttid in uttids:
            if self.uttid2info[uttid]["label"] == "bonafide":
                self.real_uttids.append(uttid)
            else:
                self.fake_uttids.append(uttid)
        self.uttids = uttids
        self._sanity_check()

    def show_statistics(self):
        if self.fake_uttids is None:
            self._split_real_and_fake()
        print(f"Number of uttids: {len(self.uttid2info)}")
        print(f"Number of real uttids: {len(self.real_uttids)}")
        print(f"Number of fake uttids: {len(self.fake_uttids)}")
    
    def _sanity_check(self):
        tsv_uttids = set(self.uttid2path.keys())
        txt_uttids = set(self.uttid2info.keys())
        assert tsv_uttids == txt_uttids, "Mismatch between tsv and txt uttids"
    
    def _read_tsv(self, tsv_path):
        utt2path = {}
        with open(tsv_path, "r") as ftsv:
            for line in ftsv:
                uttid, path = line.strip().split("\t")
                utt2path[uttid] = path
        return utt2path
                
    def _read_txt(self, txt_path):
        utt2info = {}
        with open(txt_path, "r") as ftxt:
            for line in ftxt:
                items = line.strip().split("\t")
                if len(items) == 5:
                    uttid, origin_ds, speaker, attacker, label = line.strip().split("\t")
                    utt2info[uttid] = {
                        "origin_ds": origin_ds,
                        "speaker": speaker,
                        "attacker": attacker,
                        "label": label,
                        "line": line
                    }
                else:
                    uttid, origin_ds, speaker, attacker, label, st, dur = line.strip().split("\t")
                    utt2info[uttid] = {
                        "origin_ds": origin_ds,
                        "speaker": speaker,
                        "attacker": attacker,
                        "label": label,
                        "st": float(st),
                        "dur": float(dur),
                        "line": line
                    }
        return utt2info



import pandas as pd
import pickle
class InferResultIO:

    def __init__(self, infer_path, mega_dict_path=None):
        self.infer_path = infer_path
        self.data = pd.read_csv(infer_path, dtype={"uttid": str, "prediction": float, "label": int})
        self.data["label_str"] = self.data["label"].apply(lambda x: "fake" if x == 1 else "real")
        if mega_dict_path is None:
            mega_dict_path = Path(__file__).parent.parent.parent / "asset/mega_dict.pkl"
        with open(mega_dict_path, "rb") as file:
            self.mega_dict = pickle.load(file)
        assert isinstance(self.mega_dict, MetaDatasetReader)

    def _boxplot_stat(self, predictions):
        q1 = predictions.quantile(0.25)
        median = predictions.median()
        q3 = predictions.quantile(0.75)
        iqr = q3-q1
        min_val = q1 - 1.5 * iqr
        max_val = q3 + 1.5 * iqr
        min_val = predictions[predictions >= min_val].min()
        max_val = predictions[predictions <= max_val].max()
        #return [min_val, q1, median, q3, max_val]
        return {
            'min': min_val,
            'q1': q1,
            'median': median,
            'q3': q3,
            'max': max_val
        }

    def _get_target_data(self, external_data=None):
        if external_data is None:
            return self.data
        else:
            return external_data
    
    def get_data_by_label(self, label, external_data = None):
        data = self._get_target_data(external_data)
        if isinstance(label, str):
            if label in ["fake", "real"]:
                label = 1 if label == "fake" else 0
            elif label in ["bonafide", "spoof"]:
                label = 1 if label == "spoof" else 0
            else:
                raise ValueError(f"Invalid label: {label}")
        return data[data['label']==label]

    # st = -inf, ed = inf
    def get_data_by_prediction(self, st=-np.inf, ed=np.inf, external_data=None):
        data = self._get_target_data(external_data)
        # return [st, ed)
        return data[(data['prediction'] >= st) & (data['prediction'] < ed)]
    
    def create_meta_dataset(self, external_data=None):
        data = self._get_target_data(external_data)
        uttids = data['uttid'].tolist()
        uttid2info = self.mega_dict.get_info(uttids)
        uttid2path = self.mega_dict.get_path(uttids)
        return MetaDatasetReader.from_utt_info(uttids, uttid2info, uttid2path)