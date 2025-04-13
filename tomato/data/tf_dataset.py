# encoding: utf-8
# Author: Yixuan
# 
#

"""
This file use transformers to load the dataset.
"""

from argparse import Namespace
import os
from pathlib import Path
import torch

from tomato.utils import logger
from .fad_trim import FADTrim, GeneralFAD
from . import audio_util

class TFGeneralFAD(GeneralFAD):
    """
    This dataset always load full audio, and use transformer to pad the audios
    """
    def __init__(self, args: Namespace, split: str, train_mode: bool = True):
        from transformers import AutoFeatureExtractor
        super().__init__(args, split, train_mode)
        self.tf_mdl = args.tf_mdl
        self.feat_extractor = AutoFeatureExtractor.from_pretrained(self.tf_mdl)

    def get_audio(self, uttid):
        audio_path = self.uttid2path[uttid]
        if not Path(audio_path).exists():
            raise ValueError(f"File {audio_path} does not exist")

        audio, sample_rate = audio_util.get_audio(audio_path, 
                                                  to_mono=True, trim_sil=self.trim_silence, 
                                                  frame_offset=0, num_frames=-1)
        # audio is torch tensor, shape: [1, T]
        audio = audio[:, :self.max_len]
        if audio.shape[1] == 0:
            raise ValueError(f"Audio {audio_path} is empty")
        if self.do_augment and self.train_mode:
            audio = self.augmentor(audio)
        np_audio = audio.numpy()
        return np_audio
    
    def _getitem_impl(self, idx):
        if self.fast_forward_mode:
            return None
        
        uttid = self.uttids[idx]
        uttinfo = self.uttid2info[uttid]
        raw_audio = self.get_audio(uttid)
        ret_info = {
            "uttid": uttid,
            "raw_audio": raw_audio,
            "label": uttinfo["label"],
            "origin_ds": uttinfo["origin_ds"],
            "speaker": uttinfo["speaker"],
            "attacker": uttinfo["attacker"],
            "self": self
        }
        return ret_info

    
    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: list of dict
        Returns:
            dict: containing batched features and metadata
        """
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
    
        uttids = [sample["uttid"] for sample in batch]
        raw_audio = [sample["raw_audio"].squeeze(0) for sample in batch]  # Remove channel dim
        labels = [sample["label"] for sample in batch]
        origin_ds = [sample["origin_ds"] for sample in batch]
        speakers = [sample["speaker"] for sample in batch]
        attackers = [sample["attacker"] for sample in batch]

        # Get the feature extractor from the first sample's dataset instance
        feat_extractor = batch[0]["self"].feat_extractor
        features = feat_extractor(
            raw_speech=raw_audio,
            padding=True,  # Pad all inputs in the batch to the same length
            return_attention_mask=True,  # Return attention mask
            return_tensors="pt",  # Return PyTorch tensors
            sampling_rate=16000,  # Specify the sampling rate of the input audio
        ) 
        batch_labels = torch.LongTensor(labels)
        return {
            "uttids": uttids,
            "feats": features.input_values, # shape [bz, T] 
            "padding_mask": features.attention_mask, #shape [bz, T] 
            "labels": batch_labels,
            "origin_ds": origin_ds,
            "speakers": speakers,
            "attackers": attackers
        }

    
if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH")
    args = Namespace(
        max_samples=4*16000,
        fix_len=False,
        trim_silence=True,
        do_augment=True,
        augment_type="rawboost",
        data_path=data_path,
        tf_mdl ='facebook/wav2vec2-xls-r-300m' 
    )

    ds = TFGeneralFAD(args, split='dev', train_mode=False)
    uttid = ds.uttids[1]
    print('uttid:', uttid)

    audio = ds.get_audio(uttid)
    print(type(audio)) # numpy.ndarray
    print(audio.shape) # (1, T)

    #import dataloader and test one batch
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ds.collate_fn)
    first_batch = next(iter(dataloader))

    for key, item in first_batch.items():
        # if type item is torch.Tensor, then print the shape
        if isinstance(item, torch.Tensor):
            print(f"{key}: {item.shape}")
        else:
            print(f"{key}: {item}")
