# encoding: utf-8
# Author: Yixuan
# 
#

import os
import argparse
from deprecated import deprecated

import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import librosa

import pandas as pd
from pathlib import Path

from tomato.utils import logger
from .segment import AudioSegment
from .base import AudioDataset
from .audio_augmentation import AudioAugmentation
from . import audio_util

class FADTrim(AudioDataset):
    """
    The dataset is designed for Fake Audio Detection task, it will trim the 
    input audio to a fixed length.
    output format: NCFT
    """

    def __init__(self, args: argparse.Namespace, split: str, train_mode: bool = True):
        super().__init__(args, split, train_mode)
        self.sample_rate = 16_000
        self.max_len = getattr(args, "max_len", 4) * self.sample_rate
        logger.info(f"Max length: {self.max_len}. Note: if the transformation is used, and the features are read from the disk, the max_len will be ignored, because the audios are already trimmed before saved")
        # for visualization
        self.return_audio = getattr(args, "return_audio", False)
        self.trim_silence = getattr(args, "trim_silence", True)
        self._load_both(args)
        assert len(self.uttid2path) == len(self.uttid2info), f"Length of uttid2path: {len(self.uttid2path)}, uttid2label: {len(self.uttid2info)}"
        self.uttids = list(self.uttid2path.keys())

        # transformation, e.g., lfcc, mfcc, etc
        if hasattr(args, "transformation"):
            #logger.info("Using transformation")
            self.transformation = self._load_transformation(args)
        else:
            self.transformation = None
        if train_mode and hasattr(args, "upsample"):
            logger.info("Oversampling data")
            self.uttids = self.upsample_data()
        self.uttid2idx = {uttid: idx for idx, uttid in enumerate(self.uttids)}
        
        self.do_augment = train_mode & getattr(args, "do_augment", False)
        if self.do_augment:
            self.augmentor = AudioAugmentation()
        self.train_mode = train_mode
    
    def _load_both(self, args):
        self.uttid2path = self._load_manifest(args)
        self.uttid2info = self._load_uttinfo(args)

    def get_info(self, uttid):
        return self.uttid2info[uttid]
    
    def get_path(self, uttid):
        return self.uttid2path[uttid]
    
    def get_idx_by_uttid(self, uttid):
        return self.uttid2idx[uttid]

    def _load_transformation(self, args):
        import tomato.data.transformation as tomato_tfn
        transformation_args = args.transformation
        # turn dict to argparse.Namespace
        # print(f"Transformation args: {transformation_args}")
        transformation_args = argparse.Namespace(**transformation_args)
        self.transformation_args = transformation_args

        func = transformation_args.func
        if func.lower() == "lfcc":
            return tomato_tfn.LFCC(transformation_args)
        elif func.lower() == "mfcc":
            return tomato_tfn.MFCC(transformation_args)
        else:
            raise ValueError(f"Transformation {func} is not supported")


    def _load_manifest(self, args):
        raise NotImplementedError
    

    def _load_uttinfo(self, args):
        raise NotImplementedError
    

    def __len__(self):
        return len(self.uttids)
    

    def sample_segment(self, feats):
        # feats: [C, T]
        if feats.shape[1] == self.max_len:
            return feats
        elif feats.shape[1] < self.max_len: # max len: 4 secs, 64000
            num_repeats = int(self.max_len / feats.shape[1]) + 1
            feats = feats.repeat(1, num_repeats)
        stt = np.random.randint(feats.shape[1] - self.max_len)
        feats = feats[:, stt:stt+self.max_len]
        return feats
    
    def get_audio(self, uttid):
        audio_path = self.uttid2path[uttid]
        if not os.path.exists(audio_path):
            raise ValueError(f"File {audio_path} does not exist")

        frame_offset = 0
        num_frames = -1
        info = self.uttid2info[uttid]
        if "st" in info and "dur" in info:
            st = info["st"]
            dur = info["dur"]
            st = int(st * self.sample_rate)
            dur = int(dur * self.sample_rate)
            if dur > self.max_len:
                frame_offset = np.random.randint(dur - self.max_len)
                num_frames = self.max_len
        audio, sample_rate = audio_util.get_audio(audio_path, 
                                                  to_mono=True, trim_sil=self.trim_silence, 
                                                  frame_offset=frame_offset, num_frames=num_frames) 
        if audio.shape[1] == 0:
            raise ValueError(f"Audio {audio_path} is empty")
        audio = self.sample_segment(audio)
        if self.do_augment and self.train_mode:
            audio = self.augmentor(audio)
        
        #self.augmentor = AudioAugmentation()
        #audio, noise = self.augmentor.add_pink_noise(audio, noise_std=10)
        return audio 
    
    def _getitem_impl(self, idx):
        """ if the audio is longer than 4 seconds, we randomly sample a 4-second segment
        otherwise, we pad the audio to 4 seconds by repeating the audio
        """
        if self.fast_forward_mode:
            return None
        
        uttid = self.uttids[idx]
        uttinfo = self.uttid2info[uttid]
        # get audio

        if self.transformation is not None: #and self.transformation_args.func == "lfcc":
            # TODO: add out_dir in config
            def read_and_save_feat(out_dir):
                save_path = f"{out_dir}/{uttid}.pt"
                # save the feature
                if not os.path.exists(save_path):
                    feats = self.get_audio(uttid)
                    feats = self.transformation(feats)
                    torch.save(feats, save_path)
                    logger.info(f"Saving feature to {save_path}")
                else:
                    #logger.info("Loading from saved feature")
                    feats = torch.load(save_path)
                return feats

            if "FEAT_LOCAL" in os.environ:
                out_dir = os.environ["FEAT_LOCAL"]
                audio = None
                feats = read_and_save_feat(out_dir)
            else:
                #self.do_augment = uttinfo["label"]
                audio = self.get_audio(uttid)
                feats = self.transformation(audio)
        else:
            audio = self.get_audio(uttid)
            feats = audio

        #logger.info(f"Shape of feats: {feats.shape}")
        #raise ValueError("Stop here")
        
        ret_info = {
            "uttid": uttid,
            "feats": feats,
            "label": uttinfo["label"],
            "origin_ds": uttinfo["origin_ds"],
            "speaker": uttinfo["speaker"],
            "attacker": uttinfo["attacker"]
        }
        if self.return_audio:
            ret_info["audio"] = audio
        
        return ret_info

    def __getitem__(self, idx):
        while True:
            try:
                return self._getitem_impl(idx)
            except Exception as e:
                logger.error(f"Error in loading idx {idx}: {e}")
                idx = np.random.randint(len(self.uttids))
                logger.info(f"Randomly choose idx {idx}")

    def upsample_data(self):
        real_utts = []
        fake_utts = []

        for uttid in self.uttids:
            utt_dict = self.uttid2info[uttid]
            if utt_dict["label"] == 1: 
                fake_utts.append(uttid)
            else:
                real_utts.append(uttid)

        if len(real_utts) > len(fake_utts):
            # upsample fake
            k = len(real_utts) // len(fake_utts)
            fake_utts = fake_utts * k
            #logger.info(f"Upsampling fake data by {k}, len(fake): {len(fake_utts)}, len(real): {len(real_utts)}")
            # print(f"fake: {len(fake_utts)}, real: {len(real_utts)}, k: {k}")
        elif len(real_utts) < len(fake_utts):
            logger.info(f"Upsampling real data")
            k = len(fake_utts) // len(real_utts)
            real_utts = real_utts * k
            #print(f"fake: {len(fake_utts)}, real: {len(real_utts)}, k: {k}")
        
        utt_ids = real_utts + fake_utts
        # shuffle the utt_ids
        np.random.shuffle(utt_ids)
        return utt_ids
    
    
    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: list of dict
        """
        # if it's fast forward, then all the samples are None
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        
        uttids = [sample["uttid"] for sample in batch]
        feats = [sample["feats"] for sample in batch]
        labels = [sample["label"] for sample in batch]
        origin_ds = [sample["origin_ds"] for sample in batch]
        speakers = [sample["speaker"] for sample in batch]
        attackers = [sample["attacker"] for sample in batch]

        batch_feats = torch.stack(feats)
        batch_labels = torch.LongTensor(labels)
        #logger.info(f"Batch feats shape: {batch_feats.shape}")
        #logger.info(f"Batch labels shape: {batch_labels.shape}")
        return {
            "uttids": uttids,
            "feats": batch_feats,
            "labels": batch_labels,
            "origin_ds": origin_ds,
            "speakers": speakers,
            "attackers": attackers
        } 


class GeneralFAD(FADTrim):


    def __init__(self, args: argparse.Namespace, split: str, train_mode: bool = True):
        super().__init__(args, split, train_mode)
        logger.info(f"Dataset {args.data_path}, split: {split}")
    
    def _load_manifest(self, args):
        """ Has two columns
        uttid, path
        """
        self.dataset_name = Path(args.data_path).name
        logger.info(f"Loading dataset {self.dataset_name}, split: {self.split}")
        manifest_path = os.path.join(args.data_path, f"{self.split}.tsv")
        uttid2path = {}
        with open(manifest_path, "r") as f:
            for line in f:
                uttid, path = line.strip().split("\t")
                uttid2path[uttid] = path
        return uttid2path
    

    def _load_uttinfo(self, args):
        """ Has five columns
        uttid, origin_ds, speaker, attacker, label
        label: [bonafide, spoof]
        """
        label_path = os.path.join(args.data_path, f"{self.split}.txt")
        uttid2info = {}
        with open(label_path, "r") as f:
            for line in f:
                uttid, origin_ds, speaker, attacker, label = line.strip().split("\t")
                uttid2info[uttid] = {
                    "origin_ds": origin_ds,
                    "speaker": speaker,
                    "attacker": attacker,
                    "label": 0 if label=="bonafide" else 1
                }
        return uttid2info
    

    @staticmethod
    def collate_fn(batch):
        return FADTrim.collate_fn(batch)
    

class FastGeneralFAD(GeneralFAD):
    """ 
    Use mega_dict to quickly load a dataset.
    Compared to GeneralFAD, it doesn't require the user to prepare two files (tsv and txt), 
    only the uttids or the origin_ds dataset name is needed.
    Cons: the user need to run local/create_metads_dict.py to create the mega dict,
    so it needs more memory
    This dataset is suitable for testing, since the split info is not stored in the 
    mega dict.
    """

    def __init__(self, args: argparse.Namespace, split: str, train_mode: bool = True):
        self.filtered_uttids = getattr(args, "uttids", None)
        self.origin_ds_re_pattern = getattr(args, "origin_ds", None)
        logger.info(f"the origin_ds_re_pattern is {self.origin_ds_re_pattern}")
        if self.filtered_uttids is None and self.origin_ds_re_pattern is None:
            raise ValueError("Either uttids or origin_ds should be provided")

        # load mega_dict
        mega_dict = Path(__file__).parent.parent.parent/"asset"/"mega_dict.pkl"
        import pickle
        with open(mega_dict, "rb") as file:
            self.mega_dict = pickle.load(file)
        super().__init__(args, split, train_mode)


    def _filter_by_origin_ds(self):
        import re
        self.uttid2info = {}
        self.uttid2path = {}
        for uttid in self.mega_dict.uttid2info.keys():
            origin_ds = self.mega_dict.uttid2info[uttid]["origin_ds"]
            if re.match(self.origin_ds_re_pattern, origin_ds):
                self.uttid2info[uttid] = self.mega_dict.uttid2info[uttid]
                self.uttid2path[uttid] = self.mega_dict.uttid2path[uttid]
                self.uttid2info[uttid].pop("line")
                label = self.uttid2info[uttid]["label"]
                self.uttid2info[uttid]["label"] = 0 if label == "bonafide" else 1

                #print(f"info {self.mega_dict.uttid2info[uttid]}")

    def _load_both(self, args):
        if self.filtered_uttids is not None:
            if self.origin_ds_re_pattern is not None:
                raise ValueError("Both uttids and origin_ds are provided")
            self.uttid2info = self.mega_dict.get_info(self.filtered_uttids)
            self.uttid2path = self.mega_dict.get_path(self.filtered_uttids)
        elif self.origin_ds_re_pattern is not None:
            self._filter_by_origin_ds()

    def _load_manifest(self, args):
        raise NotImplementedError("This class simply uses the mega_dict to load both files at once, so this function is not needed")
    
    def _load_uttinfo(self, args):
        raise NotImplementedError("This class simply uses the mega_dict to load both files at once, so this function is not needed")


@deprecated(reason="Use GeneralFAD instead")
class SVDD(FADTrim):
    """
    SVDD dataset
    manifest format:
        basedir
        utt_id relative_path_to_audio
    label:
        original_singer, ctrsvdd_singer, utt_id - attacker_model, label
        kiritan CtrSVDD_0059 CtrSVDD_0059_D_0000758 - A01 deepfake

    cfg attribute:
        data_path
        num_workers
        batch_size
    """

    def __init__(self, 
                 args: argparse.Namespace,
                 split: str,
                 train_mode: bool = True):
        super().__init__(args, split, train_mode)


    def _load_manifest(self, args):
        """
        under the data dir, there is a manifest file, format:
        ---
        abs_path_to_audio
        uttid relative_path_to_audio
        """
        manifest_path = os.path.join(args.data_path, f"{self.split}.tsv")
        uttid2path = {}
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                uttid, path = line.strip().split()
                uttid2path[uttid] = os.path.join(self.root_dir, path)
        return uttid2path


    def _load_uttinfo(self, args):
        label_path = os.path.join(args.data_path, f"{self.split}.txt")
        uttid2info = {}
        with open(label_path, "r") as f:
            for line in f:
                origin_ds, singer, uttid, _, attacker, label = line.strip().split()
                uttid2info[uttid] = {
                    "origin_ds": origin_ds,
                    "speaker": singer,
                    "attacker": attacker,
                    "label": 0 if label=="bonafide" else 1
                }
        return uttid2info


    @staticmethod
    def collate_fn(batch):
        return FADTrim.collate_fn(batch)


@deprecated(reason="Use GeneralFAD instead")
class MLAADMono(FADTrim):
    """ Load a monolingual dataset from MLAAD
    """

    def __init__(self, args: argparse.Namespace, split: str, train_mode: bool = True):
        super().__init__(args, split, train_mode)
        # here we are assuming that if the train_fake path exist
        # then the whole dataset is done
        
        train_fake = self._get_split_path(self.args.fake_path, "train")
        train_real = self._get_split_path(self.args.real_path, "train")

        if not train_fake.exists() or not train_real.exists():
            fake_utts = []
            real_utts = []
            for uttid in self.uttids:
                label = self.uttid2info[uttid]["label"]
                if label == 0:
                    real_utts.append(uttid)
                else:
                    fake_utts.append(uttid)
        
        if not train_fake.exists():
            logger.info(f"Splitting fake data for {args.fake_path}")
            self._split_train_dev_test(fake_utts, args.fake_path)
        if not train_real.exists():
            logger.info(f"Splitting real data for {args.real_path}")
            self._split_train_dev_test(real_utts, args.real_path)
        self._load_split_utt_from_file(split)


    def _get_split_path(self, file_path, split):
        file_path = file_path.replace("/", "_")
        split_path = split + "_" + file_path
        return Path("./tmp") / split_path


    def _load_split_utt_from_file(self, split):
        def read_from_split_file(split_file):
            logger.info(f"Loading split file {split_file}")
            utts = []
            with open(split_file, "r") as f:
                for line in f:
                    utts.append(line.strip())
            return utts
        fake_split = self._get_split_path(self.args.fake_path, split)
        fake_utts = read_from_split_file(fake_split)
        real_split = self._get_split_path(self.args.real_path, split)
        real_utts = read_from_split_file(real_split)
        self.uttids = fake_utts + real_utts
        self.tmp_fake_split = fake_split
        self.tmp_real_split = real_split


    def _split_train_dev_test(self, data_list, data_path):
        """ The data_list could be the utt_ids for the real or fake 
        datasets. If the real set is changed, only the real uttids
        needs to be called with the function
        The data_path could be the fake_path or the real_path
        """
        np.random.seed(321)
        np.random.shuffle(data_list)
        split_point1 = int(len(data_list) * 0.8)
        split_point2 = int(len(data_list) * 0.9)
        train = data_list[:split_point1]
        dev = data_list[split_point1:split_point2]
        test = data_list[split_point2:]
        
        def write_to_tmp(utts, split, file_path):
            split_path = self._get_split_path(file_path, split)
            with open(split_path, "w") as f:
                for utt in utts:
                    f.write(utt + "\n")
        
        write_to_tmp(train, "train", data_path)
        write_to_tmp(dev, "dev", data_path)
        write_to_tmp(dev, "test", data_path)
    
    
    def _check_all_files_exist(self):
        for idx in range(len(self.uttids)):
            self.__getitem__(idx)

    
    def _load_manifest(self, args):
        uttid2path = {}
        uttid2info = {}

        def load_all_paths(root_dir, cur_label):
            cur_uttid2path = {}
            # TODO: this part is very messy 
            root_dir_path = root_dir.replace("/", "_")
            temp_path = Path("./tmp") / root_dir_path 
            
            def extract_content_from_path(root, wav_path):
                cur_dirs = root.split("/")
                if cur_label == 0: # real
                    # note should use -4 for m-ailabs
                    speaker = cur_dirs[-3]
                    attacker = "real"
                else:
                    speaker = "unk" # fake
                    attacker = cur_dirs[-1]

                uttid = attacker + "_" + wav_path[:-4]
                cur_uttid2path[uttid] = os.path.join(root, wav_path)
                uttid2info[uttid] = {
                    "origin_ds": "MLAAD",
                    "speaker": speaker,
                    "attacker": attacker,
                    "label": cur_label
                }

            if temp_path.exists():
                #logger.info(f"Temp file {temp_path} found")
                with open(temp_path, "r") as f:
                    for line in f:
                        root, wav_path = line.strip().split("\t")
                        extract_content_from_path(root, wav_path)
                return cur_uttid2path

            with open(temp_path, "w") as f:
                for root, dirs, files in os.walk(root_dir, followlinks=True):
                    for file in files:
                        if file.startswith("."):
                            continue
                        if file.endswith(".wav"):
                            f.write(f"{root}\t{file}\n")
                            extract_content_from_path(root, file)
            return cur_uttid2path
        
        fake_path = load_all_paths(args.fake_path, 1)
        real_path = load_all_paths(args.real_path, 0)
        uttid2path.update(fake_path)
        uttid2path.update(real_path)
        self.uttid2label = uttid2info
        return uttid2path

    def _load_uttinfo(self, args):
        return self.uttid2label
    

    @staticmethod
    def collate_fn(batch):
        return FADTrim.collate_fn(batch)


@deprecated(reason="Use GeneralFAD instead")
class ASVspoof19(FADTrim):


    def __init__(self, args: argparse.Namespace, split: str, train_mode: bool = True):
        assert split in ["train", "dev", "test"], f"Split {split} is not supported"
        if split == "test":
            split = "eval"
        self.root_dir = f"{args.data_path}/ASVspoof2019_LA_{split}/flac"
        suffix = "trn" if split == "train" else "trl"
        self.manifest_path = os.path.join(args.data_path, f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.{suffix}.txt")
        super().__init__(args, split, train_mode)


    def _load_manifest(self, args):
        uttid2path = {}
        uttid2info = {}
        with open(self.manifest_path, "r") as f:
            for line in f:
                speaker, uttid, attacker, _, label = line.strip().split()
                uttid2path[uttid] = os.path.join(self.root_dir, f"{uttid}.flac")
                uttid2info[uttid] = {
                    "origin_ds": "ASVspoof19",
                    "speaker": speaker,
                    "attacker": attacker,
                    "label": 0 if label=="bonafide" else 1
                }
        self.uttid2info= uttid2info
        return uttid2path

    
    def _load_uttinfo(self, args):
        return self.uttid2info
    

    @staticmethod
    def collate_fn(batch):
        return FADTrim.collate_fn(batch)
    

@deprecated(reason="Use GeneralFAD instead")
class GPTSoVITS(FADTrim):

    """
    The class is used to load datasets for the experiments: does fine-tuning generate
    new artifacts.
    Meta files for the tasks are mainly stored in csv files
    """

    def __init__(self, args: argparse.Namespace, split: str, train_mode: bool = True):
        # if trim_start in args, then we will trim the audio
        if hasattr(args, "trim_start"):
            # TODO: sample rate is hard coded
            self.trim_start = int(args.trim_start * 16000)
        else:
            self.trim_start = 0
        super().__init__(args, split, train_mode)


    def __getitem__(self, idx):
        if self.fast_forward_mode:
            return None
        
        uttid = self.uttids[idx]
        audio_path = self.uttid2path[uttid]
        if not os.path.exists(audio_path):
            logger.info(f"File {audio_path} does not exist")
            raise ValueError(f"File {audio_path} does not exist")
        
        audio = AudioSegment.from_file(audio_path)
        feats = torch.from_numpy(audio.samples).float()
        #import torchaudio 
        #torchaudio.save("before_trim.wav", feats.unsqueeze(0), sample_rate=audio.sample_rate)
        if self.trim_start > 0 and feats.shape[0] > self.trim_start:
            #print(f"Trim start: {self.trim_start}")
            #print(f"shape before trim: {feats.shape}")
            feats = feats[self.trim_start:]
            # save feats as audio file
        if feats.shape[0] == 0:
            print(f"File {audio_path} is empty")
            raise ValueError(f"File {audio_path} is empty")
        feats = self.sample_segment(feats)
        
        #torchaudio.save("after_trim.wav", feats.unsqueeze(0), sample_rate=audio.sample_rate)
        #raise ValueError("Stop here")
        if self.transformation is not None:
            feats = self.transformation(feats)
        
        uttinfo = self.uttid2info[uttid]
        return {
            "uttid": uttid,
            "feats": feats,
            "label": uttinfo["label"],
            "origin_ds": uttinfo["origin_ds"],
            "speaker": uttinfo["speaker"],
            "attacker": uttinfo["attacker"]
        }
        

    def _load_manifest(self, args):
        """ Get uttid2path
        """
        self.aishell_root = Path(args.aishell_root)
        self.tts_root = Path(args.tts_root)
        self.csv_root = Path(args.csv_root)
        self.fake_path = self.csv_root/f"{self.split}_fake.csv" 
        self.real_path = self.csv_root/f"{self.split}_real.csv"
        uttid2path = {}

        def load_one_csv(data_path):
            """ data_path could fake or real path
            """
            tag = "real" if "real" in str(data_path) else "fake"
            cur_df = pd.read_csv(data_path)
            for idx, row in cur_df.iterrows():
                speaker_id = row["speaker"]
                wav_id = row["wav_id"]
                utt_id =f"{tag}_{speaker_id}_{wav_id}"
                if tag == "real":
                    relative_path = row["wavfile"]
                    abs_path = self.aishell_root / relative_path
                else:
                    abs_path = self.tts_root / speaker_id / f"{wav_id}.wav"
                uttid2path[utt_id] = str(abs_path)

        load_one_csv(self.fake_path)
        load_one_csv(self.real_path)

        # check file existence
        #for utt, path in uttid2path.items():
        #    assert os.path.exists(path), f"File {path} does not exist"
        return uttid2path


    def _load_uttinfo(self, args):
        """ Should include origin_ds, speaker, attacker, label
        """

        uttid2info = {}

        def load_one_csv(data_path):
            tag = "real" if "real" in str(data_path) else "fake"
            cur_df = pd.read_csv(data_path)
            for idx, row in cur_df.iterrows():
                utt_id = f"{tag}_{row['speaker']}_{row['wav_id']}"
                uttid2info[utt_id] = {
                    "origin_ds": "aishell1",
                    "speaker": row["speaker"], 
                    "attacker": args.tts_model,
                    "label": 0 if "real" in str(data_path) else 1
                }
        
        load_one_csv(self.fake_path)
        load_one_csv(self.real_path)
        return uttid2info
    

    @staticmethod
    def collate_fn(batch):
        return FADTrim.collate_fn(batch)

if __name__ == "__main__":
    from tomato.utils import config2arg
    config_path = os.environ["CONFIG"]
    data_args = config2arg(config_path, "data")

    fast_dataset = FastGeneralFad(data_args, "test", train_mode=False)