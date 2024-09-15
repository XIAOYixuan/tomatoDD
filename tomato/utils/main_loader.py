# encoding: utf-8
# Author: Yixuan
# 
#

import torch
import argparse
from tomato.utils import utils, logger
from tomato.data import get_dataset_class

def verify_checkpoint(checkpoint_path, model):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Compare model and checkpoint state_dicts
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint

    for key in model_state_dict.keys():
        if key not in checkpoint_state_dict:
            print(f'Key {key} found in model but not in checkpoint.')

    for key in checkpoint_state_dict.keys():
        if key not in model_state_dict:
            print(f'Key {key} found in checkpoint but not in model.')


def load_dict(config_path):
    """ In case the speech and text model have different dictionaries
    """
    dict_args = utils.config2arg(config_path, "dict")
    assert dict_args is not None
    # TODO: load dict from config, avoid repetitive loading


def load_data(config_path: str, is_infer: bool = False):
    data_args = utils.config2arg(config_path, "data") # argparse.Namespace

    # check if is multds
    class_name = data_args.model_class
    if class_name == "MultiDS":
        return load_multids_data(config_path, is_infer)

    def load_dataset_split(data_args, split, train_mode=True):
        class_name = data_args.model_class
        DatasetClass = get_dataset_class(class_name)
        split_dataset = DatasetClass(args=data_args,
                                            split=split,
                                            train_mode=train_mode)
        from torch.utils.data import DataLoader
        split_dataloader = DataLoader(split_dataset, 
                                batch_size=data_args.test_batch_size, 
                                shuffle=True,
                                drop_last=True, 
                                num_workers=data_args.test_num_workers,
                                collate_fn=DatasetClass.collate_fn)
        return split_dataset, split_dataloader

    # TODO: add splits to config
    if is_infer:
        splits = ["test"]
        #TODO: we need train here cuz we need to compute steps per epoch...
    else:
        splits = ["dev", "train"]
    all_dataset = {}
    all_dataloader = {}
    for split in splits:
        train_mode = (split == "train")
        dataset, dataloader = load_dataset_split(data_args, split, train_mode)
        logger.info(f"Loading {split} dataset, total {len(dataset)} samples...")
        all_dataset[split] = dataset
        all_dataloader[split] = dataloader
    logger.info(f"------- load data done {all_dataset.keys()}-------")
    return (all_dataset, all_dataloader)


def load_multids_data(config_path: str, is_infer: bool = False):
    """ Each MultiDataset should return a list of torch.Dataset classes
    It should be a dictioary, the name to the dataset and the dataset instance
    """
    data_args = utils.config2arg(config_path, "data") # argparse.Namespace
    
    if is_infer:
        if hasattr(data_args, "infer_splits"):
            splits = data_args.infer_splits
        else:
            splits = ["test", "future"]
    else:
        if hasattr(data_args, "train_splits"):
            splits = data_args.train_splits
        else:
            splits = ["base_dev", "base_train", "cl_dev", "cl_train", "test"]

    class_name = data_args.model_class
    DatasetClass = get_dataset_class(class_name)

    all_dataset = {}
    all_dataloader = {}
    ds_keys = {}
    # it stores the name of the test set, because they should use 
    # a different DataLoader config
    test_set_names = set()
    for split in splits:
        multi_ds = DatasetClass(args=data_args, split=split, train_mode=("train" in split))
        datasets_dict = multi_ds.get_datasets()
        for key in datasets_dict:
            all_dataset[key] = datasets_dict[key]
        ds_keys[split] = multi_ds.get_order_list()
        #for key in ds_keys[split]:
        #    logger.info(f"Loading {key} dataset...There're {len(all_dataset[key])} in total")
        #logger.info(f"Loading {split} dataset...There're {len(ds_keys[split])} in total")
        if "test" in split or "future" in split:
            test_set_names.update(ds_keys[split])

    for key in all_dataset:
        from torch.utils.data import DataLoader
        cur_dataset = all_dataset[key]
        DatasetClass = cur_dataset.__class__
        # do we need to set batch_size for each dataset?
        # probly dont, we are mimicing the real-life scenario
        # in which we don't know the exact dataset
        split_dataloader = DataLoader(cur_dataset, 
                                batch_size=data_args.test_batch_size if key in test_set_names else data_args.batch_size,
                                shuffle=True, 
                                num_workers=data_args.test_num_workers if key in test_set_names else data_args.num_workers,
                                collate_fn=DatasetClass.collate_fn)
        all_dataloader[key] = split_dataloader
    
    return (ds_keys, all_dataset, all_dataloader)


# TODO: refactor this
def load_data_old_mixed_model(config_path: str, is_infer=False):
    data_args = utils.config2arg(config_path, "data") # argparse.Namespace
    model_args = utils.config2arg(config_path, "model")

    # TODO: let the model and dataset load these
    from tomato.data import CharTokenizer, Dictionary
    # TODO: make it more general, not only for mix models
    char_dict = Dictionary.load(model_args.char_dict_path)
    char_tokenizer = CharTokenizer(char_dict)
    from tomato.models.bert import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(model_args.decoder_bert_model_name)

    def load_dataset_split(data_args, split, bert_tokenizer, char_tokenizer, train_mode=True):
        class_name = data_args.dataset_class
        DatasetClass = get_dataset_class(class_name)
        # TODO: reload tokenizer?
        split_dataset = DatasetClass(args=data_args,
                                            split=split,
                                            bert_tokenizer=bert_tokenizer, char_tokenizer=char_tokenizer,
                                            train_mode=train_mode)
        if split == "train":
            split_dataset.shuffle()
        from torch.utils.data import DataLoader
        split_dataloader = DataLoader(split_dataset, 
                                batch_size=1, 
                                shuffle=True, 
                                num_workers=data_args.num_workers,
                                collate_fn=DatasetClass.collate_fn)
        return split_dataset, split_dataloader

    # TODO: add splits to config
    if is_infer:
        splits = ["test"]
    else:
        splits = ["dev", "train"]
    all_dataset = {}
    all_dataloader = {}
    for split in splits:
        train_mode = (split == "train")
        logger.info(f"Loading {split} dataset")
        dataset, dataloader = load_dataset_split(data_args, split, bert_tokenizer, char_tokenizer, train_mode)
        all_dataset[split] = dataset
        all_dataloader[split] = dataloader
    #test_data_loader(all_dataset["train"], all_dataloader["train"])
    return (all_dataset, all_dataloader)


def load_model(config_path, cuda_device=0):
    model_args = utils.config2arg(config_path, "model")
    model_args.cuda = cuda_device
    from tomato.models import get_model_class
    ModelClass = get_model_class(model_args.model_class)
    model = ModelClass(model_args)
    #print("===================== now verify checkpoint =====================")
    #verify_checkpoint(ckpt_path, model)
    return model


def load_decoder(model, exp_name):
    from tomato.decoders import get_decoder_class 
    DecoderClass = get_decoder_class(model.decoder_class)
    asr_decoder = DecoderClass(model, exp_name)
    return asr_decoder


#TODO
def load_old_criterion(config_path, model):
    loss_args = utils.config2arg(config_path, "criterion")
    from tomato.criteria import get_criterion_class
    CriterionClass = get_criterion_class(loss_args.model_class)
    ctc_two_way_loss = CriterionClass(loss_args, model.bert_target_dictionary, model.character_target_dictionary)
    return ctc_two_way_loss


def load_criterion(config_path, model):
    loss_args = utils.config2arg(config_path, "criterion")
    from tomato.criteria import get_criterion_class
    CriterionClass = get_criterion_class(loss_args.model_class)
    loss = CriterionClass(loss_args)
    return loss 

import os
if __name__ == "__main__":
    def save_all_lfcc_feat():
        config_path = os.environ["config_path"] 
        from tqdm import tqdm
        def save_feat(dataloader):
            for batch in dataloader:
                continue

        for val in [True, False]:
            all_dataset, all_dataloader = load_data(config_path, is_infer=val)
            for key in all_dataloader:
                print(f"Saving {key} dataset...")
                save_feat(all_dataloader[key])
    save_all_lfcc_feat()