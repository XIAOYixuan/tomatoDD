# tomatoDD

This is the official repository for our research work on fake audio detection at the IMS.

This branch (interspeech25-layer-wise) is for our Interspeech25 paper:

Layer-Wise Decision Fusion for Fake Audio Detection Using XLS-R

## fairseq model download

There two ways to get the pretrained model.

### Download from fairseq's website.
After downloading the pretrained model, please use the tool to convert the format that we use.

Pretrained models: XLS-R 300M
Download it on fairseq's [website](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/xlsr/README.md).

Then, use the tool to convert the format that we use.
Notice that you need to install [fairseq](https://github.com/facebookresearch/fairseq) to load the pretrained model. To avoid any potential issues, it's better to install our [modified fairseq](https://github.com/XIAOYixuan/fairseq_ADD).

```
% install fairseq
cd [fairseq path]
pip install --editable ./  

% convert the pretrained model
cd [this repo path]
python tools/fairseq_model_converter.py [path to the pretrained model] [output directory]
```
The new model pt would be saved in the output directory as ```[output directory]/model.pt```

### Download our converted model on HuggingFace.

You can find it at [link](https://huggingface.co/Yixuan/fairseq_models/tree/main)

## train

The `train.py` script accepts the following parameters for training tasks:

```bash
python train.py     
    -c [config file path]   
    -exp [experiment tag]     
    -task [task]     
    -s [save directory]  
    -ckpt [optional: checkpoint directory path]     
```

- **`-c [config file path]`**: 
  Specify the config file path for the training.
- **`-exp [save directory]`**: 
  Specify the exp's name for the training results. All training results will be saved under `./output/[s]/[exp]`.
- **`-task [task]`**:  
  Specify which task class should be used for the training. 'oc' for one-class learning, 'xent' for cross-entropy loss.
- **`-s [save directory]`**: 
  Specify the save directory for the training results. Is useful when you want to run the same experiments on different servers, and want to see if hardware has any impact on the results.
- **`-ckpt [checkpoint directory path]`**: 
  Specify the checkpoint directory path for the training. Used for resuming training from a checkpoint.

## infer

The `infer.py` script accepts the following parameters for inference tasks:

```bash
python infer.py     
    -task [task]     
    -c [config file path]     
    -exp [experiment tag]     
    -ckpt [checkpoint directory path]     
    -ckpt_tag [best/last]     
    -tag [job name]     
    -s [split]
```


- **`-task [task]`**:  
  Specify which task class should be used for the inference. 

- **`-c [config file path]`**:  
  The path to the config file.

- **`-exp [experiment tag]`**:
  This parameter should be removed in the future, it's designed for training. Just set it to "infer" for now.

- **`-ckpt [checkpoint directory path]`**:  
  The path to the directory containing the model checkpoints.

- **`-ckpt_tag [best/last]`**:  
  Choose whether to use the `best` or `last` checkpoint model for inference.

- **`-o`**:
  All inference results will be stored under this directory. Every run will generate a csv file containing prediction score and label for each audio sample.

- **`-tag [job name]`**:  
  Assign a name to this job.  
  It will create a `.csv` file named after `tag` under `./output/infer/[exp]` ('exp' is defined by -exp) to store the results of this job.

- **`-s [split]`**:  
  Specify which data split to use for inference.  (This repository is still under development, to avoid potential crashes, please only use the `test` split.)

```bash
$ python infer.py -task xent -c example/example.yaml -exp infer -ckpt output/ckpts/[dir] -ckpt_tag best -o output/infer/[dir] -tag [tag_name]  -s test
```
