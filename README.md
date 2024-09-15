# tomatoDD

This is the official repository for our research work on fake audio detection at the IMS.

Code and model ckpt for ICASSP24 submission is on the way ...

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
  Specify which task class should be used for the inference. (Since the project is still under developement, please use 'xent' only)

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
