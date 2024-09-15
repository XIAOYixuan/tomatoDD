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
  This parameter is used to group results from the same model across different datasets.  
  It will create a directory named `exp` under `./output/infer/`, and store all the output results from this job in that directory.

- **`-ckpt [checkpoint directory path]`**:  
  The path to the directory containing the model checkpoints.

- **`-ckpt_tag [best/last]`**:  
  Choose whether to use the `best` or `last` checkpoint model for inference.

- **`-tag [job name]`**:  
  Assign a name to this job.  
  It will create a `.csv` file named after `tag` under `./output/infer/[exp]` ('exp' is defined by -exp) to store the results of this job.

- **`-s [split]`**:  
  Specify which data split to use for inference.  (This repository is still under development, to avoid potential crashes, please only use the `test` split.)
