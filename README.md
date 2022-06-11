# Brain-MRI-segmentation

## Install reuqired package
To install the required package run the command below.
```
pip install -r requirements.txt
```


## Usage
To run the model with `Unet` and `resnet50` encoder run the command below
```
python train.py
``` 

## Script option
The argpase for the training option for this script is is written inside the train.py with argparse. Something different is that instead of type in the option right after the python command, I use a list call `config` to store all the option. To add new option, you can go inside the `train.py` and modify the `config` to add something you want. Below is what the `config` variable inside `train.py` will look like.
```
config=[
        '--data-path', '../lgg-mri-segmentation/kaggle_3m/',
        '--epochs' , '100',
        '--output_dir', 'Unet',
        '--lr', '1e-4',
        '--weight-decay','0.05',
    ]
```

For the other option that we may use in this project is list below :
+ `data-path` : the path to the dataset
+ `output_dir` : the path of the output directory to store the log and checkpoint
+ `seg_struct` : the structure of the segmentation structure
    + Currently support
        + `Unet`
        + `Unet++`
+ `encoder` : The encoder that will be use in the `seg_struct`
    + [Reference here](https://smp.readthedocs.io/en/latest/encoders.html)
+ `opt` : The optimizer for training
    + Currently support
        + `adam`
        + `adamw`
+ `lr` : learning rate
+ `weight-decay` : weight decay for training

To restore from the checkpoint, the following option can be use.
+ `resume` : The path to the checkpoint
+ `eval` : For evaluation only, will not run the training procedure.
For instance, if we want te load the weight from the checkpoint `./checkpoint.pth` and do the evaluation, the following list can be written as:
```
config=[
        '--data-path', '../lgg-mri-segmentation/kaggle_3m/',
        '--epochs' , '100',
        '--output_dir', 'Unet',
        '--resume', './checkpoint.pth',
        '--eval'
    ]
```