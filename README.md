# Brain-MRI-segmentation

## Update 0613 -- Add Learning Rate Scheduler
The learning rate scheduler create from `timm` is add. 
Currently, the default scheduler is `consine`.  All of the scheduler that is supported in timm is also supported here.
For detailed and which scheduler can be use, go to [this site](https://timm.fast.ai/schedulers) for more information.


## Install reuqired package
To install the required package run the command below.
```
pip install -r requirements.txt
```

## Dataset

To prepare dataset you can use the `kaggle-api` to get the zip flie.
```
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
```
Note that if some error like command not found occur, you may need to install the `kaggle` module first. 

Or you can also download the data manually at official [kaggle website](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

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

## To Do
+ Add the option of learning rate scheduler
+ Add model Swin-Unet
+ Add ploting script for result visualization