import os
import pandas as pd
from pathlib import Path
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms as T


def diagnosis(mask_path):
    return 1 if np.max(cv2.imread(mask_path)) > 0 else 0

def read_data_to_DataFrame(ROOT_PATH=None):
    #ROOT_PATH = '../lgg-mri-segmentation/kaggle_3m/'
    mask_files = glob.glob(ROOT_PATH + '*/*_mask*')
    image_files = [file.replace('_mask', '') for file in mask_files]
    df = pd.DataFrame({"image_path": image_files,
                  "mask_path": mask_files,
                  "diagnosis": [diagnosis(x) for x in mask_files]})

    train_df, test_df = train_test_split(df, stratify=df['diagnosis'], test_size=0.1)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df, val_df = train_test_split(train_df, stratify=train_df['diagnosis'], test_size=0.15)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    
    
    return train_df, val_df, test_df


def read_data_from_csv(ROOT_PATH=None):
    train_df = pd.read_csv(os.path.join(ROOT_PATH, "train.csv"))
    val_df = pd.read_csv(os.path.join(ROOT_PATH, "val.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_PATH, "test.csv"))
    return train_df, val_df, test_df

def build_transform(mode='train'):
    if mode == 'train':
        train_transform = A.Compose([
            A.Resize(width=128, height=128, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        ])
        return train_transform
    if mode == 'val':
        val_transform = A.Compose([
            A.Resize(width=128, height=128, p=1.0),
            A.HorizontalFlip(p=0.5),
        ])
        return val_transform
    if mode == 'test':
        test_transform = A.Compose([
            A.Resize(width=128, height=128, p=1.0)
        ])
        return test_transform

    return None


class BrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0])
        image = np.array(image)/255.
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        mask = np.array(mask)/255.
        
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
            
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).type(torch.float32)
        image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        mask = np.expand_dims(mask, axis=-1).transpose((2,0,1))
        mask = torch.from_numpy(mask).type(torch.float32)
        
        return image, mask


def build_dataset(train_df, val_df, test_df):
    train_transform = build_transform(mode = 'train')
    train_dataset = BrainDataset(train_df, train_transform)
    

    val_transform = build_transform(mode = 'val')
    val_dataset = BrainDataset(val_df, val_transform)
    

    test_transform = build_transform(mode = 'test')
    test_dataset = BrainDataset(test_df, test_transform)
    
    return train_dataset, val_dataset, test_dataset

