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

    train_df, test_df = train_test_split(df, stratify=df['diagnosis'], test_size=0.2)
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
            A.Resize(width=224, height=224, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        ])
        return train_transform
    if mode == 'val':
        val_transform = A.Compose([
            A.Resize(width=224, height=224, p=1.0),
            A.HorizontalFlip(p=0.5),
        ])
        return val_transform
    if mode == 'test':
        test_transform = A.Compose([
            A.Resize(width=224, height=224, p=1.0)
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

def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    E.g. [0,0,0,1] -> 3
         [1,0,0,0] -> 1
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x

def preprocess_images(images_arr, mean_std=None):
    images_arr[images_arr > 500] = 500
    images_arr[images_arr < -1500] = -1500
    min_perc, max_perc = np.percentile(images_arr, 5), np.percentile(images_arr, 95)
    images_arr_valid = images_arr[(images_arr > min_perc) & (images_arr < max_perc)]
    mean, std = (images_arr_valid.mean(), images_arr_valid.std()) if mean_std is None else mean_std
    images_arr = (images_arr - mean) / std
    print(f'mean {mean}, std {std}')
    return images_arr, (mean, std)

def read_covid_data(ROOT_PATH='../covid-segmentation/'):
    
    images_radiopedia = np.load(os.path.join(ROOT_PATH, 'images_radiopedia.npy')).astype(np.float32)
    masks_radiopedia = np.load(os.path.join(ROOT_PATH, 'masks_radiopedia.npy')).astype(np.int8)
    images_medseg = np.load(os.path.join(ROOT_PATH, 'images_medseg.npy')).astype(np.float32)
    masks_medseg = np.load(os.path.join(ROOT_PATH, 'masks_medseg.npy')).astype(np.int8)
    test_images_medseg = np.load(os.path.join(ROOT_PATH, 'test_images_medseg.npy')).astype(np.float32)
    palette = [[0], [1], [2],[3]]
    
    masks_radiopedia_recover = onehot_to_mask(masks_radiopedia, palette).squeeze()  # shape = (H, W)
    masks_medseg_recover = onehot_to_mask(masks_medseg, palette).squeeze()  # shape = (H, W)

    images_radiopedia, mean_std = preprocess_images(images_radiopedia)
    images_medseg, _ = preprocess_images(images_medseg, mean_std)
    test_images_medseg, _ = preprocess_images(test_images_medseg, mean_std)

    masks_radiopedia_recover = onehot_to_mask(masks_radiopedia, palette).squeeze()  # shape = (H, W)
    masks_medseg_recover = onehot_to_mask(masks_medseg, palette).squeeze()  # shape = (H, W)

    val_indexes, train_indexes = list(range(24)), list(range(24, 100))

    train_images = np.concatenate((images_medseg[train_indexes], images_radiopedia))
    train_masks = np.concatenate((masks_medseg_recover[train_indexes], masks_radiopedia_recover))
    val_images = images_medseg[val_indexes]
    val_masks = masks_medseg_recover[val_indexes]

    return (train_images, train_masks), (val_images, val_masks), test_images_medseg

class CovidDataset:
    def __init__(self, images, masks, augmentations=None):
        self.images = images
        self.masks = masks
        self.augmentations = augmentations
        self.mean = [0.485]
        self.std = [0.229]
    
    def __getitem__(self, i):
        image = self.images[i]
        mask = self.masks[i]
        
        if self.augmentations is not None:
            sample = self.augmentations(image=image, mask=mask)
            # Check later for why
            image, mask = Image.fromarray(np.squeeze(sample['image'], axis=2)), sample['mask']
        
        if self.augmentations is None:
            image = Image.fromarray(image)
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        image = t(image) # Normalize images
        mask = torch.from_numpy(mask).long()
    
        return image, mask
    
    def __len__(self):
        return len(self.images)


def build_dataset(train_df, val_df, test_df):
    train_transform = build_transform(mode = 'train')
    train_dataset = BrainDataset(train_df, train_transform)
    

    val_transform = build_transform(mode = 'val')
    val_dataset = BrainDataset(val_df, val_transform)
    

    test_transform = build_transform(mode = 'test')
    test_dataset = BrainDataset(test_df, test_transform)
    
    return train_dataset, val_dataset, test_dataset


def build_covid_dataset(train_data, val_data):
    train_transform = build_transform(mode = 'train')
    train_dataset = CovidDataset(train_data[0], train_data[1], train_transform)

    val_transform = build_transform(mode = 'val')
    val_dataset = CovidDataset(val_data[0], val_data[1], val_transform)

    return train_dataset, val_dataset
