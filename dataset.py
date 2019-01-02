import os
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

class FrameFolderDataset(Dataset):
    def __init__(self, root_dir, clip_size=16, clip_stride=16):
        self.root_dir = root_dir
        self.clip_size = clip_size
        self.clip_stride = clip_stride
        self.data = self._generate_lists()
        self.mean = np.load('c3d_mean.npy') # (N, C, T, H, W)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # return self.data[idx]
        return self._get_clip(idx)
    
    def _generate_lists(self):
        filenames = np.array(os.listdir(self.root_dir))
        num_files = len(filenames)
        if num_files < self.clip_size:
            warnings.warn('Total number of images is insufficient')
        
        clip_start_ids = np.arange(0, num_files, self.clip_stride)
        clip_start_ids = clip_start_ids[clip_start_ids + self.clip_size <= num_files]
        clips_ids = np.array([np.arange(i, i + self.clip_size, dtype=np.int) for i in clip_start_ids])

        data = []
        for clip_id in clips_ids:
            img_paths = [os.path.join(self.root_dir, filename) for filename in filenames[clip_id]]
            data.append(img_paths)

        return data
    
    def _get_clip(self, idx):
        img_paths = self.data[idx]
        imgs = np.array([np.array(Image.open(path).resize((171, 128), Image.BICUBIC), dtype=np.float) for path in img_paths])
        imgs = imgs.transpose(3, 0, 1, 2) # (T, H, W, C) => (C, T, H, W)
        imgs = (imgs - self.mean[0])[:, :, 8:120, 30:142]
        # imgs = imgs[:, :, 8:120, 30:142]
        imgs = torch.tensor(imgs, dtype=torch.float32)

        return imgs

class ImageFramesDataset(Dataset):
    def __init__(self, root_dir, num_instances=32, clip_size=16, clip_stride=1):
        self.root_dir = root_dir
        self.num_instances = num_instances
        self.clip_size = clip_size
        self.clip_stride = clip_stride
        self.data = self._generate_lists()
    
    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        return self.data[idx]

    def _generate_lists(self):
        filenames = os.listdir(self.root_dir)
        if len(filenames) < (self.num_instances * self.clip_size):
            warnings.warn('total number of videos is insufficient')
        
        data = []
        filenames_split = np.array_split(filenames, self.num_instances)
        for files_instance in filenames_split:
            num_files_in_instance = len(files_instance)
            clip_start_ids = np.arange(0, num_files_in_instance, self.clip_stride)
            clip_start_ids = clip_start_ids[clip_start_ids + self.clip_size <= num_files_in_instance]
            clips_ids = np.array([np. arange(i, i + self.clip_size, dtype=np.int) for i in clip_start_ids])
            
            instance_path = []
            for clip in clips_ids:
                img_paths = [os.path.join(self.root_dir, path) for path in files_instance[clip]]
                instance_path.append(img_paths)
            data.append(instance_path)

        return data

class SegmentDataset(Subset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.mean = np.load('c3d_mean.npy') # (N, C, T, H, W)

    def __getitem__(self, idx):
        return self._get_clip(idx)
    
    def __len__(self):
        return len(self.dataset[self.indices])

    def _get_clip(self, idx):
        img_paths = self.dataset[self.indices][idx]
        imgs = np.array([np.array(Image.open(path).resize((171, 128), Image.BICUBIC), dtype=np.float) for path in img_paths])
        imgs = imgs.transpose(3, 0, 1, 2) # (T, H, W, C) => (C, T, H, W)
        imgs = (imgs - self.mean[0])[:, :, 8:120, 30:142]
        imgs = torch.tensor(imgs, dtype=torch.float32)

        return imgs