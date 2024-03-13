# Here you'll find things that are useful but 
# have more specific utilities

import numpy as np
import copy

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time

####################################
######## Useful functions ##########
####################################

def set_device(use_GPU=True, verbose=True):
    """
    Set torch.cuda device to use.
    RH 2021

    Args:
        use_GPU (int):
            If 1, use GPU.
            If 0, use CPU.
    """
    if use_GPU:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            print("no GPU available. Using CPU.") if verbose else None
        else:
            print(f"device: '{device}'") if verbose else None
    else:
        device = "cpu"
        print(f"device: '{device}'") if verbose else None

    return device
    
def squeeze_integers(arr):
    """
    Make integers in an array consecutive numbers
     starting from 0. ie. [7,2,7,4,1] -> [3,2,3,1,0].
    Useful for removing unused class IDs from y_true
     and outputting something appropriate for softmax.
    RH 2021

    Args:
        arr (np.ndarray):
            array of integers.
    
    Returns:
        arr_squeezed (np.ndarray):
            array of integers with consecutive numbers
    """
    uniques = np.unique(arr)
    arr_squeezed = copy.deepcopy(arr)
    for val in np.arange(0, np.max(arr)+1):
        if np.isin(val, uniques):
            continue
        else:
            arr_squeezed[arr_squeezed>val] = arr_squeezed[arr_squeezed>val]-1
    return arr_squeezed
    
    
####################################
###### DataLoader functions ########
####################################   

class WindowedDataset(Dataset):
    def __init__(self, X_untiled, y_input, win_range, transform=None, target_transform=None):
        self.X_untiled = X_untiled # first dim will be subsampled from
        self.y_input = y_input # first dim will be subsampled from
        self.win_range = win_range
        self.n_samples = y_input.shape[0]
        self.usable_idx = torch.arange(-self.win_range[0] , self.n_samples-self.win_range[1]+1)
        
        if X_untiled.shape[0] != y_input.shape[0]:
            raise ValueError('RH: X and y must have same first dimension shape')

    def __len__(self):
        return self.n_samples
    
    def check_bound_errors(self, idx):
        idx_toRemove = []
        for val in idx:
            if (val+self.win_range[0] < 0) or (val+self.win_range[1] > self.n_samples):
                idx_toRemove.append(val)
        if len(idx_toRemove) > 0:
            raise ValueError(f'RH: input idx is too close to edges. Remove idx: {idx_toRemove}')

    def __getitem__(self, idx):
#         print(idx)
#         self.check_bound_errors(idx)
        X_subset_tiled = self.X_untiled[idx+self.win_range[0] : idx+self.win_range[1]]
        y_subset = self.y_input[idx]
        return X_subset_tiled, y_subset

def make_WindowedDataloader(X, y, win_range=[-10,10], batch_size=64, drop_last=True, **kwargs_dataloader):
    dataset = WindowedDataset(X, y, win_range)

    sampler = torch.utils.data.SubsetRandomSampler(dataset.usable_idx, generator=None)
    
    
    if kwargs_dataloader is None:
        kwargs_dataloader = {'shuffle': False,
                             'pin_memory': False,
                             'num_workers':0
                            }
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=drop_last,
                            sampler=sampler,
                            **kwargs_dataloader,
                            )
    dataloader.sample_shape = [dataloader.batch_size] + list(dataset[-win_range[0]][0].shape)
    return dataloader, dataset, sampler




def test():
    print('hi')