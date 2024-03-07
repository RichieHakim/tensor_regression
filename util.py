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


class Convergence_checker:
    """
    Checks for convergence during an optimization process. Uses Ordinary Least Squares (OLS) to 
     fit a line to the last 'window_convergence' number of iterations.

    RH 2022
    """
    def __init__(
        self,
        tol_convergence=-1e-2,
        fractional=False,
        window_convergence=100,
        mode='greater',
        max_iter=None,
        max_time=None,
    ):
        """
        Initialize the convergence checker.
        
        Args:
            tol_convergence (float): 
                Tolerance for convergence.
                Corresponds to the slope of the line that is fit.
                If fractional==True, then tol_convergence is the fractional
                 change in loss over the window_convergence
            fractional (bool):
                If True, then tol_convergence is the fractional change in loss
                 over the window_convergence. ie: delta(lossWin) / mean(lossWin)
            window_convergence (int):
                Number of iterations to use for fitting the line.
            mode (str):
                Where deltaLoss = loss[current] - loss[window convergence steps ago]
                For typical loss curves, deltaLoss should be negative. So common
                 modes are: 'greater' with tol_convergence = -1e-x, and 'less' with
                 tol_convergence = 1e-x.
                Mode for how criterion is defined.
                'less': converged = deltaLoss < tol_convergence (default)
                'abs_less': converged = abs(deltaLoss) < tol_convergence
                'greater': converged = deltaLoss > tol_convergence
                'abs_greater': converged = abs(deltaLoss) > tol_convergence
                'between': converged = tol_convergence[0] < deltaLoss < tol_convergence[1]
                    (tol_convergence must be a list or tuple, if mode='between')
            max_iter (int):
                Maximum number of iterations to run for.
                If None, then no maximum.
            max_time (float):
                Maximum time to run for (in seconds).
                If None, then no maximum.
        """
        self.window_convergence = window_convergence
        self.tol_convergence = tol_convergence
        self.fractional = fractional

        self.line_regressor = torch.cat((torch.linspace(0,1,window_convergence)[:,None], torch.ones((window_convergence,1))), dim=1)

        if mode=='less':          self.fn_criterion = (lambda diff: diff < self.tol_convergence)
        elif mode=='abs_less':    self.fn_criterion = (lambda diff: abs(diff) < self.tol_convergence)
        elif mode=='greater':     self.fn_criterion = (lambda diff: diff > self.tol_convergence)
        elif mode=='abs_greater': self.fn_criterion = (lambda diff: abs(diff) > self.tol_convergence)
        elif mode=='between':     self.fn_criterion = (lambda diff: self.tol_convergence[0] < diff < self.tol_convergence[1])
        assert self.fn_criterion is not None, f"mode '{mode}' not recognized"

        self.max_iter = max_iter
        self.max_time = max_time

        self.iter = -1

    def OLS(self, y):
        """
        Ordinary least squares.
        Fits a line and bias term (stored in self.line_regressor)
         to y input.
        """
        X = self.line_regressor
        theta = torch.inverse(X.T @ X) @ X.T @ y
        y_rec = X @ theta
        bias = theta[-1]
        theta = theta[:-1]

        return theta, y_rec, bias

    def __call__(
        self,
        loss_history=None,
        loss_single=None,
    ):
        """
        Forward pass of the convergence checker.
        Checks if the last 'window_convergence' number of iterations are
         within 'tol_convergence' of the line fit.

        Args:
            loss_history (list or array):
                List of loss values for entire optimization process.
                If None, then internally tracked loss_history is used.
            loss_single (float):
                Single loss value for current iteration.

        Returns:
            delta_window_convergence (float):
                Difference between first and last element of the fit line
                 over the range of 'window_convergence'.
                 diff_window_convergence = (y_rec[-1] - y_rec[0])
            loss_smooth (float):
                The mean loss over 'window_convergence'.
            converged (bool):
                True if the 'diff_window_convergence' is less than
                 'tol_convergence'.
        """
        if self.iter == 0:
            self.t0 = time.time()
        self.iter += 1

        if loss_history is None:
            if not hasattr(self, 'loss_history'):
                assert loss_single is not None, "loss_history and loss_single are both None"
                self.loss_history = []
            self.loss_history.append(loss_single)
            loss_history = self.loss_history

        if len(loss_history) < self.window_convergence:
            return torch.nan, torch.nan, False
        loss_window = torch.as_tensor(loss_history[-self.window_convergence:], device='cpu', dtype=torch.float32)
        loss_smooth = loss_window.mean()

        theta, y_rec, bias = self.OLS(y=loss_window)

        delta_window_convergence = (y_rec[-1] - y_rec[0]) if not self.fractional else (y_rec[-1] - y_rec[0]) / ((y_rec[-1] + y_rec[0])/2)
        converged = self.fn_criterion(delta_window_convergence)

        if self.max_iter is not None:
            converged = converged or (len(loss_history) >= self.max_iter)
        if self.max_time is not None:
            converged = converged or (time.time() - self.t0 > self.max_time)
        
        return delta_window_convergence.item(), loss_smooth.item(), converged



def test():
    print('hi')