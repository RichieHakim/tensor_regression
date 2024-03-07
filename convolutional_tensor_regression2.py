from typing import List, Optional, Union
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import opt_einsum

import bnpm

from .util import Convergence_checker

    
@torch.jit.script
def conv_timedomain(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # X  ## shape: (n, t_x)
    # K  ## shape: (k, 2, t_k)
    n_imaginary = K.shape[1]
    K = K.permute(2, 0, 1)  ## shape: (t_k, k, 2)
    K = K.reshape(K.shape[0], -1)  ## shape: (t_k, k * 2)
    K = K.permute(1, 0)  ## shape: (k * 2, t_k)
    c = torch.nn.functional.conv1d(
            X[:, None, :],  ## shape: (n,     1, t_x)
            K[:, None, :],  ## shape: (k * 2, 1, t_k)
            padding='same',
        )  ## shape: (n, k * 2, t_x)
    if n_imaginary > 1:
        c = c.permute(2, 0, 1)  ## shape: (t_x, n, k * 2)
        c = c.reshape(c.shape[0], c.shape[1], c.shape[2]//n_imaginary, n_imaginary)  ## shape: (t_x, n, k, 2)
        if c.shape[3] > 1:
            c = torch.linalg.norm(
                c,
                dim=3,
            )  ## shape: (t_x, n, k)
            c = c.permute(1, 2, 0)  ## shape: (n, k, t_x)
        elif c.shape[3] == 1:
            c = c.squeeze(3)
    return c  ## shape: (n, k, t_x)

# @torch.jit.script
def next_fast_len(size: int):
    """
    Taken from PyTorch Forecasting:
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Implementation from pyro

    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining = remaining // n
        if remaining == 1:
            return next_size
        next_size += 1

# @torch.jit.script
def apply_padding_mode(
    conv_result: torch.Tensor, 
    x_length: int, 
    y_length: int, 
    mode: str = "valid",
) -> torch.Tensor:
    """
    This is adapted from torchaudio.functional._apply_convolve_mode. \n
    NOTE: This function has a slight change relative to torchaudio's version.
    For mode='same', ceil rounding is used. This results in fftconv matching the
    result of conv1d. However, this then results in it not matching the result of
    scipy.signal.fftconvolve. This is a tradeoff. The difference is only a shift
    in 1 sample when y_length is even. This phenomenon is a result of how conv1d
    handles padding, and the fact that conv1d is actually cross-correlation, not
    convolution. \n

    RH 2024

    Args:
        conv_result (torch.Tensor):
            Result of the convolution.
            Padding applied to last dimension.
        x_length (int):
            Length of the first input.
        y_length (int):
            Length of the second input.
        mode (str):
            Padding mode to use.

    Returns:
        torch.Tensor:
            Result of the convolution with the specified padding mode.
    """
    n = x_length + y_length - 1
    valid_convolve_modes = ["full", "valid", "same"]
    if mode == "full":
        return conv_result
    elif mode == "valid":
        len_target = max(x_length, y_length) - min(x_length, y_length) + 1
        idx_start = (n - len_target) // 2
        return conv_result[..., idx_start : idx_start + len_target]
    elif mode == "same":
        # idx_start = (conv_result.size(-1) - x_length) // 2  ## This is the original line from torchaudio
        idx_start = math.ceil((n - x_length) / 2)  ## This line is different from torchaudio
        return conv_result[..., idx_start : idx_start + x_length]
    else:
        raise ValueError(f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}.")


# @torch.jit.script
def fftconvolve(
    x: torch.Tensor, 
    y: torch.Tensor, 
    mode: str='valid',
    n: Optional[int]=None,
    fast_length: bool=False,
    x_fft: Optional[torch.Tensor]=None,
):
    """
    Convolution using the FFT method. \n
    This is adapted from of torchaudio.functional.fftconvolve that handles
    complex numbers. Code is added for handling complex inputs. \n
    NOTE: For mode='same' and y length even, torch's conv1d convention is used,
    which pads 1 more at the end and 1 fewer at the beginning (which is
    different from numpy/scipy's convolve). See apply_padding_mode for more
    details. \n

    RH 2024

    Args:
        x (torch.Tensor):
            First input. (signal) \n
            Convolution performed along the last dimension.
        y (torch.Tensor):
            Second input. (kernel) \n
            Convolution performed along the last dimension.
        mode (str):
            Padding mode to use. ['full', 'valid', 'same']
        fast_length (bool):
            Whether to use scipy.fftpack.next_fast_len to 
             find the next fast length for the FFT.
            Set to False if you want to use backpropagation.

    Returns:
        torch.Tensor:
            Result of the convolution.
    """
    ## Compute the convolution
    if x_fft is None:
        n_original = x.shape[-1] + y.shape[-1] - 1
        # n = scipy.fftpack.next_fast_len(n_original) if fast_length else n_original
        if fast_length:
            n = next_fast_len(n if n is not None else n_original) 
        else:
            n = n_original
        x_fft = torch.fft.fft(x, n=n, dim=-1)            
    else:
        n = x_fft.shape[-1] if x_fft is not None else n
    
    y_fft = torch.fft.fft(y, n=n, dim=-1)
    f = x_fft * y_fft
    fftconv_xy = torch.fft.ifft(f, n=n, dim=-1)
    return apply_padding_mode(
        conv_result=fftconv_xy,
        x_length=x.shape[-1],
        y_length=y.shape[-1],
        mode=mode,
    )

class FFTConvolve(torch.nn.Module):
    def __init__(
        self, 
        x: Optional[torch.Tensor]=None, 
        n: Optional[int]=None, 
        next_fast_length: bool=False,
        use_x_fft: bool=True,
    ):
        super(FFTConvolve, self).__init__()
        if x is not None:
            self.set_x_fft(x=x, n=n, next_fast_length=next_fast_length)
        else:
            self.n = None
            self.x_fft = None

        self.use_x_fft = use_x_fft

    def set_x_fft(self, x: torch.Tensor, n: Optional[int]=None, next_fast_length: bool=False):
        if next_fast_length:
            self.n = next_fast_len(size=n)
        self.x_fft = torch.fft.fft(x, n=self.n, dim=-1).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str='same',
        n: Optional[int]=None,
        fast_length: Union[int, bool]=False,
        x_fft: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        x_fft = self.x_fft if x_fft is None else x_fft
        n = self.n if n is None else n
        return fftconvolve(x=x, y=y, mode=mode, n=n, fast_length=fast_length, x_fft=x_fft if self.use_x_fft else None)


@torch.jit.script
def standardizer(x: torch.Tensor, dim: int) -> torch.Tensor:
    return (x - torch.mean(x, dim=dim, keepdim=True)) / torch.std(x, dim=dim, keepdim=True)


def conv_rrr(
    X: torch.Tensor,
    K: torch.Tensor,
    B1: torch.Tensor,
    B2: torch.Tensor,
    bias: torch.Tensor,
    fftconv: Optional[FFTConvolve]=None,
    lowMem: bool=False,
) -> torch.Tensor:
    conv = fftconv if fftconv is not None else conv_timedomain
    if lowMem:
        # K  ## shape: (k, 2, t_k)
        # B1  ## shape: (n, k)
        # B2  ## shape: (k, m)
        # bias  ## shape: (m,)
        if fftconv is not None:
            K = torch.view_as_complex(K.permute(0, 2, 1).contiguous())[:, None, :] if K.shape[1] > 1 else K
            X = X[:, None, :]
            abs_wrapper = lambda x: torch.abs(x) if torch.is_complex(K) else torch.real(x)
        else:
            abs_wrapper = lambda x: x
        out = torch.cat([opt_einsum.contract('nkt, nk -> kt', standardizer(abs_wrapper(conv(X, K[ii][None, :, :])), dim=2), B1[:, ii][:, None]) for ii in range(K.shape[0])], dim=0)  ## XKL, shape: (k, t_x)
        out = torch.einsum('kt, km -> mt', out, B2)  ## XKLL, shape: (m, t_x)
        out = out + bias  ## XKLLB, shape: (m, t_x)
        return out
    else:
        # K  ## shape: (k, 2, t_k)
        # B1  ## shape: (n, k)
        # B2  ## shape: (k, m)
        # bias  ## shape: (m,)
        # out = conv(X, K)  ## XK, shape: (n, k, t_x)
        if fftconv is None:
            out = conv(X, K)  ## XK, shape: (n, k, t_x)
        else:
            K = torch.view_as_complex(K.permute(0, 2, 1).contiguous())[None, :, :] if K.shape[1] == 2 else K.permute(1, 0, 2)
            abs_wrapper = lambda x: torch.abs(x) if torch.is_complex(K) else x
            out = abs_wrapper(conv(X[:, None, :], K))  ## XK, shape: (n, k, t_x)
        out = standardizer(out, dim=2)  ## XK_s, shape: (n, k, t_x)
        # out = torch.einsum('nkt, nk -> kt', out, B1)  ## XKL, shape: (k, t_x)
        # out = torch.einsum('kt, km -> mt', out, B2)  ## XKLL, shape: (m, t_x)
        out = opt_einsum.contract('nkt, nk, km -> mt', out, B1, B2)  ## XKLL, shape: (m, t_x)
        out = out + bias  ## XKLLB, shape: (m, t_x)
        return out

class Convolutional_Reduced_Rank_Regression(torch.nn.Module):
    def __init__(
        self, 
        n_features_in: int, 
        n_features_out: int, 
        rank_normal: int=5,
        rank_complex: int=5,
        window_size: int=101,
        lr: float=0.01,
        L2_B1: float=0.01,
        L2_B2: float=0.01,
        L2_K: float=0.01,
        optimizer=torch.optim.AdamW,
        device: str='cpu',
        dtype: torch.dtype=torch.float32,
        tol_convergence: float=1e-2,
        max_iter_convergence: Optional[int]=None,
        window_convergence: int=10,
        scale_init_K_normal: float=0.1,
        scale_init_K_complex: float=0.1,
        scale_init_B1: float=0.1,
        scale_init_B2: float=0.1,
        scale_init_bias: float=0.1,
        K_normal_init: torch.Tensor=None,
        K_complex_init: torch.Tensor=None,
    ):
        """
        Convolutional Reduced Rank Regression model.
        RH 2024

        Args:
            n_features (int):
                Number of features.
            rank (int):
                Rank of each component.
            complex_dims (list of ints):
                List of the number of complex dimensions for each
                 component.
            non_negative (list of booleans):
                List of booleans indicating whether each component
                 is non-negative.
            scale (float):
                Scale of uniform distribution used to initialize
                 each component.
            device (str):
                Device to use.
        """
        super(Convolutional_Reduced_Rank_Regression, self).__init__()

        ## Assert window_size is odd
        assert window_size % 2 == 1, f"window_size must be odd, got {window_size}"
        ## Assert rank_normal and rank_complex are positive integers
        assert all([isinstance(x, int) for x in [rank_normal, rank_complex]]), f"rank_normal and rank_complex must be integers, got {rank_normal} and {rank_complex}"
        assert all([x >= 0 for x in [rank_normal, rank_complex]]), f"rank_normal and rank_complex must be positive, got {rank_normal} and {rank_complex}"
        rank_total = rank_normal + rank_complex
        ## Assert K_init is None or has the right shape
        for K_init in [K_normal_init, K_complex_init]:
            if K_init is not None:
                assert K_init.ndim == 3, f"K_init must be 3D, got {K_init.ndim}D"
                assert K_init.shape[0] == rank_total, f"K_init must have shape[0] == rank, got {K_init.shape[0]} and {rank_total}"
            
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.rank_normal = rank_normal
        self.rank_complex = rank_complex
        self.rank = rank_total
        self.window_size = window_size
        self.lr = lr
        self.L2_B1 = L2_B1
        self.L2_B2 = L2_B2
        self.L2_K = L2_K
        self.device = device
        self.dtype = dtype
        self.optimizer_type = optimizer
        
        self.converged = False
        self.epoch = 0
        self.i_batch = 0

        self.lowMem = False

        self.scale_init_K_normal, self.scale_init_K_complex, self.scale_init_B1, self.scale_init_B2, self.scale_init_bias = \
            (scale_init_K_normal, scale_init_K_complex, scale_init_B1, scale_init_B2, scale_init_bias)
        
        self.Y_std = None

        self.convergence_checker = Convergence_checker(
            tol_convergence=tol_convergence,
            fractional=True,
            window_convergence=window_convergence,
            mode='abs_less',
            max_iter=max_iter_convergence,
            max_time=None,
        )

        K_normal, K_complex, B1, B2, bias = self.init_params()
        K_normal = K_normal if K_normal_init is None else K_normal_init
        K_complex = K_complex if K_complex_init is None else K_complex_init
        self.K_normal_, self.K_complex_, self.B1_, self.B2_, self.bias_ = (torch.nn.Parameter(x, requires_grad=True) for x in [K_normal, K_complex, B1, B2, bias])

        self.optimizer = self.make_optimizer(lr=self.lr)
        self.criteria = torch.nn.MSELoss()
        self.fftconv = None
        
        self.loss_all = {}
        
        self.to(self.device)
    
    def init_params(self):
        shape_K_normal = (self.rank_normal, 1, self.window_size)
        shape_K_complex = (self.rank_complex, 2, self.window_size)
        shape_B1 = (self.n_features_in, self.rank)
        shape_B2 = (self.rank, self.n_features_out)
        shape_bias = (self.n_features_out, 1)
        K_normal, K_complex, B1, B2, bias = (torch.nn.init.orthogonal_(torch.empty(s, dtype=self.dtype, device=self.device)) for s in [shape_K_normal, shape_K_complex, shape_B1, shape_B2, shape_bias])
        K_normal, K_complex = standardizer(K_normal, dim=2) if self.rank_normal > 0 else K_normal, standardizer(K_complex, dim=2) if self.rank_complex > 0 else K_complex
        B1, B2, bias = standardizer(B1, dim=1), standardizer(B2, dim=1), standardizer(bias, dim=0)

        K_normal, K_complex = ((K / torch.std(torch.linalg.norm(K, dim=1, keepdim=True), dim=2, keepdim=True)) if K.shape[0] > 0 else K for K in [K_normal, K_complex])

        K_normal *= self.scale_init_K_normal
        K_complex *= self.scale_init_K_complex
        B1 *= self.scale_init_B1
        B2 *= self.scale_init_B2
        bias *= self.scale_init_bias
        
        return K_normal, K_complex, B1, B2, bias
    
    def make_optimizer(
        self, 
        optimizer_type: Optional[torch.optim.Optimizer]=None,
        lr: Optional[float]=None,
    ):
        self.optimizer_type = optimizer_type if optimizer_type is not None else self.optimizer_type
        self.lr = lr if lr is not None else self.lr
        self.optimizer = self.optimizer_type(self.parameters(), lr=self.lr)
        return self.optimizer

    def forward(self, X):
        Y_pred_normal  = conv_rrr(X=X, K=self.K_normal_, B1=self.B1_[:, :self.rank_normal], B2=self.B2_[:self.rank_normal, :], bias=self.bias_, fftconv=self.fftconv, lowMem=self.lowMem) if self.rank_normal > 0 else 0
        Y_pred_complex = conv_rrr(X=X, K=self.K_complex_, B1=self.B1_[:, self.rank_normal:], B2=self.B2_[self.rank_normal:, :], bias=self.bias_, fftconv=self.fftconv, lowMem=self.lowMem) if self.rank_complex > 0 else 0
        return Y_pred_normal + Y_pred_complex

    def L2_regularization(self, l2_B1: float, l2_B2: float, l2_K: float):
        penalty_B1 = (l2_B1 * torch.sum(torch.stack([torch.sum(v**2) for v in [self.B1_, self.B2_, self.bias_]])))
        penalty_B2 = (l2_B2 * torch.sum(torch.stack([torch.sum(v**2) for v in [self.B1_, self.B2_, self.bias_]])))
        penalty_K_normal = (l2_K * torch.sum(torch.stack([torch.sum(v**2) for v in [standardizer(self.K_normal_, dim=2)[0]]]))) if self.rank_normal > 0 else 0
        penalty_K_complex = (l2_K * torch.sum(torch.stack([torch.sum(v**2) for v in [torch.linalg.norm(standardizer(self.K_complex_, dim=2), dim=1)]]))) if self.rank_complex > 0 else 0
        return penalty_B1 + penalty_B2 + penalty_K_normal + penalty_K_complex

    def train_step(self, X, Y):
        ## If optimizer is LBFGS, use closure
        torch.cuda.synchronize(device=None)
        if self.optimizer_type == torch.optim.LBFGS:
            def closure():
                self.optimizer.zero_grad()
                Y_hat = self.forward(X)
                loss = self.criteria(Y_hat, Y) + self.L2_regularization(l2_B1=self.L2_B1, l2_B2=self.L2_B2, l2_K=self.L2_K)
                loss.backward()
                self.loss = loss.item()
                # self.loss = 0
                return loss
            self.optimizer.step(closure)
            return self.loss
        else:
            self.optimizer.zero_grad()
            Y_hat = self.forward(X)
            loss = self.criteria(Y_hat, Y) + self.L2_regularization(l2_B1=self.L2_B1, l2_B2=self.L2_B2, l2_K=self.L2_K)
            loss.backward()
            self.optimizer.step()
            return loss.item()
            # return 0

    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        n_epochs: int=1000,
        batched=True,
        fftconv: bool=False,
        lowMem=False,
        dataset_kwargs: dict={
            'batch_size': 50000,
            'min_batch_size': -1,
            'randomize_batch_indices': True,
            'shuffle_batch_order': True,
            'shuffle_iterable_order': False,        
        },
        dataloader_kwargs: dict={
            'pin_memory': False,
            'num_workers': 0,
            'persistent_workers': False,
            'prefetch_factor': 2,
        },
        reset_optimizer_epoch_schedule: List[Optional[int]]=[0,],
        print_every: int=-1,
        plot_every: int=-1,
        plot_updateLimits_every: int=3,
    ):
        assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
        assert Y.ndim == 2, f"Y must be 2D, got {Y.ndim}D"
        assert X.shape[0] == self.n_features_in, f"X must have {self.n_features_in} features, got {X.shape[0]}"
        assert Y.shape[0] == self.n_features_out, f"Y must have {self.n_features_out} features, got {Y.shape[0]}"
        assert X.shape[1] == Y.shape[1], f"X and Y must have the same number of samples, got {X.shape[1]} and {Y.shape[1]}"

        self.lowMem = lowMem
        
        X, Y = (torch.as_tensor(arr, dtype=self.dtype, device='cpu' if batched else self.device) for arr in [X, Y])
        # X, Y = (standardizer(arr, dim=1) for arr in [X, Y])
        X = X / torch.std(X)
        self.Y_std = torch.std(Y)
        Y = Y / self.Y_std

        self.fftconv = FFTConvolve(x=X[:, None, :], n=X.shape[-1] + Y.shape[-1] - 1, next_fast_length=True) if fftconv else None

        if batched:
            dataset = torch.utils.data.TensorDataset(X.T, Y.T)
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset, 
                sampler=bnpm.torch_helpers.BatchRandomSampler(
                    len_dataset=X.shape[1],
                    **dataset_kwargs,
                ),
                batch_size=1,
                **dataloader_kwargs,
            )

        if plot_every > 0:
            self.initialize_plot()
        def plot():
            if plot_every > 0:
                if (len(self.loss_all) % plot_every == 0):
                    self.update_plot()
                if (len(self.loss_all) % (plot_every * plot_updateLimits_every) == 0):
                    self.update_plot_limits()
        
        def print_loss(loss, loss_smooth, delta_window_convergence):
            if (len(self.loss_all) % print_every == 0) and (print_every > 0):
                print(f"Epoch {self.epoch} | i_batch: {self.i_batch} | Loss: {loss} | Loss smooth: {loss_smooth} | Delta window convergence: {delta_window_convergence}")

        def run_train_step(X, Y):
            loss = self.train_step(X=X, Y=Y)
            delta_window_convergence, loss_smooth, self.converged = self.convergence_checker(loss_single=loss)
            self.loss_all[(self.epoch, self.i_batch)] = float(loss)
            plot()
            print_loss(loss, loss_smooth, delta_window_convergence)
            if self.converged:
                print(f"Converged at epoch {self.epoch} with loss {loss}")

        for self.epoch in range(n_epochs):
            if self.epoch in reset_optimizer_epoch_schedule:
                self.optimizer = self.make_optimizer()
            if batched:
                for self.i_batch, (X, Y) in enumerate(dataloader):
                    X, Y = (arr[0].T.to(self.device) for arr in [X, Y])
                    if self.converged:
                        break
            else:
                run_train_step(X, Y)
            if self.converged:
                break
    
    def predict(self, X):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        X = X / torch.std(X)
        Y_std = self.Y_std if self.Y_std is not None else 1
        ## set FFTConvolve to not use precomputed x_fft
        with bnpm.misc.temp_set_attr(self.fftconv, 'use_x_fft', False):
            ## set requires_grad to False
            with torch.no_grad():
                ## set to eval mode
                with bnpm.torch_helpers.temp_eval(self):
                    Y_pred = self.forward(X) * Y_std
        return Y_pred
    
    def predict_single_rank(self, X, rank):
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        X = X / torch.std(X)
        Y_std = self.Y_std if self.Y_std is not None else 1
        with bnpm.misc.temp_set_attr(self.fftconv, 'use_x_fft', False):
            with torch.no_grad():
                with bnpm.torch_helpers.temp_eval(self):
                    k = self.K_normal_[rank, :, :][None, :, :] if rank < self.rank_normal else self.K_complex_[rank - self.rank_normal, :, :][None, :, :]
                    Y_pred = conv_rrr(
                        X=X, 
                        K=k, 
                        B1=self.B1_[:, rank][:, None], 
                        B2=self.B2_[rank, :][None, :], 
                        bias=self.bias_, 
                        fftconv=self.fftconv, 
                        lowMem=self.lowMem,
                    ) * Y_std
        return Y_pred

    def initialize_plot(self):
        b1, b2, k_normal, k_complex = (x.detach().cpu().numpy() for x in [self.B1_, self.B2_, self.K_normal_, self.K_complex_])

        self.fig, self.axs = plt.subplots(nrows=2 + self.rank, ncols=1, figsize=(5, 15))
        self.lines = {}
        
        self.axs[0].set_title("B1")
        self.lines[0] = self.axs[0].plot(b1.T)

        self.axs[1].set_title("B2")
        self.lines[1] = self.axs[1].plot(b2.T)

        for ii, k in enumerate(k_normal):
            self.axs[ii + 2].set_title(f"K_normal_{ii}")
            self.lines[ii + 2] = self.axs[ii + 2].plot(k.T)
        for ii, k in enumerate(k_complex):
            self.axs[ii + 2 + self.rank_normal].set_title(f"K_complex_{ii}")
            self.lines[ii + 2 + self.rank_normal] = self.axs[ii + 2 + self.rank_normal].plot(k.T)

        plt.show(block=False)
        self.update_plot()

    def update_plot(self):
        b1, b2, k_normal, k_complex = (x.detach().cpu().numpy() for x in [self.B1_, self.B2_, self.K_normal_, self.K_complex_])
        
        def update_ax(ax, lines, data):
            n_lines = len(lines)
            [lines[ii].set_ydata(data[ii]) for ii in range(n_lines)]
            ax.draw_artist(ax.patch)
            [ax.draw_artist(lines[ii]) for ii in range(n_lines)]
                    
        update_ax(self.axs[0], self.lines[0], b1)
        update_ax(self.axs[1], self.lines[1], b2)
        [update_ax(self.axs[ii + 2], self.lines[ii + 2], k) for ii, k in enumerate(k_normal)]
        [update_ax(self.axs[ii + 2 + self.rank_normal], self.lines[ii + 2 + self.rank_normal], k) for ii, k in enumerate(k_complex)]

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.pause(0.001)

    def update_plot_limits(self):
        b1, b2, k_normal, k_complex = (x.detach().cpu().numpy() for x in [self.B1_, self.B2_, self.K_normal_, self.K_complex_])
        self.axs[0].set_ylim(np.min(b1), np.max(b1))
        self.axs[1].set_ylim(np.min(b2), np.max(b2))
        [self.axs[ii + 2].set_ylim(np.min(k), np.max(k)) for ii, k in enumerate(k_normal)]
        [self.axs[ii + 2 + self.rank_normal].set_ylim(np.min(k), np.max(k)) for ii, k in enumerate(k_complex)]

    def to_device(self, device):
        self.device = device

        self.fftconv.x_fft = self.fftconv.x_fft.to(device)
        self.fftconv.to(device)

        self.to(self.device)
        return self


def _crop_same_to_valid(
    X: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """
    Crop the input to the same size as the valid convolution.
    Use conventions from pytorch for the padding indexing.
    Applies to the last dimension.
    """
    ## Even kernel size
    if kernel_size % 2 == 0:
        return X[..., kernel_size//2:-kernel_size//2]
    ## Odd kernel size
    else:
        return X[..., kernel_size//2:-(kernel_size//2)]

    
