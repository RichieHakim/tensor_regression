import numpy as np
import copy
import gc
import matplotlib.pyplot as plt

import torch
import torch.cuda
from torch.autograd import Variable
from torch.optim import LBFGS

import tensorly as tl

import time

####################################
######## Helper functions ##########
####################################

def make_BcpInit(B_dims, rank, non_negative, complex_dims=None, scale=1, device='cpu', dtype=torch.float32):
    """
    Make initial Beta Kruskal tensor.
    RH 2021
    
    Args:
        B_dims (list of ints):
            List of dimensions of each component.
        rank (int):
            Rank of each component.
        non_negative (list of booleans):
            List of booleans indicating whether each component
             is non-negative.
        complex_dims (list of ints):
            List of the number of complex dimensions for each
             component.
        scale (float):
            Scale of uniform distribution used to initialize
             each component.
        device (str):
            Device to use.
    
    Returns:
        B_cp (list of torch.Tensor):
            Beta Kruskal tensor.
    """
    if complex_dims is None:
        complex_dims = list([1]*len(B_dims))
    
    B_cp = []
    for ii in range(len(B_dims)):
        B_cp.append(torch.empty(B_dims[ii], rank, complex_dims[ii], dtype=dtype, device=device))
        B_cp[-1] = torch.nn.init.orthogonal_(B_cp[-1])
        B_cp[-1] = B_cp[-1] / torch.norm(B_cp[-1], dim=0)[None,...]
        if non_negative[ii]:
            B_cp[-1] = B_cp[-1] + torch.std(B_cp[-1])*2
            B_cp[-1] = B_cp[-1] / torch.norm(B_cp[-1], dim=0)[None,...]
        if complex_dims[ii] == 1:
            B_cp[-1] = B_cp[-1].squeeze(-1)
        B_cp[-1] = B_cp[-1] * scale

    # print('shape')
    # for ii in B_cp:
    #     print(ii.shape)
    #     print(torch.var(ii, dim=0))

    return B_cp
    
def non_neg_fn(B_cp, non_negative, softplus_kwargs=None):
    """
    Apply softplus to specified dimensions of Bcp.
    Generator function that yields a list of tensors.
    RH 2021

    Args:
        B_cp (list of torch.Tensor):
            Beta Kruskal tensor (before softplus).
            List of tensors of shape
             (n_features, rank).
        non_negative (list of booleans):
            List of booleans indicating whether each component
             is non-negative.
        softplus_kwargs (dict):
            Keyword arguments for torch.nn.functional.softplus.
    
    Yields:
        B_cp_nonNegative (list of torch.Tensor):
            Beta Kruskal tensor with softplus applied to
             specified dimensions.
    """
    if softplus_kwargs is None:
        softplus_kwargs = {
            'beta': 50,
            'threshold': 1,
        }
    
    for ii in range(len(B_cp)):
        if non_negative[ii]:
            yield torch.nn.functional.softplus(B_cp[ii], **softplus_kwargs)
        else:
            yield B_cp[ii]

def gaussian(x, mu, sig , plot_pref=False):
    '''
    A gaussian function (normalized similarly to scipy's function)
    RH 2021
    
    Args:
        x (np.ndarray): 1-D array of the x-axis of the kernel
        mu (float): center position on x-axis
        sig (float): standard deviation (sigma) of gaussian
        plot_pref (boolean): True/False or 1/0. Whether you'd like the kernel plotted
        
    Returns:
        gaus (np.ndarray): gaussian function (normalized) of x
        params_gaus (dict): dictionary containing the input params
    '''

    gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x-mu)/sig, 2)/2)

    if plot_pref:
        plt.figure()
        plt.plot(x , gaus)
        plt.xlabel('x')
        plt.title(f'$\mu$={mu}, $\sigma$={sig}')
    
    params_gaus = {
        "x": x,
        "mu": mu,
        "sig": sig,
    }

    return gaus , params_gaus

# def edge_clamp(B_cp, edge_idx, clamp_val=0, device='cpu', dtype=torch.float32):
#     """
#     Clamp edges of each component of Bcp to 
#     clamp_val.
#     RH 2020
    
#     Args:
#         B_cp (list of torch.Tensor):
#             Beta Kruskal tensor.
#             List of tensors of shape
#              (n_features, rank).
    
#     Returns:
#         B_cp_clamped (list of torch.Tensor):
#             Beta Kruskal tensor with edges clamped.
#     """
#     eIdxBool = torch.ones(B_cp[0].shape[0], dtype=dtype, device=device)
#     eIdxBool[edge_idx] = clamp_val
#     return [B_cp[0]*eIdxBool[:,None,None] , B_cp[1], B_cp[2]]


# def lin_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
#     """
#     Compute the regression model.
#     y_hat = inner(X, outer(softplus(Bcp))
#     where:
#         X.shape[1:] == Bcp.shape[:-1]
#         X.shape[0] == len(y_hat)
#         Bcp.shape[-1] == len(unique(y_true))
#         softplus is performed only on specified dimensions of Bcp.
#         inner prod is performed on dims [1:] of X and
#          dims [:-1] of Bcp.
#     JZ2021 / RH 2021

#     Args:
#         X (torch.Tensor):
#             N-D array of data.
#         Bcp (list of torch.Tensor):
#             Beta Kruskal tensor (before softplus). (Bcp_n)
#             List of tensors of shape 
#              (n_features, rank).
#         weights (list of floats):
#             List of weights for each component.
#             len(weights) == rank == Bcp[0].shape[1]
#         non_negative (list of booleans):
#             List of booleans indicating whether each component
#              is non-negative.
#         bias (float):
#             Bias term. Scalar.
#         softplus_kwargs (dict):
#             Keyword arguments for torch.nn.functional.softplus.
    
#     Returns:
#         y_hat (torch.Tensor):
#             N-D array of predictions.
#     """
    
#     if Bcp[0].shape[1] == 0:
#         return torch.zeros(1).to(X.device)

#     complex_dims = X.shape[-1]

#     # print(len(Bcp))

#     # Bcp = edge_clamp(Bcp, edge_idx=torch.hstack([torch.arange(0,600), torch.arange(1320,1920)]), clamp_val=0, device=X.device, dtype=X.dtype)
#     return tl.tenalg.inner(X,
#                            tl.cp_tensor.cp_to_tensor((weights, list(non_neg_fn(
#                                                                                 [Bcp[ii][:,:,0] for ii in range(len(Bcp))],
#                                                                                 non_negative,
#                                                                                 softplus_kwargs))
#                                                      )),
#                            n_modes=X.ndim-1+complex_dims
#                         ).squeeze() + bias

# def encode(X, Bcp_e, weights, non_negative, softplus_kwargs=None):
#     """
#     Compute the regression model.
#     y_hat = inner(X, outer(softplus(Bcp))
#     where:
#         X.shape[1:] == Bcp.shape[:-1]
#         X.shape[0] == len(y_hat)
#         Bcp.shape[-1] == len(unique(y_true))
#         softplus is performed only on specified dimensions of Bcp.
#         inner prod is performed on dims [1:] of X and
#          dims [:-1] of Bcp.
#     JZ2021 / RH 2021

#     Args:
#         X (torch.Tensor):
#             N-D array of data.
#         Bcp_e (list of torch.Tensor):
#             Beta Kruskal tensor (before softplus). (Bcp_n)
#             List of tensors of shape 
#              (n_features, rank).
#         weights (list of floats):
#             List of weights for each component.
#             len(weights) == rank == Bcp[0].shape[1]
#         non_negative (list of booleans):
#             List of booleans indicating whether each component
#              is non-negative.
#         bias (float):
#             Bias term. Scalar.
#         softplus_kwargs (dict):
#             Keyword arguments for torch.nn.functional.softplus.
    
#     Returns:
#         y_hat (torch.Tensor):
#             N-D array of predictions.
#     """

#     return lin_model(X, Bcp_e, weights, non_negative, bias=0, softplus_kwargs=softplus_kwargs)

# # Calls 3 things -- encoding (innner ndim-1) with BCP_e (dimentionality of kruskal tensor X.shape[1:] or X.ndims-1 ),
# # convolution
# # decoding should be outer product -- keep things focused



# def decode(I_c, Bcp_d, weights, non_negative, bias, softplus_kwargs=None):
#     """
#     Compute the regression model.
#     y_hat = inner(X, outer(softplus(Bcp))
#     where:
#         X.shape[1:] == Bcp.shape[:-1]
#         X.shape[0] == len(y_hat)
#         Bcp.shape[-1] == len(unique(y_true))
#         softplus is performed only on specified dimensions of Bcp.
#         inner prod is performed on dims [1:] of X and
#          dims [:-1] of Bcp.
#     JZ2021 / RH 2021

#     Args:
#         X (torch.Tensor):
#             N-D array of data.
#         Bcp_d (list of torch.Tensor):
#             Beta Kruskal tensor (before softplus). (Bcp_n)
#             List of tensors of shape 
#              (n_neurons, rank).
#         weights (list of floats):
#             List of weights for each component.
#             len(weights) == rank == Bcp[0].shape[1]
#         non_negative (list of booleans):
#             List of booleans indicating whether each component
#              is non-negative.
#         bias (float):
#             Bias term. Scalar.
#         softplus_kwargs (dict):
#             Keyword arguments for torch.nn.functional.softplus.
    
#     Returns:
#         y_hat (torch.Tensor):
#             N-D array of predictions.
#     """
#     return lin_model(I_c, Bcp_d, weights, non_negative, bias=bias, softplus_kwargs=softplus_kwargs)



# @torch.jit.script
def conv(X, kernels, **conv1d_kwargs):
    """
    Convolution of X with kernels
    RH 2021

    Args:
        X (torch.Tensor):
            N-D array. Convolution will be performed
             along first dimension (columns).
            Dims 1+ are convolved independently and 
             increase the dimensionality of the output.
        kernels (torch.Tensor):
            N-D array. Convolution will be performed
             along first dimension (columns).
            Dims 1+ are convolved independently and 
             increase the dimensionality of the output.
        conv1d_kwargs (dict or keyword args):
            Keyword arguments for 
             torch.nn.functional.conv1d.
            See torch.nn.functional.conv1d for details.
            You can use padding='same'
        
    Returns:
        output (torch.Tensor):
            N-D array. Convolution of X with kernels
    """
    X_dims = list(X.shape)
    t_dim = X_dims[0]

    kernel_dims = list(kernels.shape)
    w_dim = [kernel_dims[0]]

    conv_out_shape = [-1] + X_dims[1:] + kernel_dims[1:] 
    
    X_rshp = X.reshape((t_dim, 1, -1)).permute(2,1,0) # flatten non-time dims and shape to (non-time dims, 1, time dims) (D X R, 1, t)
    kernel_rshp = kernels.reshape(w_dim + [1, -1]).permute(2,1,0) # flatten rank + complex dims and shape to (rank X complex dims, 1, W) (R X C, 1, W)

    # print(X_rshp.dtype)
    # print(kernel_rshp.dtype)
    convolved = torch.nn.functional.conv1d(X_rshp, kernel_rshp, **conv1d_kwargs)  
   
    convolved_rshp = convolved.permute(2, 0, 1).reshape((conv_out_shape)) # (T, D, R, C)

    return convolved_rshp

@torch.jit.script
def complex_magnitude(convolved):
    """
    convolved: (dots_dim x R_dim x complex_dim x (t_dim - w_dim)//stride + 1)
    dim: location of complex_dim in convolved

    RH 2021 / JZ 2021
    """
    if convolved.shape[-1] < 2:
        return convolved.squeeze(-1)
    else:
        return torch.norm(convolved, dim=-1)

# def spectral_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
#     """
#     Compute the regression model.
#     y_hat = inner(X, outer(softplus(Bcp))
#     where:
#         X.shape[1:] == Bcp.shape[:-1]
#         X.shape[0] == len(y_hat)
#         Bcp.shape[-1] == len(unique(y_true))
#             Bias term. Scalar.
#         softplus_kwargs (dict):
#             Keyword arguments for torch.nn.functional.softplus.
    
#     Returns:
#         y_hat (torch.Tensor):
#             N-D array of predictions.
#     """

#     RH 2021

#     Args:
#         X (torch.Tensor):
#             N-D array of data.
#         Bcp (list of torch.Tensor):
#             Complex Beta Kruskal tensor (before softplus). (Bcp_c)
#             List of tensors of shape 
#              (n_features, rank, complex_dimensionality).
#         weights (list of floats):
#             List of weights for each component.
#             len(weights) == rank == Bcp[0].shape[1]
#         non_negative (list of booleans):
#             List of booleans indicating whether each component
#              is non-negative.
#         bias (float):
#             Bias term. Scalar.
#         softplus_kwargs (dict):
#             Keyword arguments for torch.nn.functional.softplus.
    
#     Returns:
#         y_hat (torch.Tensor):
#             N-D array of predictions.
#     """

#     if Bcp[0].shape[1] == 0:
#         return torch.zeros(1).to(X.device)

#     y_hat_all = []
#     for ii in range(Bcp[0].shape[2]):
#         y_hat_all += [tl.tenalg.inner(X,
#                            tl.cp_tensor.cp_to_tensor((weights, list(non_neg_fn(
#                                                                                 [Bcp[0][:,:,ii]] + [ Bcp[jj][:,:,0] for jj in range(1,len(Bcp)) ],
#                                                                                 # [Bcp[0][:,:,ii] - torch.mean(Bcp[0][:,:,ii])] + [ Bcp[jj][:,:,0] for jj in range(1,len(Bcp)) ],
#                                                                                 non_negative,
#                                                                                 softplus_kwargs))
#                                                      )),
#                            n_modes=X.ndim-1
#                         ).squeeze()[None,...]]
                        
#     y_hat = torch.norm(torch.vstack(y_hat_all), dim=0) + bias
#     return y_hat


# def stepwise_linear_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
#     """
#     Computes regression model in a stepwise manner.
#     y_hat = inner(X, outer(softplus(Bcp))
#     where:
#         X.shape[1:] == Bcp.shape[:-1]
#         X.shape[0] == len(y_hat)
#         Bcp.shape[-1] == len(unique(y_true))
#         softplus is performed only on specified dimensions of Bcp.
#         inner prod is performed on dims [1:] of X and
#          dims [:-1] of Bcp.
#     RH 2021

#     Args:
#         X (torch.Tensor):
#             N-D array of data.
#         Bcp (list of torch.Tensor):
#             Beta Kruskal tensor (before softplus). (Bcp_n)
#             List of tensors of shape 
#              (n_features, rank).
#         weights (list of floats):
#             List of weights for each component.
#             len(weights) == rank == Bcp[0].shape[1]
#         non_negative (list of booleans):
#             List of booleans indicating whether each component
#              is non-negative.
#         bias (float):
#             Bias term. Scalar.
#         softplus_kwargs (dict):
#             Keyword arguments for torch.nn.functional.softplus.
    
#     Returns:
#         y_hat (torch.Tensor):
#             N-D array of predictions.
#     """

#     if Bcp[0].shape[1] == 0:
#         return torch.zeros(1).to(X.device)

#     # make non-negative
#     Bcp_nn = list(non_neg_fn(Bcp, non_negative[1:], softplus_kwargs))
    
#     # X_1 = torch.einsum('twd,wrc -> tdrc', X, Bcp_nn[0])
#     # X_1b = torch.norm(X_1, dim=3)
#     # X_2 = torch.einsum('tdr,drn -> trn', X_1b, Bcp_nn[1])
#     # X_3 = torch.einsum('trn -> tn', X_2)
#     # return X_3 + bias
    
#     # X_1a = torch.einsum('twd,wrs -> tdr', X, Bcp_nn[0])
#     # X_1b = torch.einsum('tdr,drs -> tr', X_1a, Bcp_nn[1])
#     # X_1c = torch.einsum('tr,nrs -> tn', X_1b, Bcp_nn[2]) + bias

#     X_1c = torch.einsum('tr,nrs -> tn', 
#                         torch.einsum('tdr,drs -> tr',
#                                      torch.einsum('twd,wrs -> tdr',
#                                                      X, Bcp_nn[0]),
#                                      Bcp_nn[1]), 
#                         Bcp_nn[2]) + bias
#     return X_1c

# @torch.jit.script
def slmm_helper(X, Bcp_0, Bcp_1, bias):
    # return torch.einsum('tr,nr -> tn',
    #                             torch.einsum('tdr,dr -> tr', X, Bcp_0),
    #                     Bcp_1) + bias

    return torch.sum(X * Bcp_0[None,:,:], dim=1) @ Bcp_1.T + bias
def stepwise_linear_model_multirank(X, Bcp, weights, bias):
    """
    Computes regression model in a stepwise manner.
    y_hat = inner(X, outer(softplus(Bcp))
    where:
        X.shape[1:] == Bcp.shape[:-1]
        X.shape[0] == len(y_hat)
        Bcp.shape[-1] == len(unique(y_true))
        softplus is performed only on specified dimensions of Bcp.
        inner prod is performed on dims [1:] of X and
         dims [:-1] of Bcp.
    RH 2021

    Args:
        X (torch.Tensor):
            N-D array of data.
        Bcp (list of torch.Tensor):
            Beta Kruskal tensor (before softplus). (Bcp_n)
            List of tensors of shape 
             (n_features, rank).
        weights (list of floats):
            List of weights for each component.
            len(weights) == rank == Bcp[0].shape[1]
        non_negative (list of booleans):
            List of booleans indicating whether each component
             is non-negative.
        bias (float):
            Bias term. Scalar.
        softplus_kwargs (dict):
            Keyword arguments for torch.nn.functional.softplus.
    
    Returns:
        y_hat (torch.Tensor):
            N-D array of predictions.
    """

    if Bcp[0].ndim == 1:
        for ii, bcp in enumerate(Bcp):
            Bcp[ii] = bcp[:,None]
    if X.ndim < 3:
        X = X[...,None]
    if Bcp[0].shape[1] == 0:
        return torch.zeros(1).to(X.device)

    return slmm_helper(X, Bcp[0], Bcp[1], bias)

# def stepwise_linear_model_singlerank(X, Bcp, weights, bias):
#     """
#     Computes regression model in a stepwise manner.
#     y_hat = inner(X, outer(softplus(Bcp))
#     where:
#         X.shape[1:] == Bcp.shape[:-1]
#         X.shape[0] == len(y_hat)
#         Bcp.shape[-1] == len(unique(y_true))
#         softplus is performed only on specified dimensions of Bcp.
#         inner prod is performed on dims [1:] of X and
#          dims [:-1] of Bcp.
#     RH 2021

#     Args:
#         X (torch.Tensor):
#             N-D array of data.
#         Bcp (list of torch.Tensor):
#             Beta Kruskal tensor (before softplus). (Bcp_n)
#             List of tensors of shape 
#              (n_features, rank).
#         weights (list of floats):
#             List of weights for each component.
#             len(weights) == rank == Bcp[0].shape[1]
#         non_negative (list of booleans):
#             List of booleans indicating whether each component
#              is non-negative.
#         bias (float):
#             Bias term. Scalar.
#         softplus_kwargs (dict):
#             Keyword arguments for torch.nn.functional.softplus.
    
#     Returns:
#         y_hat (torch.Tensor):
#             N-D array of predictions.
#     """

#     # if Bcp[0].shape[1] == 0:
#     #     return torch.zeros(1).to(X.device)

#     # print(X.shape)
#     # print(Bcp[0].shape)
#     # print(Bcp[1].shape)

#     # X_1c = torch.einsum('t,n -> tn',
#     #                             torch.einsum('td,d -> t', X, Bcp[0]),
#     #                     Bcp[1]) + bias
#     # print(weights)
#     weights = weights[1][None]
#     # print('hi')
#     # print(weights)
#     # X_1b = tl.cp_tensor.cp_to_tensor((weights, [Bcp[0][:,None], Bcp[1][:,None]]))
#     X_1c = tl.tenalg.inner(X, tl.cp_tensor.cp_to_tensor((weights, Bcp)), n_modes=X.ndim-1) + bias
#     # [tl.tenalg.inner(X,
# #                            tl.cp_tensor.cp_to_tensor((weights, list(non_neg_fn(
# #                                                                                 [Bcp[0][:,:,ii]] + [ Bcp[jj][:,:,0] for jj in range(1,len(Bcp)) ],
# #                                                                                 # [Bcp[0][:,:,ii] - torch.mean(Bcp[0][:,:,ii])] + [ Bcp[jj][:,:,0] for jj in range(1,len(Bcp)) ],
# #                                                                                 non_negative,
# #                                                                                 softplus_kwargs))
# #                                                      )),
# #                            n_modes=X.ndim-1
# #                         ).squeeze()[None,...]] + bias
#     return X_1c


# def stepwise_latents_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
#     """
#     Computes regression model in a stepwise manner.
#     y_hat = inner(X, outer(softplus(Bcp))
#     where:
#         X.shape[1:] == Bcp.shape[:-1]
#         X.shape[0] == len(y_hat)
#         Bcp.shape[-1] == len(unique(y_true))
#         softplus is performed only on specified dimensions of Bcp.
#         inner prod is performed on dims [1:] of X and
#          dims [:-1] of Bcp.
#     RH 2021

#     Args:
#         X (torch.Tensor):
#             N-D array of data.
#         Bcp (list of torch.Tensor):
#             Beta Kruskal tensor (before softplus). (Bcp_n)
#             List of tensors of shape 
#              (n_features, rank).
#         weights (list of floats):
#             List of weights for each component.
#             len(weights) == rank == Bcp[0].shape[1]
#         non_negative (list of booleans):
#             List of booleans indicating whether each component
#              is non-negative.
#         bias (float):
#             Bias term. Scalar.
#         softplus_kwargs (dict):
#             Keyword arguments for torch.nn.functional.softplus.
    
#     Returns:
#         y_hat (torch.Tensor):
#             N-D array of predictions.
#     """

#     if Bcp[0].shape[1] == 0:
#         return torch.zeros(1).to(X.device)

#     # make non-negative
#     Bcp_nn = list(non_neg_fn(Bcp, non_negative, softplus_kwargs))
    
#     # X_1 = torch.einsum('twd,wrc -> tdrc', X, Bcp_nn[0])
#     # X_1b = torch.norm(X_1, dim=3)
#     # X_2 = torch.einsum('tdr,drn -> trn', X_1b, Bcp_nn[1])
#     # X_3 = torch.einsum('trn -> tn', X_2)
#     # return X_3 + bias
    
#     X_1a = torch.einsum('twd,wrs -> tdr', X, Bcp_nn[0])
#     X_1b = torch.einsum('tdr,drs -> tr', X_1a, Bcp_nn[1])
#     # X_1c = torch.einsum('tr,dr -> dt', X_1a, Bcp_nn[1]) + bias
#     # return X_1c, X_1b
#     return X_1b


# def stepwise_spectral_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
#     """
#     Computes spectral regression model in a stepwise manner.
#     y_hat = inner(X, outer(softplus(Bcp))
#     where:
#         X.shape[1:] == Bcp.shape[:-1]
#         X.shape[0] == len(y_hat)
#         Bcp.shape[-1] == len(unique(y_true))
#         softplus is performed only on specified dimensions of Bcp.
#         inner prod is performed on dims [1:] of X and
#          dims [:-1] of Bcp.
#     RH 2021

#     Args:
#         X (torch.Tensor):
#             N-D array of data.
#         Bcp (list of torch.Tensor):
#             Beta Kruskal tensor (before softplus). (Bcp_n)
#             List of tensors of shape 
#              (n_features, rank).
#         weights (list of floats):
#             List of weights for each component.
#             len(weights) == rank == Bcp[0].shape[1]
#         non_negative (list of booleans):
#             List of booleans indicating whether each component
#              is non-negative.
#         bias (float):
#             Bias term. Scalar.
#         softplus_kwargs (dict):
#             Keyword arguments for torch.nn.functional.softplus.
    
#     Returns:
#         y_hat (torch.Tensor):
#             N-D array of predictions.
#     """

#     if Bcp[0].shape[1] == 0:
#         return torch.zeros(1).to(X.device)

#     # make non-negative
#     Bcp_nn = list(non_neg_fn(Bcp, non_negative, softplus_kwargs))
    
#     # X_1 = torch.einsum('twd,wrc -> tdrc', X, Bcp_nn[0])
#     # X_1b = torch.norm(X_1, dim=3)
#     # X_2 = torch.einsum('tdr,drn -> trn', X_1b, Bcp_nn[1])
#     # X_3 = torch.einsum('trn -> tn', X_2)
#     # return X_3 + bias
    
#     X_1a = torch.norm(torch.einsum('twd,wrc -> tdrc', X, Bcp_nn[0]), dim=3)
#     X_1b = torch.einsum('tdr,drs -> tr', X_1a, Bcp_nn[1])
#     X_1c = torch.einsum('tr,nrs -> tn', X_1b, Bcp_nn[2]) + bias
#     return X_1c

# @torch.jit.script
def forward_model(X:torch.Tensor, kernel:list, shifter, Bcp:list, weights, non_negative, bias, softplus_kwargs=None):
    """

    RH 2021
    """

    kernel_nn = list(non_neg_fn(kernel, [non_negative[0]]*2, softplus_kwargs))
    Bcp_nn = list(non_neg_fn(Bcp, non_negative[1:], softplus_kwargs))

    rank_n = kernel_nn[0].shape[1] if kernel_nn[0].ndim > 1 else kernel[0].ndim
    rank_s = kernel_nn[1].shape[1] if kernel_nn[1].ndim > 1 else kernel[1].ndim

    if rank_n > 0 and rank_s > 0:
        X_conv =  [conv(X, kernel_nn[0]).squeeze(-1)]
        # X_conv[0] = X_conv[0].squeeze(-1)[..., None]
        # X_conv += [complex_magnitude(conv(X, kernel_nn[1]))]
        # X_conv[1] = complex_magnitude(X_conv[1])
        X_conv += [torch.norm(torch.stack([ conv(X,
                                               shifter(kernel_nn[1],
                                                       shift_angle=shift,
                                                       deg_or_rad='deg',
                                                       dim=0)).squeeze(-1)
                                          for shift in [0,90] ],
                                        dim=-1),
                             dim=-1)]
        # print(X_conv[0].shape, X_conv[1].shape)
        
        for ii,x in enumerate(X_conv):
            if x.ndim < 3:
                X_conv[ii] = x[..., None]
        X_conv = torch.cat(X_conv, dim=-1)
        # print(X_conv.shape)

    elif rank_n > 0 and rank_s == 0:
        X_conv =  conv(X, kernel_nn[0]).squeeze(-1)

    elif rank_n == 0 and rank_s > 0:
        X_conv = torch.norm(torch.stack([ conv(X,
                                               shifter(kernel_nn[1],
                                                       shift_angle=shift,
                                                       deg_or_rad='deg',
                                                       dim=0)).squeeze(-1)
                                          for shift in [0,90] ],
                                        dim=-1),
                             dim=-1)

    pred =  stepwise_linear_model_multirank(X_conv, Bcp_nn, weights, bias)

    return pred

# def forward_model_lowMemory(X:torch.Tensor, kernel:list, shifter, Bcp:list, weights, non_negative, bias, softplus_kwargs=None):
#     """

#     RH 2021
#     """

#     kernel_nn = list(non_neg_fn(kernel, [non_negative[0]]*2, softplus_kwargs))
#     Bcp_nn = list(non_neg_fn(Bcp, non_negative[1:], softplus_kwargs))

#     rank_n = kernel_nn[0].shape[1] if kernel_nn[0].ndim > 1 else kernel[0].ndim
#     rank_s = kernel_nn[1].shape[1] if kernel_nn[1].ndim > 1 else kernel[1].ndim

#     for iter_rank in range(rank_n + rank_s):
#         if rank_n > 0 and rank_s > 0:
#             X_conv =  [conv(X, kernel_nn[0][:,iter_rank][:,None]).squeeze(-1)]
#             # X_conv[0] = X_conv[0].squeeze(-1)[..., None]
#             # X_conv += [complex_magnitude(conv(X, kernel_nn[1]))]
#             # X_conv[1] = complex_magnitude(X_conv[1])
#             X_conv += [torch.norm(torch.stack([ conv(X,
#                                                 shifter(kernel_nn[1][:,iter_rank][:,None],
#                                                         shift_angle=shift,
#                                                         deg_or_rad='deg',
#                                                         dim=0)).squeeze(-1)
#                                             for shift in [0,90] ],
#                                             dim=-1),
#                                 dim=-1)]
#             # print(X_conv[0].shape, X_conv[1].shape)
            
#             for ii,x in enumerate(X_conv):
#                 if x.ndim < 3:
#                     X_conv[ii] = x[..., None]
#             X_conv = torch.cat(X_conv, dim=-1)
#             # print(X_conv.shape)

#         elif rank_n > 0 and rank_s == 0:
#             X_conv =  conv(X, kernel_nn[0][:,iter_rank][:,None]).squeeze(-1)

#         elif rank_n == 0 and rank_s > 0:
#             X_conv = torch.norm(torch.stack([ conv(X,
#                                                 shifter(kernel_nn[1][:,iter_rank][:,None],
#                                                         shift_angle=shift,
#                                                         deg_or_rad='deg',
#                                                         dim=0)).squeeze(-1)
#                                             for shift in [0,90] ],
#                                             dim=-1),
#                                 dim=-1)

#         if iter_rank == 0:
#             pred = stepwise_linear_model_multirank(X_conv[...,None],
#                                                 [Bcp_nn[ii][:,iter_rank][:,None] for ii in range(len(Bcp_nn))],
#                                                 weights,
#                                                 bias)
#         else:
#             pred += stepwise_linear_model_multirank(X_conv[...,None],
#                                                 [Bcp_nn[ii][:,iter_rank][:,None] for ii in range(len(Bcp_nn))],
#                                                 weights,
#                                                 bias)

#     return pred

def spectral_penalty(y_pred, y_true=None, y_true_fft=None, n_fft=None, smoothing_kernel=None, passthrough=False, lam=0, plot_pref=False):
    """
    Computes the MSE of the spectrums of y_pred and y_true..

    Args:
        y_pred (torch.Tensor):
            N-D array of predictions.
        y_true (torch.Tensor):
            N-D array of ground truth.
        y_true_fft (torch.Tensor):
            N-D array of spectrums of ground truth.
        smoothing_kernel (torch.Tensor):
            1-D array of smoothing kernel.
    
    Returns:
        loss (torch.Tensor):
            Scalar loss.
    RH 2021
    """
    if passthrough:
        return torch.tensor(0.)

    if y_true is not None:
        y_true_fft = torch.abs(torch.fft.rfft(y_true, dim=0))

    assert y_true_fft is not None, "y_true_fft is None"

    # print(f'before {torch.norm(y_pred)}')
    # print(f'before {torch.sum(torch.isnan(y_pred))}')
    # print(f'before {torch.sum(torch.isinf(y_pred))}')
    # print(f'before {torch.sum(torch.isfinite(y_pred))}')
    # print(y_pred.shape)
    # plt.figure()
    # plt.plot(y_pred[:,0].cpu().detach().numpy(), label='pred')
    # plt.pause(0.01)
    # y_pred_fft = torch.abs(torch.fft.rfft(y_pred, dim=0, n=n_fft, norm='backward'))
    # y_pred_fft = torch.fft.rfft(y_pred, dim=0, n=n_fft, norm='backward')
    y_pred_fft = conv(torch.abs(torch.fft.rfft(y_pred, dim=0, n=n_fft)), smoothing_kernel)
    
    # return y_pred, y_pred_fft
    # print(torch.sum(y_pred_fft))
    # return torch.sum(y_pred_fft)*0
    # return 0

    # plt.figure()
    # plt.plot(y_pred_fft[:,0].cpu().detach().numpy(), label='pred')
    # plt.plot(y_true_fft[:,0].cpu().detach().numpy(), label='pred')
    # plt.pause(0.01)

    # print(y_pred_fft.shape)
    # print(f'after {torch.sum(torch.isnan(y_pred_fft))}')
    
    # if smoothing_kernel is not None:
    #     y_pred_fft = conv(y_pred_fft, smoothing_kernel)
    #     # y_pred_fft = conv(y_pred_fft, smoothing_kernel, padding='same')
    # else:
    #     y_pred_fft = y_pred_fft[(y_true_fft.shape[0]-y_pred_fft.shape[0])//2 : -(y_true_fft.shape[0]-y_pred_fft.shape[0])//2]

    # compute the MFSE (mean fractional squared error) of the spectrums
    # mse = torch.mean((y_pred_fft - y_true_fft)**2)
    mse = torch.mean(((y_pred_fft - y_true_fft) / (y_true_fft+1e-8))**2)
    # mse = torch.mean((torch.log(y_pred_fft**2) - torch.log(y_true_fft**2))**2)
    # mse = torch.mean(torch.log(torch.abs(y_pred_fft - y_true_fft)**2 + 1e-8))
    # mse = torch.mean((torch.abs(y_pred_fft - y_true_fft))**2)
    # mse = torch.mean((torch.abs(y_pred_fft) - torch.abs(y_true_fft))**2)
    # mse = torch.mean((torch.abs(y_pred_fft - y_true_fft)/ torch.abs(y_true_fft))**2)
    # mse = torch.mean((torch.abs(y_pred_fft - y_true_fft))**2)
    # mse = torch.mean(torch.abs((y_pred_fft - y_true_fft)**2))
    # mse = torch.mean(((torch.abs(y_pred_fft - y_true_fft)**2)/torch.abs(y_true_fft)))
    # mse = torch.mean(( (torch.abs(y_pred_fft) - torch.abs(y_true_fft))**2 ) / torch.abs(y_true_fft))
    # mse = torch.mean(torch.log((torch.abs(y_pred_fft - y_true_fft))**2))
    # mse = torch.mean((torch.abs(y_pred_fft - y_true_fft)))
    # mse = torch.mean(((y_pred_fft - y_true_fft)/y_true_fft)**2)
    # mse = torch.abs(torch.mean(((y_pred_fft - y_true_fft)/y_true_fft)**2))
    # print(mse.dtype)
    # mse = mse.float()
    
    if plot_pref:
        plt.figure()
        plt.plot(y_true_fft[:,0].cpu().detach().numpy(), label='pred')
        plt.plot(y_pred_fft[:,0].cpu().detach().numpy(), label='pred')
        plt.xlabel('smooth')
        plt.pause(0.01)

    # print(mse * lam)
    return mse * lam
    


# def L1_penalty(B_cp, lambda_L2):
#     """
#     Compute the L1 penalty.
#     RH 2021

#     Args:
#         B_cp (list of torch.Tensor):
#             Beta Kruskal tensor (before softplus)
    
#     Returns:
#         L2_penalty (torch.Tensor):
#             L2 penalty.
#     """

#     penalty=0
#     for ii, comp in enumerate(B_cp):
#         penalty+= torch.sum(torch.abs(comp)) * lambda_L2[ii]

#     return penalty
def L2_penalty(B_cp, lambda_L2):
    """
    Computes the L2 penalty.
    RH 2021

    Args:
        B_cp (list of torch.Tensor):
            Beta Kruskal tensor (before softplus)
    
    Returns:
        L2_penalty (torch.Tensor):
            L2 penalty.
    """

    penalty=0
    for ii, comp in enumerate(B_cp):
        penalty+= torch.sqrt(torch.sum(comp**2)) * lambda_L2[ii]
    return penalty

def diff_highOrder(traces, order):
    buffer = torch.zeros(traces.shape[1:], device=traces.device).unsqueeze(0)
    for ii in range(order):
        traces = torch.diff(traces, dim=0, prepend=buffer, append=buffer)
    return traces
def smoothness_penalty(Bcp_w, derivative_order=2, lambda_smooth=1):
    """
    Computes the smoothness penalty.
    RH 2021

    Args:
        Bcp_w (list of torch.Tensor):
            Beta Kruskal tensor (before softplus)
    
    Returns:
        smoothness_penalty (torch.Tensor):
            Smoothness penalty.
    """

    penalty=0
    for ii, comp in enumerate(Bcp_w):
        if comp.numel() > 0:
            penalty+= torch.mean(diff_highOrder(comp, order=derivative_order)**2) * lambda_smooth
    return penalty


class phase_shifter():
    def __init__(self, signal_len, discard_imaginary_component=True, device='cpu', dtype=torch.float32, pin_memory=False):
        """
        Initializes the shift_signal_angle_obj class.
        This is the object version. It can be faster than
         the functional version if needing to call it 
         multiple times.
        See __call__ for more details.
        RH 2021

        Args:
            signal_len (int):
                The shape of the signal to be shifted.
                The first (0_th) dimension must be the shift dimension.
            discard_imaginary_component (bool):
                Whether to discard the imaginary component of the signal
            device (str):
                The device to put self.angle_mask on
            dtype (torch.dtype):
                The dtype to use for self.angle_mask
            pin_memory (bool):
                Whether to pin self.angle_mask to memory
        """

        self.signal_len = signal_len
        signal_len = torch.as_tensor(signal_len)
        half_len_minus = int(torch.ceil(signal_len/2))
        half_len_plus = int(torch.floor(signal_len/2))
        self.angle_mask = torch.cat([
            -torch.ones(half_len_minus, dtype=dtype, device=device, pin_memory=pin_memory),
             torch.ones(half_len_plus,  dtype=dtype, device=device, pin_memory=pin_memory)
             ])
        self.discard_imaginary_component = discard_imaginary_component

    def __call__(self, signal, shift_angle=90, deg_or_rad='deg', dim=0):
        """
        Shifts the frequency angles of a signal by a given amount.
        A signal containing multiple frequecies will see each 
         frequency shifted independently by the shift_angle.
        RH 2021

        Args:
            signal (torch.Tensor):
                The signal to be shifted
            shift_angle (float):
                The amount to shift the angle by
            deg_or_rad (str):
                Whether the shift_angle is in degrees or radians
            dim (int):
                The axis to shift along
            
        Returns:
            output (torch.Tensor):
                The shifted signal
        """
        
        if shift_angle == 0:
            return signal
        
        shift_angle = np.deg2rad(shift_angle) if deg_or_rad == 'deg' else shift_angle

        signal_fft = torch.fft.fft(signal, dim=dim) # convert to spectral domain
        mag, ang = torch.abs(signal_fft), torch.angle(signal_fft) # extract magnitude and angle
        ang_shifted = ang + (self.angle_mask.reshape([len(self.angle_mask)] + [1]*(len(signal.shape)-1))) * shift_angle # shift the angle. The bit in the middle is for matching the shape of the signal
        signal_fft_shifted = mag * torch.exp(1j*ang_shifted) # remix magnitude and angle
        signal_shifted = torch.fft.ifft(signal_fft_shifted, dim=dim) # convert back to signal domain
        if self.discard_imaginary_component:
            signal_shifted = torch.real(signal_shifted) # discard imaginary component
        return signal_shifted


####################################
########### Main class #############
####################################

class CP_linear_regression():
    def __init__(self, 
                    X_shape, 
                    y_shape,
                    dtype=torch.float32,
                    rank_normal=1, 
                    temporal_window=5,
                    rank_spectral=1,
                    non_negative=False, 
                    weights=None, 
                    Bcp_init=None, 
                    Bcp_init_scale=1, 
                    # n_complex_dim=0, 
                    bias_init=0, 
                    device='cpu', 
                    do_spectralPenalty=False,
                    spectrum_smoothing_factor=100,
                    softplus_kwargs=None):
        """
        Multinomial logistic CP tensor regression class.
        Bias is not considered in this model because there
         are multiple ways to handle bias with tensor
         regression. User must add a bias entry to X
         manually if they wish. Mean subtracting X along
         one dimension is recommended.
        RH 2021 / JZ 2021

        Args:
            X (np.ndarray or torch.Tensor):
                Input data. First dimension must match len(y).
                For batched fitting, use the shape of a single
                 sample. The custom dataloader class has a 
                 parameter for sample shape (dataloader.sample_shape).
            y (np.ndarray or torch.Tensor):
                Class labels. Length must match first dimension of X.
            rank (int):
                Rank of the CP (Kruskal) tensor used to compute the
                 beta tensor that is multiplied by X.
            temporal_window (int):
                Number of timesteps for the temporal window of the
                 kernel with which to convolve the time dimension of X.
            non_negative (False or True or list):
                If False, the CP tensor is allowed to be negative.
                If True, the CP tensor is forced to be non-negative.
                If a list, then the list should be length X.ndim,
                 and each element should be a boolean indicating 
                 whether that dimension of the beta tensor should be
                 forced to be non-negative.
                 Entries correspond to the dimension index+1 of X, 
                 and last dim is the classification (one hot) dim of
                 the beta tensor.
            weights (np.ndarray or torch.Tensor):
                Weights for each component. Should be a
                 vector of length(rank).
            Bcp_init (list):
                Optional. List of initial Bcp matrices.
                If None, then the initial Bcp matrices are set as 
                 random number from a uniform distribution between
                 either 0 to 1 if that dimension is non_negative or
                 -1 to 1 if dimension is not non_negative.
            Bcp_init_scale (float):
                Optional. Scale of uniform distribution used to
                 initialize each component of Bcp.
            bias_init (float):
                Optional. Initial bias. Scalar.
            device (str):
                Device to run the model on.
        """        

        # self.X = torch.tensor(X, dtype=torch.float32).to(device)
        # self.y = torch.tensor(y, dtype=torch.float32).to(device)
        
        self.dtype = dtype
        
        if weights is None:
            self.weights = torch.ones((rank_normal+rank_spectral), dtype=self.dtype, requires_grad=False, device=device)
            # self.weights = torch.ones((rank), requires_grad=True, device=device)
        else:
            self.weights = torch.tensor(weights, dtype=self.dtype, requires_grad=False, device=device)

        if softplus_kwargs is None:
            self.softplus_kwargs = {'beta': 50, 'threshold': 1}
        else:
            self.softplus_kwargs = softplus_kwargs

        self.X_shape = X_shape
        self.y_shape = y_shape

        self.rank_normal = rank_normal
        self.temporal_window = temporal_window
        self.idx_conv = self.get_idxConv(X_shape[0]).to(device)
        self.rank_spectral = rank_spectral
        self.rank = rank_normal + rank_spectral
        self.device = device

        if non_negative == True:
            self.non_negative = [True]*(len(X_shape) + len(y_shape[1:]))
        elif non_negative == False:
            self.non_negative = [False]*(len(X_shape) + len(y_shape[1:]))
        else:
            self.non_negative = non_negative        
        
        # if len(y_shape) == 1:
        #     y_shape = list(y_shape) + [1]
        # self.bias = torch.tensor([bias_init], dtype=torch.float32, requires_grad=True, device=device) 
        self.bias = torch.zeros(y_shape[1:], dtype=self.dtype, requires_grad=True, device=device) 
        self.y_shape = y_shape

        # B_dims = list(X_shape[1:])
        B_dims = list(X_shape[1:]) + list(y_shape[1:])
        # complex_dims = list([n_complex_dim+1] + [1]*(len(B_dims)-1))
        # print(complex_dims)

        Bw_dims = [temporal_window]

        self.shifter = phase_shifter(signal_len=temporal_window,
                                     discard_imaginary_component=True,
                                     device=self.device,
                                     dtype=self.dtype,
                                     pin_memory=False)

        if Bcp_init is None:
            self.Bcp_w = make_BcpInit(Bw_dims, self.rank_normal,   [self.non_negative[0]], complex_dims=None, scale=Bcp_init_scale, device=self.device, dtype=self.dtype) + \
                         make_BcpInit(Bw_dims, self.rank_spectral, [self.non_negative[0]], complex_dims=None, scale=Bcp_init_scale, device=self.device, dtype=self.dtype)
            
            self.Bcp_n = make_BcpInit(B_dims, self.rank_normal+self.rank_spectral, self.non_negative[1:], complex_dims=[1]*len(B_dims), scale=Bcp_init_scale, device=self.device, dtype=self.dtype) # 'normal Beta_cp Kruskal tensor'

            for ii in range(len(B_dims)):
                self.Bcp_n[ii].requires_grad = True
            for ii in range(len(self.Bcp_w)):
                self.Bcp_w[ii].requires_grad = True

        else:
            self.Bcp_n = Bcp_init[1]
            self.Bcp_w = Bcp_init[0]

        
        ####### INIT FOR SPECTRUM PENALTY #######
        self.do_spectralPenalty = do_spectralPenalty
        self.spectral_smoothing_kernel = torch.tensor(
            gaussian(
                np.arange(-(spectrum_smoothing_factor//2),
                spectrum_smoothing_factor//2+1),
                0,
                spectrum_smoothing_factor/7)[0],
             dtype=self.dtype, requires_grad=False, device=self.device)

        self.loss_running = []

    def fit(self,
            X,
            y,
            lambda_L2=0.01, 
            lambda_spectralPenalty=0.01,
            lambda_smooth=0.01,
            smooth_diff_order=2,
            max_iter=1000, 
            tol=1e-5, 
            patience=10,
            verbose=False,
            running_loss_logging_interval=10, 
            LBFGS_kwargs=None):
        """
        Fit a beta tensor (self.Bcp) to the data using the
         LBFGS optimizer.
        Note that self.Bcp is not the final Kruskal tensor, 
         non_neg_fn(self.Bcp, non_negative) 
         is the final Kruskal tensor.
        Use self.return_Bcp_final() to get the final tensor.
        Note that logging the loss (self.loss_running)
         requires running the model extra times. If data is
         large, then set running_loss_logging_interval to a
         large number.
        RH 2021

        Args:
            lambda_L2 (float):
                L2 regularization parameter.
            max_iter (int):
                Maximum number of iterations.
            tol (float):
                Tolerance for the stopping criterion.
            patience (int):
                Number of iterations with no improvement to wait
                 before early stopping.
            verbose (0, 1, or 2):
                If 0, then no output.
                If 1, then only output whether the model has
                 converged or not.
                If 2, then output the loss at each iteration.
            running_loss_logging_interval (int):
                Number of iterations between logging of running loss.
            LBFGS_kwargs (dict):
                Keyword arguments for LBFGS optimizer.
        
        Returns:
            convergence_reached (bool):
                True if convergence was reached.
        """

        if LBFGS_kwargs is None:
            {
                'lr' : 1, 
                'max_iter' : 20, 
                'max_eval' : 20, 
                'tolerance_grad' : 1e-07, 
                'tolerance_change' : 1e-09, 
                'history_size' : 100, 
                'line_search_fn' : "strong_wolfe"
            }

        if isinstance(lambda_L2, int) or isinstance(lambda_L2, float):
            lambda_L2 = torch.tensor([lambda_L2]*(1 + len(self.Bcp_n)), device=X.device, dtype=self.dtype)
        elif isinstance(lambda_L2, list):
            if len(lambda_L2) == 1:
                lambda_L2 = torch.tensor([lambda_L2]*(1 + len(self.Bcp_n)), device=X.device, dtype=self.dtype)
        
        if self.do_spectralPenalty:
            self.y_spectrum = conv(torch.abs(torch.fft.rfft(y[self.idx_conv], n=self.y_shape[0], dim=0)), self.spectral_smoothing_kernel) # note whether you do padding='same' or not and apply the same to spectral penalty
        else:
            self.y_spectrum = None

        tl.set_backend('pytorch')

        self.optimizer = torch.optim.LBFGS(self.Bcp_n + self.Bcp_w + [self.bias], **LBFGS_kwargs)
        loss_fn = torch.nn.MSELoss()

        def loss_all(y_hat, y, lambda_L2, lambda_spectralPenalty):
            loss_rec = loss_fn(y_hat, y[self.idx_conv])
            loss_L2_w = L2_penalty(self.Bcp_w, [lambda_L2[0]]*2)
            loss_L2_n = L2_penalty(self.Bcp_n, lambda_L2[1:])
            loss_spectral = spectral_penalty(   y_pred=y_hat, 
                                        y_true_fft=self.y_spectrum, 
                                        n_fft=self.y_shape[0],
                                        smoothing_kernel=self.spectral_smoothing_kernel, 
                                        passthrough=1-self.do_spectralPenalty, 
                                        lam=lambda_spectralPenalty,
                                        plot_pref=False)
            loss_smoothness = smoothness_penalty(self.Bcp_w, derivative_order=smooth_diff_order, lambda_smooth=lambda_smooth)
            loss_all = loss_rec + loss_L2_w + loss_L2_n + loss_spectral + loss_smoothness
            return loss_all, loss_rec.item(), loss_L2_w.item(), loss_L2_n.item(), loss_spectral.item(), loss_smoothness.item()

        def closure():
            self.optimizer.zero_grad()
            y_hat = forward_model(X, self.Bcp_w, self.shifter, self.Bcp_n, self.weights, self.non_negative, self.bias, self.softplus_kwargs)
            loss, loss_rec, loss_L2_w, loss_L2_n, loss_spectral, loss_smoothness =  loss_all(y_hat, y, lambda_L2, lambda_spectralPenalty)
            loss.backward()
            return loss
        
        def print_info(iter, loss_all, loss_rec, loss_L2_w, loss_L2_n, loss_spectral, loss_smoothness, variance_ratio, precis=5):
            print(f'Iter: {iter}, \
loss: {loss_all:.{precis}}, \
loss_rec: {loss_rec:.{precis}}, \
loss_L2_w: {loss_L2_w:.{precis}}, \
loss_L2_n: {loss_L2_n:.{precis}}, \
loss_spectral: {loss_spectral:.{precis}}, \
loss_smoothness: {loss_smoothness:.{precis}}, \
var_ratio (y_hat/y_true): {variance_ratio:.{precis}}')
              
        if verbose==3:
            self.fig, self.axs = plt.subplots(2 + len(self.Bcp_n) + self.rank_spectral + self.rank_normal, figsize=(7,20))

        convergence_reached = False
        for ii in range(max_iter):
            if ii%running_loss_logging_interval == 0:

                y_hat = forward_model(X, self.Bcp_w, self.shifter, self.Bcp_n, self.weights, self.non_negative, self.bias, self.softplus_kwargs)
                
                loss, loss_rec, loss_L2_w, loss_L2_n, loss_spectral, loss_smoothness =  loss_all(y_hat, y, lambda_L2, lambda_spectralPenalty)
                self.loss_running.append(loss.item())
                if verbose==2:
                    print_info(ii, self.loss_running[-1], loss_rec, loss_L2_w, loss_L2_n, loss_spectral, loss_smoothness, variance_ratio=torch.var(y_hat.detach()).item() / torch.var(y).item())            
                elif verbose==3:
                    self.update_plot_outputs(self.fig, self.axs)
                    # plt.pause(0.01)
                    print_info(ii, self.loss_running[-1], loss_rec, loss_L2_w, loss_L2_n, loss_spectral, loss_smoothness, variance_ratio=torch.var(y_hat.detach()).item() / torch.var(y).item())            
            if len(self.loss_running) > patience:
                if np.sum(np.abs(np.diff(self.loss_running[-patience+1:]))) < tol:
                    convergence_reached = True
                    break
            elif np.isnan(self.loss_running[-1]):
                convergence_reached = False
                print('Loss is NaN. Stopping.')
                break

            self.optimizer.step(closure)
        if (verbose==True) or (verbose>=1):
            if convergence_reached:
                print('Convergence reached')
            else:
                print('Reached maximum number of iterations without convergence')
        return convergence_reached


    def fit_Adam(self,X,y,
            lambda_L2=0.01, 
            lambda_spectralPenalty=0.01,
            lambda_smooth=0.01,
            smooth_diff_order=2,
            max_iter=1000, 
            tol=1e-5, 
            patience=10,            
            verbose=False,
            plotting_interval=100,
            Adam_kwargs=None):
        """
        Fit a beta tensor (self.Bcp) to the data using the
         Adam optimizer.
        Note that self.Bcp is not the final Kruskal tensor, 
         non_neg_fn(self.Bcp, non_negative) 
         is the final Kruskal tensor.
        Use self.return_Bcp_final() to get the final tensor.
        Note that logging the loss (self.loss_running)
         requires running the model extra times. If data is
         large, then set running_loss_logging_interval to a
         large number.
        RH 2021

        Args:
            lambda_L2 (float):
                L2 regularization parameter.
            max_iter (int):
                Maximum number of iterations.
            tol (float):
                Tolerance for the stopping criterion.
            patience (int):
                Number of iterations with no improvement to wait
                 before early stopping.
            verbose (0, 1, or 2):
                If 0, then no output.
                If 1, then only output whether the model has
                 converged or not.
            Adam_kwargs (dict):
                Keyword arguments for Adam optimizer.
        
        Returns:
            convergence_reached (bool):
                True if convergence was reached.
        """

        if Adam_kwargs is None:
            {
                'lr' : 1, 
                'betas' : (0.9, 0.999), 
                'eps' : 1e-08, 
                'weight_decay' : 0, 
                'amsgrad' : False
            }

        tl.set_backend('pytorch')

        if isinstance(lambda_L2, int) or isinstance(lambda_L2, float):
            lambda_L2 = torch.tensor([lambda_L2]*(1 + len(self.Bcp_n)), device=X.device, dtype=self.dtype)
        elif isinstance(lambda_L2, list):
            if len(lambda_L2) == 1:
                lambda_L2 = torch.tensor([lambda_L2]*(1 + len(self.Bcp_n)), device=X.device, dtype=self.dtype)
        
        if self.do_spectralPenalty:
            self.y_spectrum = conv(torch.abs(torch.fft.rfft(y[self.idx_conv], n=self.y_shape[0], dim=0)), self.spectral_smoothing_kernel) # note whether you do padding='same' or not and apply the same to spectral penalty
        else:
            self.y_spectrum = None

        self.optimizer = torch.optim.Adam(self.Bcp_n + self.Bcp_w + [self.bias], **Adam_kwargs)
        # optimizer = torch.optim.Adam(self.Bcp + [self.weights], **Adam_kwargs)
        loss_fn = torch.nn.MSELoss()

        def loss_all(y_hat, y, lambda_L2, lambda_spectralPenalty):
            loss_rec = loss_fn(y_hat, y[self.idx_conv])
            loss_L2_w = L2_penalty(self.Bcp_w, [lambda_L2[0]]*2)
            loss_L2_n = L2_penalty(self.Bcp_n, lambda_L2[1:])
            loss_spectral = spectral_penalty(   y_pred=y_hat, 
                                        y_true_fft=self.y_spectrum, 
                                        n_fft=self.y_shape[0],
                                        smoothing_kernel=self.spectral_smoothing_kernel, 
                                        passthrough=1-self.do_spectralPenalty, 
                                        lam=lambda_spectralPenalty,
                                        plot_pref=False)
            loss_smoothness = smoothness_penalty(self.Bcp_w, derivative_order=smooth_diff_order, lambda_smooth=lambda_smooth)
            loss_all = loss_rec + loss_L2_w + loss_L2_n + loss_spectral + loss_smoothness
            return loss_all, loss_rec.item(), loss_L2_w.item(), loss_L2_n.item(), loss_spectral.item(), loss_smoothness.item()

        if verbose==3:
            self.fig, self.axs = plt.subplots(2 + len(self.Bcp_n) + self.rank_spectral + self.rank_normal, figsize=(7,20))

        def print_info(iter, loss_all, loss_rec, loss_L2_w, loss_L2_n, loss_spectral, loss_smoothness, variance_ratio, precis=5):
            print(f'Iter: {iter}, \
loss: {loss_all:.{precis}}, \
loss_rec: {loss_rec:.{precis}}, \
loss_L2_w: {loss_L2_w:.{precis}}, \
loss_L2_n: {loss_L2_n:.{precis}}, \
loss_spectral: {loss_spectral:.{precis}}, \
loss_smoothness: {loss_smoothness:.{precis}}, \
var_ratio (y_hat/y_true): {variance_ratio:.{precis}}')

        convergence_reached = False
        for ii in range(max_iter):
            self.optimizer.zero_grad()

            y_hat = forward_model(X, self.Bcp_w, self.shifter, self.Bcp_n, self.weights, self.non_negative, self.bias, self.softplus_kwargs)
            
            loss, loss_rec, loss_L2_w, loss_L2_n, loss_spectral, loss_smoothness =  loss_all(y_hat, y, lambda_L2, lambda_spectralPenalty)
            loss.backward()
            self.optimizer.step()
            self.loss_running.append(loss.item())
            if verbose==2:
                print_info(ii, self.loss_running[-1], loss_rec, loss_L2_w, loss_L2_n, loss_spectral, loss_smoothness, variance_ratio=torch.var(y_hat.detach()).item() / torch.var(y).item())            
            elif verbose==3:
                if ii%plotting_interval==0:
                    self.update_plot_outputs(self.fig, self.axs)
                # plt.pause(0.01)
                print_info(ii, self.loss_running[-1], loss_rec, loss_L2_w, loss_L2_n, loss_spectral, loss_smoothness, variance_ratio=torch.var(y_hat.detach()).item() / torch.var(y).item())            
            if len(self.loss_running) > patience:
                if np.sum(np.abs(np.diff(self.loss_running[-patience+1:]))) < tol:
                    convergence_reached = True
                    break
            elif np.isnan(self.loss_running[-1]):
                convergence_reached = False
                print('Loss is NaN. Stopping.')
                break

        if (verbose==True) or (verbose>=1):
            if convergence_reached:
                print('Convergence reached')
            else:
                print('Reached maximum number of iterations without convergence')
        return convergence_reached

    # def fit_batch_Adam(self,
    #         dataloader,
    #         lambda_L2=0.01, 
    #         max_iter=1000, 
    #         tol=1e-5, 
    #         patience=10,
    #         n_iter_inner=10,
    #         verbose=False,
    #         Adam_kwargs=None,
    #         device=None):
    #     """
    #     JZ 2021 / RH 2021
    #     """
            
    #     if Adam_kwargs is None:
    #         {
    #             'lr' : 1, 
    #             'betas' : (0.9, 0.999), 
    #             'eps' : 1e-08, 
    #             'weight_decay' : 0, 
    #             'amsgrad' : False
    #         }

    #     tl.set_backend('pytorch')
        
    #     if device is None:
    #         device = self.device

    #     optimizer = torch.optim.Adam(self.Bcp + [self.bias], **Adam_kwargs)
    #     loss_fn = torch.nn.MSELoss()
        
    #     convergence_reached = False
    #     for ii in range(max_iter):
    #         for batch_idx, data in enumerate(dataloader):
    #             # print(data)
    #             X, y = data
    #             # X = torch.tensor(X, dtype=torch.float32).to(device)
    #             # y = torch.tensor(y, dtype=torch.float32).to(device)
    #             X = X.to(device)
    #             y = y.to(device)   
    #             for iter_inner in range(n_iter_inner):             
    #                 optimizer.zero_grad()
    #                 y_hat = lin_model(X, self.Bcp, self.weights, self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)
    #                 loss = loss_fn(y_hat, y) + lambda_L2 * L2_penalty(self.Bcp)
    #                 loss.backward()
    #                 optimizer.step()
    #                 self.loss_running.append(loss.item())
    #                 if verbose==2:
    #                     print(f'Epoch: {ii}, Inner iteration: {iter_inner}, Loss: {self.loss_running[-1]}  ;  Variance ratio (y_hat / y_true): {torch.var(y_hat.detach()).item() / torch.var(y).item()}' )
    #             if ii > patience:
    #                 if np.sum(np.abs(np.diff(self.loss_running[ii-patience:]))) < tol:
    #                     convergence_reached = True
    #                     break
    #     if (verbose==True) or (verbose>=1):
    #         if convergence_reached:
    #             print('Convergence reached')
    #         else:
    #             print('Reached maximum number of iterations without convergence')
    #     return convergence_reached

    # def fit_batch_LBFGS(self,
    #         dataloader,
    #         lambda_L2=0.01, 
    #         max_iter=1000, 
    #         tol=1e-5, 
    #         patience=10,
    #         n_iter_inner=10,
    #         verbose=False,
    #         LBFGS_kwargs=None,
    #         device=None):
    #     """
    #     JZ 2021 / RH 2021
    #     """

    #     if LBFGS_kwargs is None:
    #         {
    #             'lr' : 1, 
    #             'max_iter' : 20, 
    #             'max_eval' : 20, 
    #             'tolerance_grad' : 1e-07, 
    #             'tolerance_change' : 1e-09, 
    #             'history_size' : 100, 
    #             'line_search_fn' : "strong_wolfe"
    #         }

    #     tl.set_backend('pytorch')
        
    #     if device is None:
    #         device = self.device

    #     tl.set_backend('pytorch')

    #     optimizer = torch.optim.LBFGS(self.Bcp + [self.bias], **LBFGS_kwargs)
    #     def closure():
    #         optimizer.zero_grad()
    #         y_hat = lin_model(X, self.Bcp, self.weights, self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)
    #         loss = loss_fn(y_hat, y) + lambda_L2 * L2_penalty(self.Bcp)
    #         loss.backward()
    #         return loss
    #     loss_fn = torch.nn.MSELoss()
        
    #     convergence_reached = False
    #     for ii in range(max_iter):
    #         for batch_idx, data in enumerate(dataloader):
    #             # print(data)
    #             gc.collect()
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             torch.cuda.empty_cache()
    #             X, y = data[0].to(device), data[1].to(device)
    #             # X = torch.tensor(X, dtype=torch.float32).to(device)
    #             # y = torch.tensor(y, dtype=torch.float32).to(device)
    #             gc.collect()
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             torch.cuda.empty_cache()
    #             # X = X.to(device)
    #             # y = y.to(device) 
    #             for iter_inner in range(n_iter_inner):               
    #                 y_hat = lin_model(X, self.Bcp, self.weights, self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)
    #                 loss = loss_fn(y_hat, y) + lambda_L2 * L2_penalty(self.Bcp)
    #                 optimizer.step(closure)

    #                 self.loss_running.append(loss.item())
    #                 if verbose==2:
    #                     print(f'Epoch: {ii}, Inner iteration: {iter_inner}, Loss: {self.loss_running[-1]}  ;  Variance ratio (y_hat / y_true): {torch.var(y_hat.detach()).item() / torch.var(y).item()}' )
    #             if ii > patience:
    #                 if np.sum(np.abs(np.diff(self.loss_running[ii-patience:]))) < tol:
    #                     convergence_reached = True
    #                     break
                
    #             # del X, y, y_hat
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             torch.cuda.empty_cache()
    #             gc.collect()

    #     if (verbose==True) or (verbose>=1):
    #         if convergence_reached:
    #             print('Convergence reached')
    #         else:
    #             print('Reached maximum number of iterations without convergence')
    #     return convergence_reached


    ####################################
    ############ POST-HOC ##############
    ####################################


    def predict(self, X, Bcp=None, device=None, plot_pref=False):
        """
        Predict class labels for X given a Bcp (beta Kruskal tensor).
        Uses 'model' function in this module.
        RH 2021

        Args:
            X (np.ndarray or torch.Tensor):
                Input data. First dimension must match len(y).
                If None, then use the data that was passed to the
                 constructor.
            Bcp (list of np.ndarray or torch.Tensor):
                List of Bcp lists of matrices.
                Bcp[0] should be Bcp_n, Bcp[1] should be Bcp_w.
                Note that Bcp_w[0] should be shape (X.shape[1],
                 rank, complex_dimensionality), where 
                 complex_dimensionality is usually 2 (real and complex).
                If None, then use the Bcp that was passed to the
                 constructor.
            y_true (np.ndarray or torch.Tensor):
                True class labels. Only necessary if plot_pref is
                 True and inputting data from a different X than
                 what the model was trained on.
            device (str):
                Device to run the model on.
            plot_pref (bool):
                If True, then make plots
        
        Returns:
            logits (np.ndarray):
                Raw output of the model.
        """
        if device is None:
            device = self.device
        
        # if X is None:
        #     X = self.X
        if isinstance(X, torch.Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32, requires_grad=False).to(device)
        elif X.device != device:
            X = X.to(device)

        # if y_true is None:
        #     y_true = self.y.detach().cpu().numpy()

                        
        if Bcp is None:
            Bcp_n = self.Bcp_n
            Bcp_w = self.Bcp_w
        elif isinstance(Bcp[0][0], torch.Tensor) == False:
            Bcp_n = [0]*len(Bcp[0])
            for ii in range(len(Bcp[0])):
                Bcp_n[ii] = torch.tensor(Bcp[0][ii], dtype=torch.float32, requires_grad=False).to(device)
            Bcp_w = [0]*len(Bcp[1])
            for ii in range(len(Bcp[1])):
                Bcp_w[ii] = torch.tensor(Bcp[1][ii], dtype=torch.float32, requires_grad=False).to(device)
        elif Bcp[0][0].device != device:
            Bcp_n = [0]*len(Bcp[0])
            for ii in range(len(Bcp[0])):
                Bcp_n[ii] = Bcp[0][ii].to(device)
            Bcp_w = [0]*len(Bcp[1])
            for ii in range(len(Bcp[1])):
                Bcp_w[ii] = Bcp[1][ii].to(device)
        
        y_hat = forward_model(X, self.Bcp_w, self.shifter, self.Bcp_n, self.weights, self.non_negative, self.bias, self.softplus_kwargs)

        return y_hat.cpu().detach()

    
    # def predict_latents(self, X, Bcp=None, device=None, plot_pref=False):
    #     """
    #     Predict latent variables for X given a Bcp (beta Kruskal tensor).
    #     Uses 'model' function in this module.
    #     RH 2021

    #     Args:
    #         X (np.ndarray or torch.Tensor):
    #             Input data. First dimension must match len(y).
    #             If None, then use the data that was passed to the
    #              constructor.
    #         Bcp (list of np.ndarray or torch.Tensor):
    #             List of Bcp lists of matrices.
    #             Bcp[0] should be Bcp_n, Bcp[1] should be Bcp_c.
    #             Note that Bcp_c[0] should be shape (X.shape[1],
    #              rank, complex_dimensionality), where 
    #              complex_dimensionality is usually 2 (real and complex).
    #             If None, then use the Bcp that was passed to the
    #              constructor.
    #         y_true (np.ndarray or torch.Tensor):
    #             True class labels. Only necessary if plot_pref is
    #              True and inputting data from a different X than
    #              what the model was trained on.
    #         device (str):
    #             Device to run the model on.
    #         plot_pref (bool):
    #             If True, then make plots
        
    #     Returns:
    #         logits (np.ndarray):
    #             Raw output of the model.
    #     """
    #     if device is None:
    #         device = self.device
        
    #     # if X is None:
    #     #     X = self.X
    #     if isinstance(X, torch.Tensor) == False:
    #         X = torch.tensor(X, dtype=torch.float32, requires_grad=False).to(device)
    #     elif X.device != device:
    #         X = X.to(device)

    #     # if y_true is None:
    #     #     y_true = self.y.detach().cpu().numpy()

                        
    #     if Bcp is None:
    #         Bcp_n = self.Bcp_n
    #         Bcp_c = self.Bcp_c
    #         Bcp_w = self.Bcp_w
    #     elif isinstance(Bcp[0][0], torch.Tensor) == False:
    #         Bcp_n = [0]*len(Bcp[0])
    #         for ii in range(len(Bcp[0])):
    #             Bcp_n[ii] = torch.tensor(Bcp[0][ii], dtype=torch.float32, requires_grad=False).to(device)
    #         Bcp_c = [0]*len(Bcp[1])
    #         for ii in range(len(Bcp[1])):
    #             Bcp_c[ii] = torch.tensor(Bcp[1][ii], dtype=torch.float32, requires_grad=False).to(device)

    #         # Bcp_w = [0]*len(Bcp[1])
    #         # for ii in range(len(Bcp[1])):
    #         #     Bcp_w[ii] = torch.tensor(Bcp[1][ii], dtype=torch.float32, requires_grad=False).to(device)

    #     elif Bcp[0][0].device != device:
    #         Bcp_n = [0]*len(Bcp[0])
    #         for ii in range(len(Bcp[0])):
    #             Bcp_n[ii] = Bcp[0][ii].to(device)
    #         Bcp_c = [0]*len(Bcp[1])
    #         for ii in range(len(Bcp[1])):
    #             Bcp_c[ii] = Bcp[1][ii].to(device)
            
    #         # Bcp_w = [0]*len(Bcp[1])
    #         # for ii in range(len(Bcp[1])):
    #         #     Bcp_w[ii] = Bcp[1][ii].to(device)


    #     y_hat = stepwise_latents_model(X, Bcp_n, self.weights[:self.rank_normal], self.non_negative, bias=None, softplus_kwargs=self.softplus_kwargs)
    #             # spectral_model(X, Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)
    #     # y_hat = spectral_model(X, Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)

        # return y_hat.cpu().detach().numpy()


    def get_idxConv(self, input_length):
        return torch.arange(self.temporal_window//2, input_length-self.temporal_window//2)



    
    def return_Bcp_final(self):
        """
        Return the final Kruskal tensor as a numpy array.
        Simply passes self.Bcp to non_neg_fn.
        RH 2021

        Returns:
            Bcp_nonNeg (np.ndarray):
                Final Kruskal tensor. Ready to multiply by the
                 data to predict class labels (y_hat).
        """
        Bcp_n = list(non_neg_fn(self.Bcp_n, self.non_negative[1:], softplus_kwargs=self.softplus_kwargs))
        Bcp_w = list(non_neg_fn(self.Bcp_w, [self.non_negative[0]]*2, softplus_kwargs=self.softplus_kwargs))
        Bcp_n_nonNeg = [Bcp_n[ii].detach().cpu().numpy() for ii in range(len(Bcp_n))]
        Bcp_w_nonNeg = [Bcp_w[ii].detach().cpu().numpy() for ii in range(len(Bcp_w))]
        return Bcp_n_nonNeg , Bcp_w_nonNeg
    
    def detach_Bcp(self):
        """
        Detach the Bcp Kruskal tensor list.
        RH 2021

        Returns:
            Bcp_detached (list of np.ndarray):
                Detached Bcp tensors.
        """
        Bcp_n_detached = [Bcp.detach().cpu().numpy() for Bcp in self.Bcp_n]
        Bcp_w_detached = [Bcp.detach().cpu().numpy() for Bcp in self.Bcp_w]
        return Bcp_n_detached, Bcp_w_detached

    def get_params(self):
        """
        Get the parameters of the model.
        RH 2021
        """
        return {
                'X_shape': self.X_shape,
                'y_shape': self.y_shape,
                'dtype': self.dtype,
                'device': self.device,
                'weights': self.weights.detach().cpu().numpy(),
                'Bcp_n': [Bcp.detach().cpu().numpy() for Bcp in self.Bcp_n],
                'Bcp_w': [Bcp.detach().cpu().numpy() for Bcp in self.Bcp_w],
                'bias': self.bias.detach().cpu().numpy(),
                'non_negative': self.non_negative,
                'softplus_kwargs': self.softplus_kwargs,
                'rank': self.rank,
                'rank_normal': self.rank_normal,
                'rank_spectral': self.rank_spectral,
                'idxConv': self.idx_conv.detach().cpu().numpy(),
                'spectral_smoothing_kernel': self.spectral_smoothing_kernel.detach().cpu().numpy(),
                'temporal_window': self.temporal_window,
                'do_spectralPenalty': self.do_spectralPenalty,
                'loss_running': self.loss_running,
                'shifter': self.shifter,
                }

    def set_params(self, params):
        """
        Set the parameters of the model.
        RH 2021
        """
        self.X_shape = params['X_shape']
        self.y_shape = params['y_shape']
        self.dtype = params['dtype']
        self.device = params['device']
        self.weights = torch.as_tensor(params['weights'], dtype=self.dtype).to(self.device)
        self.Bcp_n = [torch.as_tensor(Bcp, dtype=self.dtype).to(self.device) for Bcp in params['Bcp_n']]
        self.Bcp_w = [torch.as_tensor(Bcp, dtype=self.dtype).to(self.device) for Bcp in params['Bcp_w']]
        self.bias = torch.as_tensor(params['bias'], dtype=self.dtype).to(self.device)
        self.non_negative = params['non_negative']
        self.softplus_kwargs = params['softplus_kwargs']
        self.rank = params['rank']
        self.rank_normal = params['rank_normal']
        self.rank_spectral = params['rank_spectral']
        self.idx_conv = torch.as_tensor(params['idxConv'], dtype=self.dtype).to(self.device)
        self.spectral_smoothing_kernel = torch.as_tensor(params['spectral_smoothing_kernel'], dtype=self.dtype).to(self.device)
        self.temporal_window = params['temporal_window']
        self.do_spectralPenalty = params['do_spectralPenalty']
        self.loss_running = params['loss_running']
        self.shifter = params['shifter']


    def display_params(self):
        """
        Display the parameters of the model.
        RH 2021
        """
        # print('X:', self.X.shape)
        # print('y:', self.y.shape)
        print('weights:', self.weights)
        print('Bcp:', self.Bcp)
        print('non_negative:', self.non_negative)
        print('softplus_kwargs:', self.softplus_kwargs)
        print('rank:', self.rank)
        print('device:', self.device)
        print('loss_running:', self.loss_running)
        
    def plot_outputs(self):
        """
        Plot the outputs of the model.
        RH 2021
        """
        plt.figure()
        plt.plot(self.loss_running)
        plt.xlabel('logged iteration')
        plt.ylabel('loss')
        plt.title('loss')

        Bcp_n_final, Bcp_w_final = self.return_Bcp_final()
       
        if self.rank_normal > 0:
            fig_n, axs = plt.subplots(len(Bcp_n_final))
            for ii, val in enumerate(Bcp_n_final):
                axs[ii].set_title(f'factor {ii+1}')
                axs[ii].plot(val.squeeze())
            fig_n.suptitle('Bcp_n components')

        if self.rank_spectral > 0:
            fig_c, axs = plt.subplots(len(Bcp_w_final[1:]) + Bcp_w_final[0].shape[1])
            jj = 0
            for ii, val in enumerate(Bcp_w_final):
                axs[ii+jj].set_title(f'factor {ii+1}')
                if ii==0:
                    for jj in range(val.shape[1]):
                        axs[jj].plot(val[:,jj,:].squeeze())
                else:
                    axs[ii+jj].plot(val.squeeze())
            fig_c.suptitle('Bcp_w components')

    def update_plot_outputs(self, fig, axs):
        """
        Update the outputs of the model.
        RH 2021
        """
        axs[0].clear()
        axs[0].plot(self.loss_running)
        axs[0].set_title('loss')

        axs[1].clear()

        Bcp_n_final, Bcp_w_final = self.return_Bcp_final()
       

        ii=0
        for ii, val in enumerate(Bcp_n_final):
            axs[ii+1].clear()
            axs[ii+1].plot(val.squeeze())
            axs[ii+1].set_title(f'factor {ii+1}')
            # axs[ii+1].set_ylim(bottom=-0.09, top=0.09)
        
        jj=0
        for jj in range(self.rank_normal):
            if self.rank_normal == 1:
                Bcp_w_final[0] = Bcp_w_final[0][:,None]
            axs[ii+jj+2].clear()
            axs[ii+jj+2].plot(Bcp_w_final[0][:,jj].squeeze())
            axs[ii+jj+2].set_title(f'factor {jj}')
        
        for kk in range(self.rank_spectral):
            if self.rank_spectral == 1:
                Bcp_w_final[1] = Bcp_w_final[1][:,None]
            axs[kk+ii+jj+3].clear()
            axs[kk+ii+jj+3].plot(Bcp_w_final[1][:,kk].squeeze())
            axs[kk+ii+jj+3].set_title(f'factor {kk+1}')

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.0001)