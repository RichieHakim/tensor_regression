import numpy as np
import copy
import gc
import matplotlib.pyplot as plt

import torch
import torch.cuda
from torch.autograd import Variable
from torch.optim import LBFGS

import tensorly as tl

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
    # Bcp_init = [torch.nn.init.orthogonal_(torch.empty(B_dims[ii], rank, complex_dims[ii], dtype=dtype), gain=scale).to(device) for ii in range(len(B_dims))] # initialize orthogonal matrices
    Bcp_init = [torch.nn.init.orthogonal_(torch.empty(B_dims[ii], rank, complex_dims[ii], dtype=dtype), gain=scale).to(device) for ii in range(len(B_dims))] # initialize orthogonal matrices
    Bcp_init = [(Bcp_init[ii] + torch.std(Bcp_init[ii])*2*non_negative[ii])/((non_negative[ii]+1)) if Bcp_init[0].shape[0]>1 else Bcp_init[ii] for ii in range(len(Bcp_init))] # make non-negative by adding 2 std to each non_neg component and dividing by 2. Only if an std can be calculated (if n samples > 1)
    
    # Bcp_init = [torch.nn.init.orthogonal_(     torch.empty(B_dims[ii], rank, complex_dims[ii], dtype=dtype), gain=scale).to(device) for ii in range(len(B_dims))] # initialize orthogonal matrices
    # Bcp_init = [(torch.nn.init.ones_(torch.empty(B_dims[ii], rank, complex_dims[ii], dtype=dtype)) * scale).to(device) for ii in range(len(B_dims))] # initialize orthogonal matrices
    # Bcp_init = [torch.nn.init.kaiming_uniform_(torch.empty(B_dims[ii], rank), a=0, mode='fan_in').to(device) for ii in range(len(B_dims))]
    # Bcp_init = [(torch.nn.init.orthogonal_(torch.empty(B_dims[ii], rank), gain=scale) + torch.nn.init.ones_(torch.empty(B_dims[ii], rank))).to(device) for ii in range(len(B_dims))]
    # Bcp_init = [torch.nn.init.ones_(torch.empty(B_dims[ii], rank)).to(device) * scale for ii in range(len(B_dims))]
    # Bcp_init = [torch.nn.init.sparse_(torch.empty(B_dims[ii], rank), sparsity=0.75, std=scale).to(device) for ii in range(len(B_dims))]
    # Bcp_init = [(torch.rand((B_dims[ii], rank))*scale - (1-non_negative[ii])*(scale/2)).to(device) for ii in range(len(B_dims))]
    # Bcp_init = [(torch.randn((B_dims[ii], rank))*0.0025).to(device) for ii in range(len(B_dims))]
    # for ii in range(len(B_dims)):
    #     Bcp_init[ii].requires_grad = True

    return Bcp_init
    
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


def edge_clamp(B_cp, edge_idx, clamp_val=0, device='cpu', dtype=torch.float32):
    """
    Clamp edges of each component of Bcp to 
    clamp_val.
    RH 2020
    
    Args:
        B_cp (list of torch.Tensor):
            Beta Kruskal tensor.
            List of tensors of shape
             (n_features, rank).
    
    Returns:
        B_cp_clamped (list of torch.Tensor):
            Beta Kruskal tensor with edges clamped.
    """
    eIdxBool = torch.ones(B_cp[0].shape[0], dtype=dtype, device=device)
    eIdxBool[edge_idx] = clamp_val
    return [B_cp[0]*eIdxBool[:,None,None] , B_cp[1], B_cp[2]]


def lin_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
    """
    Compute the regression model.
    y_hat = inner(X, outer(softplus(Bcp))
    where:
        X.shape[1:] == Bcp.shape[:-1]
        X.shape[0] == len(y_hat)
        Bcp.shape[-1] == len(unique(y_true))
        softplus is performed only on specified dimensions of Bcp.
        inner prod is performed on dims [1:] of X and
         dims [:-1] of Bcp.
    JZ2021 / RH 2021

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
    
    if Bcp[0].shape[1] == 0:
        return torch.zeros(1).to(X.device)

    # Bcp = edge_clamp(Bcp, edge_idx=torch.hstack([torch.arange(0,600), torch.arange(1320,1920)]), clamp_val=0, device=X.device, dtype=X.dtype)
    return tl.tenalg.inner(X,
                           tl.cp_tensor.cp_to_tensor((weights, list(non_neg_fn(
                                                                                [Bcp[ii][:,:,0] for ii in range(len(Bcp))],
                                                                                non_negative,
                                                                                softplus_kwargs))
                                                     )),
                           n_modes=X.ndim-1
                        ).squeeze() + bias


def spectral_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
    """
    Compute the regression model.
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
            Complex Beta Kruskal tensor (before softplus). (Bcp_c)
            List of tensors of shape 
             (n_features, rank, complex_dimensionality).
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

    if Bcp[0].shape[1] == 0:
        return torch.zeros(1).to(X.device)

    y_hat_all = []
    for ii in range(Bcp[0].shape[2]):
        y_hat_all += [tl.tenalg.inner(X,
                           tl.cp_tensor.cp_to_tensor((weights, list(non_neg_fn(
                                                                                [Bcp[0][:,:,ii]] + [ Bcp[jj][:,:,0] for jj in range(1,len(Bcp)) ],
                                                                                # [Bcp[0][:,:,ii] - torch.mean(Bcp[0][:,:,ii])] + [ Bcp[jj][:,:,0] for jj in range(1,len(Bcp)) ],
                                                                                non_negative,
                                                                                softplus_kwargs))
                                                     )),
                           n_modes=X.ndim-1
                        ).squeeze()[None,...]]
                        
    y_hat = torch.norm(torch.vstack(y_hat_all), dim=0) + bias
    return y_hat


def stepwise_linear_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
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

    if Bcp[0].shape[1] == 0:
        return torch.zeros(1).to(X.device)

    # make non-negative
    Bcp_nn = list(non_neg_fn(Bcp, non_negative, softplus_kwargs))
    
    # X_1 = torch.einsum('twd,wrc -> tdrc', X, Bcp_nn[0])
    # X_1b = torch.norm(X_1, dim=3)
    # X_2 = torch.einsum('tdr,drn -> trn', X_1b, Bcp_nn[1])
    # X_3 = torch.einsum('trn -> tn', X_2)
    # return X_3 + bias
    
    # X_1a = torch.einsum('twd,wrs -> tdr', X, Bcp_nn[0])
    # X_1b = torch.einsum('tdr,drs -> tr', X_1a, Bcp_nn[1])
    # X_1c = torch.einsum('tr,nrs -> tn', X_1b, Bcp_nn[2]) + bias

    X_1c = torch.einsum('tr,nrs -> tn', 
                        torch.einsum('tdr,drs -> tr',
                                     torch.einsum('twd,wrs -> tdr',
                                                     X, Bcp_nn[0]),
                                     Bcp_nn[1]), 
                        Bcp_nn[2]) + bias
    return X_1c


def stepwise_latents_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
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

    if Bcp[0].shape[1] == 0:
        return torch.zeros(1).to(X.device)

    # make non-negative
    Bcp_nn = list(non_neg_fn(Bcp, non_negative, softplus_kwargs))
    
    # X_1 = torch.einsum('twd,wrc -> tdrc', X, Bcp_nn[0])
    # X_1b = torch.norm(X_1, dim=3)
    # X_2 = torch.einsum('tdr,drn -> trn', X_1b, Bcp_nn[1])
    # X_3 = torch.einsum('trn -> tn', X_2)
    # return X_3 + bias
    
    X_1a = torch.einsum('twd,wrs -> tdr', X, Bcp_nn[0])
    X_1b = torch.einsum('tdr,drs -> tr', X_1a, Bcp_nn[1])
    # X_1c = torch.einsum('tr,dr -> dt', X_1a, Bcp_nn[1]) + bias
    # return X_1c, X_1b
    return X_1b


def stepwise_spectral_model(X, Bcp, weights, non_negative, bias, softplus_kwargs=None):
    """
    Computes spectral regression model in a stepwise manner.
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

    if Bcp[0].shape[1] == 0:
        return torch.zeros(1).to(X.device)

    # make non-negative
    Bcp_nn = list(non_neg_fn(Bcp, non_negative, softplus_kwargs))
    
    # X_1 = torch.einsum('twd,wrc -> tdrc', X, Bcp_nn[0])
    # X_1b = torch.norm(X_1, dim=3)
    # X_2 = torch.einsum('tdr,drn -> trn', X_1b, Bcp_nn[1])
    # X_3 = torch.einsum('trn -> tn', X_2)
    # return X_3 + bias
    
    X_1a = torch.norm(torch.einsum('twd,wrc -> tdrc', X, Bcp_nn[0]), dim=3)
    X_1b = torch.einsum('tdr,drs -> tr', X_1a, Bcp_nn[1])
    X_1c = torch.einsum('tr,nrs -> tn', X_1b, Bcp_nn[2]) + bias
    return X_1c


def L2_penalty(B_cp):
    """
    Compute the L2 penalty.
    RH 2021

    Args:
        B_cp (list of torch.Tensor):
            Beta Kruskal tensor (before softplus)
    
    Returns:
        L2_penalty (torch.Tensor):
            L2 penalty.
    """
    ii=0
    for comp in B_cp:
    # for comp in B_cp[1]:
        ii+= torch.sqrt(torch.sum(comp**2))
        
        # val = torch.sqrt(torch.mean(comp**2))
        # if torch.isnan(val):
        #     val = 0
        # else:
        #     ii+= torch.sqrt(torch.mean(comp**2))

    return ii


####################################
########### Main class #############
####################################

class CP_linear_regression():
    def __init__(self, 
                    X_shape, 
                    y_shape,
                    dtype=torch.float32,
                    rank_normal=1, 
                    rank_spectral=1,
                    non_negative=False, 
                    weights=None, 
                    Bcp_init=None, 
                    Bcp_init_scale=1, 
                    n_complex_dim=0, 
                    bias_init=0, 
                    device='cpu', 
                    softplus_kwargs=None):
        """
        Multinomial logistic CP tensor regression class.
        Bias is not considered in this model because there
         are multiple ways to handle bias with tensor
         regression. User must add a bias entry to X
         manually if they wish. Mean subtracting X along
         one dimension is recommended.
        RH 2021

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

        self.rank_normal = rank_normal
        self.rank_spectral = rank_spectral
        self.rank = rank_normal + rank_spectral
        self.device = device

        if non_negative == True:
            self.non_negative = [True]*(len(X_shape))
        elif non_negative == False:
            self.non_negative = [False]*(len(X_shape))
        else:
            self.non_negative = non_negative        
        
        # if len(y_shape) == 1:
        #     y_shape = list(y_shape) + [1]
        # self.bias = torch.tensor([bias_init], dtype=torch.float32, requires_grad=True, device=device) 
        self.bias = torch.zeros(y_shape[1:], dtype=self.dtype, requires_grad=True, device=device) 
        self.y_shape = y_shape

        # B_dims = list(X_shape[1:])
        B_dims = list(X_shape[1:]) + list(y_shape[1:])
        complex_dims = list([n_complex_dim+1] + [1]*(len(B_dims)-1))
        if Bcp_init is None:
            self.Bcp_n = make_BcpInit(B_dims, self.rank_normal,   self.non_negative, complex_dims=None,         scale=Bcp_init_scale, device=self.device, dtype=self.dtype) # 'normal Beta_cp Kruskal tensor'
            self.Bcp_c = make_BcpInit(B_dims, self.rank_spectral, self.non_negative, complex_dims=complex_dims, scale=Bcp_init_scale, device=self.device, dtype=self.dtype) # 'complex Beta_cp Kruskal tensor'
            # y_scale = torch.var(lin_model(self.X, self.Bcp, self.weights, self.non_negative, self.bias, self.softplus_kwargs).detach()) / torch.var(self.y)
            # print(f'final y_init: {y_scale}')
            for ii in range(len(B_dims)):
                # self.Bcp[ii] = self.Bcp[ii] / y_scale
                self.Bcp_n[ii].requires_grad = True
                self.Bcp_c[ii].requires_grad = True
            # y_scale_final = torch.var(lin_model(self.X, self.Bcp, self.weights, self.non_negative, self.bias, self.softplus_kwargs).detach()) / torch.var(self.y)
            # print(f'final y_scale: {y_scale_final}')
        else:
            self.Bcp_n = Bcp_init[0]
            self.Bcp_c = Bcp_init[1]

        self.loss_running = []

    def fit(self,
            X,
            y,
            lambda_L2=0.01, 
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

        tl.set_backend('pytorch')

        self.optimizer = torch.optim.LBFGS(self.Bcp_n + self.Bcp_c + [self.bias], **LBFGS_kwargs)

        def closure():
            self.optimizer.zero_grad()
            y_hat = lin_model(X, self.Bcp_n, self.weights[:self.rank_normal], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs) + \
                        stepwise_spectral_model(X, self.Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)
            # y_hat = (stepwise_linear_model(X, self.Bcp_n, self.weights[:self.rank_normal], self.non_negative, self.bias,softplus_kwargs=self.softplus_kwargs) + \
            #         stepwise_spectral_model(X, self.Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)).detach()
            loss = loss_fn(y_hat, y) + lambda_L2 * (L2_penalty(self.Bcp_n) + L2_penalty(self.Bcp_c))
            loss.backward()
            return loss
            
        loss_fn = torch.nn.MSELoss()
        # loss_fn = torch.nn.HuberLoss(delta=1)

        if verbose==3:
            self.fig, self.axs = plt.subplots(1 + len(self.Bcp_n) + len(self.Bcp_c) + self.Bcp_c[0].shape[1], figsize=(7,20))

        convergence_reached = False
        for ii in range(max_iter):
            if ii%running_loss_logging_interval == 0:
                y_hat = (lin_model(X, self.Bcp_n, self.weights[:self.rank_normal], self.non_negative, self.bias,softplus_kwargs=self.softplus_kwargs) + \
                        stepwise_spectral_model(X, self.Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)).detach()
                # y_hat = (stepwise_linear_model(X, self.Bcp_n, self.weights[:self.rank_normal], self.non_negative, self.bias,softplus_kwargs=self.softplus_kwargs) + \
                #         stepwise_spectral_model(X, self.Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)).detach()
                self.loss_running.append((loss_fn(y_hat, y) + lambda_L2 * (L2_penalty(self.Bcp_n) + L2_penalty(self.Bcp_c))).item())
                if verbose==2:
                    print(f'Iteration: {ii}, Loss: {self.loss_running[-1]}  ;  Variance ratio (y_hat / y_true): {torch.var(y_hat.detach()).item() / torch.var(y).item()}' )
                elif verbose==3:
                    self.update_plot_outputs(self.fig, self.axs)
                    # plt.pause(0.01)
                    print(f'Iteration: {ii}, Loss: {self.loss_running[-1]}  ;  Variance ratio (y_hat / y_true): {torch.var(y_hat.detach()).item() / torch.var(y).item()}' )
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

        self.optimizer = torch.optim.Adam(self.Bcp_n + self.Bcp_c + [self.bias], **Adam_kwargs)
        # optimizer = torch.optim.Adam(self.Bcp + [self.weights], **Adam_kwargs)
        loss_fn = torch.nn.MSELoss()

        if verbose==3:
            self.fig, self.axs = plt.subplots(1 + len(self.Bcp_n) + len(self.Bcp_c) + self.Bcp_c[0].shape[1], figsize=(7,20))

        convergence_reached = False
        for ii in range(max_iter):
            self.optimizer.zero_grad()
            y_hat = (lin_model(X, self.Bcp_n, self.weights[:self.rank_normal], self.non_negative, self.bias,softplus_kwargs=self.softplus_kwargs) + \
                    stepwise_spectral_model(X, self.Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs))
                    # spectral_model(X, self.Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)
            loss = loss_fn(y_hat, y) + lambda_L2 * (L2_penalty(self.Bcp_n) + L2_penalty(self.Bcp_c))
            loss.backward()
            self.optimizer.step()
            self.loss_running.append(loss.item())
            if verbose==2:
                print(f'Iteration: {ii}, Loss: {self.loss_running[-1]}  ;  Variance ratio (y_hat / y_true): {torch.var(y_hat.detach()).item() / torch.var(y).item()}' )
            elif verbose==3 and ii%plotting_interval == 0:
                self.update_plot_outputs(self.fig, self.axs)
                # plt.pause(0.01)
                print(f'Iteration: {ii}, Loss: {self.loss_running[-1]}  ;  Variance ratio (y_hat / y_true): {torch.var(y_hat.detach()).item() / torch.var(y).item()}' )
            if ii > patience:
                if np.sum(np.abs(np.diff(self.loss_running[ii-patience:]))) < tol:
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
                Bcp[0] should be Bcp_n, Bcp[1] should be Bcp_c.
                Note that Bcp_c[0] should be shape (X.shape[1],
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
            Bcp_c = self.Bcp_c
        elif isinstance(Bcp[0][0], torch.Tensor) == False:
            Bcp_n = [0]*len(Bcp[0])
            for ii in range(len(Bcp[0])):
                Bcp_n[ii] = torch.tensor(Bcp[0][ii], dtype=torch.float32, requires_grad=False).to(device)
            Bcp_c = [0]*len(Bcp[1])
            for ii in range(len(Bcp[1])):
                Bcp_c[ii] = torch.tensor(Bcp[1][ii], dtype=torch.float32, requires_grad=False).to(device)
        elif Bcp[0][0].device != device:
            Bcp_n = [0]*len(Bcp[0])
            for ii in range(len(Bcp[0])):
                Bcp_n[ii] = Bcp[0][ii].to(device)
            Bcp_c = [0]*len(Bcp[1])
            for ii in range(len(Bcp[1])):
                Bcp_c[ii] = Bcp[1][ii].to(device)

        y_hat = lin_model(X, Bcp_n, self.weights[:self.rank_normal], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs) + \
                spectral_model(X, Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)
        # y_hat = spectral_model(X, Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)

        return y_hat.cpu().detach()

    
    def predict_latents(self, X, Bcp=None, device=None, plot_pref=False):
        """
        Predict latent variables for X given a Bcp (beta Kruskal tensor).
        Uses 'model' function in this module.
        RH 2021

        Args:
            X (np.ndarray or torch.Tensor):
                Input data. First dimension must match len(y).
                If None, then use the data that was passed to the
                 constructor.
            Bcp (list of np.ndarray or torch.Tensor):
                List of Bcp lists of matrices.
                Bcp[0] should be Bcp_n, Bcp[1] should be Bcp_c.
                Note that Bcp_c[0] should be shape (X.shape[1],
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
            Bcp_c = self.Bcp_c
        elif isinstance(Bcp[0][0], torch.Tensor) == False:
            Bcp_n = [0]*len(Bcp[0])
            for ii in range(len(Bcp[0])):
                Bcp_n[ii] = torch.tensor(Bcp[0][ii], dtype=torch.float32, requires_grad=False).to(device)
            Bcp_c = [0]*len(Bcp[1])
            for ii in range(len(Bcp[1])):
                Bcp_c[ii] = torch.tensor(Bcp[1][ii], dtype=torch.float32, requires_grad=False).to(device)
        elif Bcp[0][0].device != device:
            Bcp_n = [0]*len(Bcp[0])
            for ii in range(len(Bcp[0])):
                Bcp_n[ii] = Bcp[0][ii].to(device)
            Bcp_c = [0]*len(Bcp[1])
            for ii in range(len(Bcp[1])):
                Bcp_c[ii] = Bcp[1][ii].to(device)

        y_hat = stepwise_latents_model(X, Bcp_n, self.weights[:self.rank_normal], self.non_negative, bias=None, softplus_kwargs=self.softplus_kwargs)
                # spectral_model(X, Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)
        # y_hat = spectral_model(X, Bcp_c, self.weights[self.rank_normal:], self.non_negative, self.bias, softplus_kwargs=self.softplus_kwargs)

        return y_hat.cpu().detach().numpy()


    
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
        Bcp_n = list(non_neg_fn(self.Bcp_n, self.non_negative, softplus_kwargs=self.softplus_kwargs))
        Bcp_c = list(non_neg_fn(self.Bcp_c, self.non_negative, softplus_kwargs=self.softplus_kwargs))
        Bcp_n_nonNeg = [Bcp_n[ii].detach().cpu().numpy() for ii in range(len(Bcp_n))]
        Bcp_c_nonNeg = [Bcp_c[ii].detach().cpu().numpy() for ii in range(len(Bcp_c))]
        return Bcp_n_nonNeg , Bcp_c_nonNeg
    
    def detach_Bcp(self):
        """
        Detach the Bcp Kruskal tensor list.
        RH 2021

        Returns:
            Bcp_detached (list of np.ndarray):
                Detached Bcp tensors.
        """
        Bcp_n_detached = [Bcp.detach().cpu().numpy() for Bcp in self.Bcp_n]
        Bcp_c_detached = [Bcp.detach().cpu().numpy() for Bcp in self.Bcp_c]
        return Bcp_n_detached, Bcp_c_detached

    def get_params(self):
        """
        Get the parameters of the model.
        RH 2021
        """
        return {
                # 'X': self.X.detach().cpu().numpy(),
                # 'y': self.y.detach().cpu().numpy(),
                'weights': self.weights.detach().cpu().numpy(),
                'Bcp_n': self.detach_Bcp()[0],
                'Bcp_c': self.detach_Bcp()[1],
                'non_negative': self.non_negative,
                'softplus_kwargs': self.softplus_kwargs,
                'rank': self.rank,
                'device': self.device,
                'loss_running': self.loss_running}

    def set_params(self, params):
        """
        Set the parameters of the model.
        RH 2021

        Args:
            params (dict):
                Dictionary of parameters.
        """
        # self.X = params['X']
        # self.y = params['y']
        self.weights = params['weights']
        self.Bcp = params['Bcp']
        self.non_negative = params['non_negative']
        self.softplus_kwargs = params['softplus_kwargs']
        self.rank = params['rank']
        self.device = params['device']
        self.loss_running = params['loss_running']

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

        Bcp_n_final, Bcp_c_final = self.return_Bcp_final()
       
        if self.rank_normal > 0:
            fig_n, axs = plt.subplots(len(Bcp_n_final))
            for ii, val in enumerate(Bcp_n_final):
                axs[ii].set_title(f'factor {ii+1}')
                axs[ii].plot(val.squeeze())
            fig_n.suptitle('Bcp_n components')

        if self.rank_spectral > 0:
            fig_c, axs = plt.subplots(len(Bcp_c_final[1:]) + Bcp_c_final[0].shape[1])
            jj = 0
            for ii, val in enumerate(Bcp_c_final):
                axs[ii+jj].set_title(f'factor {ii+1}')
                if ii==0:
                    for jj in range(val.shape[1]):
                        axs[jj].plot(val[:,jj,:].squeeze())
                else:
                    axs[ii+jj].plot(val.squeeze())
            fig_c.suptitle('Bcp_c components')

    def update_plot_outputs(self, fig, axs):
        """
        Update the outputs of the model.
        RH 2021
        """
        axs[0].clear()
        axs[0].plot(self.loss_running)
        axs[0].set_title('loss')

        axs[1].clear()
        # axs[1].plot(self.optimizer.)

        Bcp_n_final, Bcp_c_final = self.return_Bcp_final()
       
        if self.rank_normal > 0:
            for ii, val in enumerate(Bcp_n_final):
                axs[ii+1].clear()
                axs[ii+1].plot(val.squeeze())
                axs[ii+1].set_title(f'factor {ii+1}')
                # axs[ii+1].set_ylim(bottom=-0.09, top=0.09)

        if self.rank_spectral > 0:
            for ii, val in enumerate(Bcp_c_final):
                if ii==0:
                    for jj in range(val.shape[1]):
                        axs[ii+1+len(Bcp_n_final)+jj].clear()
                        axs[ii+1+len(Bcp_n_final)+jj].plot(val[:,jj,:].squeeze())
                        axs[ii+1+len(Bcp_n_final)+jj].set_title(f'factor {ii+1}')
                else:
                    axs[ii+1+len(Bcp_n_final)+jj].clear()
                    axs[ii+1+len(Bcp_n_final)+jj].plot(val.squeeze())
                    axs[ii+1+len(Bcp_n_final)+jj].set_title(f'factor {ii+1}')

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.0001)