import numpy as np
import copy
import matplotlib.pyplot as plt

import torch
import torch.cuda
from torch.autograd import Variable
from torch.optim import LBFGS

import tensorly as tl

####################################
######## Useful functions ##########
####################################

def squeeze_integers(intVec):
    """
    Make integers in an array consecutive numbers
     starting from 0. ie. [7,2,7,4,1] -> [3,2,3,1,0].
    Useful for removing unused class IDs from y_true
     and outputting something appropriate for softmax.
    This is v2. The old version is busted.
    RH 2021
    
    Args:
        intVec (np.ndarray):
            1-D array of integers.
    
    Returns:
        intVec_squeezed (np.ndarray):
            1-D array of integers with consecutive numbers
    """
    uniques = np.unique(intVec)
    unique_positions = np.arange(len(uniques))
    return unique_positions[np.array([np.where(intVec[ii]==uniques)[0] for ii in range(len(intVec))]).squeeze()]


####################################
######## Helper functions ##########
####################################
    

def confusion_matrix(y_hat, y_true):
    """
    Compute the confusion matrix from y_hat and y_true.
    y_hat should be either predictions ().
    RH 2021

    Args:
        y_hat (np.ndarray): 
            numpy array of predictions or probabilities. 
            Either PREDICTIONS: 2-D array of booleans
             ('one hots') or 1-D array of predicted 
             class indices.
            Or PROBABILITIES: 2-D array floats ('one hot
             like')
        y_true (np.ndarray):
            1-D array of true class indices.
    """
    n_classes = np.max(y_true)+1
    if y_hat.ndim == 1:
        y_hat = idx_to_oneHot(y_hat, n_classes)
    cmat = y_hat.T @ idx_to_oneHot(y_true, n_classes)
    return cmat / np.sum(cmat, axis=0)[None,:]
    
def idx_to_oneHot(arr, n_classes=None):
    """
    Convert an array of class indices to matrix of
     one-hot vectors.
    RH 2021

    Args:
        arr (np.ndarray):
            1-D array of class indices.
        n_classes (int):
            Number of classes.
    
    Returns:
        oneHot (np.ndarray):
            2-D array of one-hot vectors.
    """
    if n_classes is None:
        n_classes = np.max(arr)+1
    oneHot = np.zeros((arr.size, n_classes))
    oneHot[np.arange(arr.size), arr] = 1
    return oneHot

def make_BcpInit(B_dims, rank, non_negative, scale=1, device='cpu'):
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
        scale (float):
            Scale of uniform distribution used to initialize
             each component.
        device (str):
            Device to use.
    
    Returns:
        B_cp (list of torch.Tensor):
            Beta Kruskal tensor.
    """
    Bcp_init = [(torch.rand((B_dims[ii], rank))*scale - (1-non_negative[ii])*(scale/2)).to(device) for ii in range(len(B_dims))]
    for ii in range(len(B_dims)):
        Bcp_init[ii].requires_grad = True
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

def model(X, Bcp, weights, non_negative, softplus_kwargs=None):
    """
    Compute the regression model.
    y_hat = softmax(inner(X, outer(softplus(Bcp)), n_modes=len(Bcp)-1))
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
            Beta Kruskal tensor (before softplus).
            List of tensors of shape 
             (n_features, rank).
        weights (list of floats):
            List of weights for each component.
            len(weights) == rank == Bcp[0].shape[1]
        non_negative (list of booleans):
            List of booleans indicating whether each component
             is non-negative.
        softplus_kwargs (dict):
            Keyword arguments for torch.nn.functional.softplus.
    
    Returns:
        y_hat (torch.Tensor):
            N-D array of predictions.
    """
    return torch.nn.functional.softmax(
                tl.tenalg.inner(X,
                    tl.cp_tensor.cp_to_tensor((weights, list(non_neg_fn(
                                                                Bcp,
                                                                non_negative,
                                                                softplus_kwargs)) )),
                    n_modes=len(Bcp)-1),
                dim=1)
        
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
        ii+= torch.sqrt(torch.sum(comp**2))
    return ii


####################################
########### Main class #############
####################################

class CP_logistic_regression():
    def __init__(self, X, y, rank=5, non_negative=False, weights=None, Bcp_init=None, Bcp_init_scale=1, device='cpu', softplus_kwargs=None):
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
            device (str):
                Device to run the model on.
        """        

        self.X = torch.as_tensor(X, dtype=torch.float32).to(device)
        self.y = torch.as_tensor(y, dtype=torch.long).to(device)
        
        if weights is None:
            self.weights = torch.ones((rank), device=device)
        else:
            self.weights = torch.tensor(weights)

        if softplus_kwargs is None:
            self.softplus_kwargs = {'beta': 50, 'threshold': 1}
        else:
            self.softplus_kwargs = softplus_kwargs

        self.rank = rank
        self.device = device

        if non_negative == True:
            self.non_negative = [True]*(self.X.ndim)
        elif non_negative == False:
            self.non_negative = [False]*(self.X.ndim)
        else:
            self.non_negative = non_negative        

        
        self.n_classes = len(torch.unique(self.y))
        B_dims = np.concatenate((np.array(self.X.shape[1:]), [self.n_classes]))
        if Bcp_init is None:
            self.Bcp = make_BcpInit(B_dims, self.rank, self.non_negative, scale=Bcp_init_scale, device=self.device)
        else:
            self.Bcp = Bcp_init

        self.loss_running = []

    def fit(self,
            lambda_L2=0.01, 
            max_iter=1000, 
            tol=1e-5, 
            patience=10,
            weights=None,
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
            weights (np.array):
                Weights for each class.
                If None, then all classes are weighted equally.
                To balance the classes, set weights to be the
                 the inverse of the class counts.
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
                'lr' : 10000, 
                'max_iter' : 20, 
                'max_eval' : None, 
                'tolerance_grad' : 1e-07, 
                'tolerance_change' : 1e-09, 
                'history_size' : 100, 
                'line_search_fn' : "strong_wolfe"
            }

        tl.set_backend('pytorch')

        optimizer = torch.optim.LBFGS(self.Bcp, **LBFGS_kwargs)

        def closure():
            optimizer.zero_grad()
            y_hat = model(self.X, self.Bcp, self.weights, self.non_negative, softplus_kwargs=self.softplus_kwargs)
            loss = loss_fn(y_hat, self.y) + lambda_L2 * L2_penalty(self.Bcp)
            loss.backward()
            return loss
            
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, dtype=torch.float32).to(self.device)
        )

        convergence_reached = False
        for ii in range(max_iter):
            if ii%running_loss_logging_interval == 0:
                y_hat = model(self.X, self.Bcp, self.weights, self.non_negative, softplus_kwargs=self.softplus_kwargs)
                self.loss_running.append(loss_fn(y_hat, self.y).item())
                if verbose==2:
                    print(f'Iteration: {ii}, Loss: {self.loss_running[-1]}')

            if ii > patience:
                if np.sum(np.abs(np.diff(self.loss_running[ii-patience:]))) < tol:
                    convergence_reached = True
                    break

            optimizer.step(closure)
        if (verbose==True) or (verbose>=1):
            if convergence_reached:
                print('Convergence reached')
            else:
                print('Reached maximum number of iterations without convergence')
        return convergence_reached

    def fit_Adam(self,
            lambda_L2=0.01, 
            max_iter=1000, 
            tol=1e-5, 
            patience=10,
            weights=None,
            verbose=False,
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
            weights (np.array):
                Weights for each class.
                If None, then all classes are weighted equally.
                To balance the classes, set weights to be the
                 the inverse of the class counts.
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

        optimizer = torch.optim.Adam(self.Bcp, **Adam_kwargs)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, dtype=torch.float32).to(self.device),
        )

        convergence_reached = False
        for ii in range(max_iter):
            optimizer.zero_grad()
            y_hat = model(self.X, self.Bcp, self.weights, self.non_negative, softplus_kwargs=self.softplus_kwargs)
            loss = loss_fn(y_hat, self.y) + lambda_L2 * L2_penalty(self.Bcp)
            loss.backward()
            optimizer.step()
            self.loss_running.append(loss.item())
            if verbose==2:
                print(f'Iteration: {ii}, Loss: {self.loss_running[-1]}')
            if ii > patience:
                if np.sum(np.abs(np.diff(self.loss_running[ii-patience:]))) < tol:
                    convergence_reached = True
                    break
        if (verbose==True) or (verbose>=1):
            if convergence_reached:
                print('Convergence reached')
            else:
                print('Reached maximum number of iterations without convergence')
        return convergence_reached


    def predict(self, X=None, y_true=None, Bcp=None, device=None):
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
                List of Bcp matrices.
                If None, then use the Bcp that was passed to the
                 constructor.
            y_true (np.ndarray or torch.Tensor):
                True class labels. Only necessary if plot_pref is
                 True and inputting data from a different X than
                 what the model was trained on.
            device (str):
                Device to run the model on.
        Returns:
            logits (np.ndarray):
                Raw output of the model.
            preds (np.ndarray):
                Predicted class labels.
            cm (np.ndarray):
                Confusion matrix.
            acc (float):
                Accuracy.
        """
        if device is None:
            device = self.device
        
        if X is None:
            X = self.X
        elif isinstance(X, torch.Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32, requires_grad=False).to(device)
        elif X.device != device:
            X = X.to(device)

        if y_true is None:
            y_true = self.y.detach().cpu().numpy()

                        
        if Bcp is None:
            Bcp = self.Bcp
        elif isinstance(Bcp[0], torch.Tensor) == False:
            for ii in range(len(Bcp)):
                Bcp[ii] = torch.tensor(Bcp[ii], dtype=torch.float32, requires_grad=False).to(device)
        elif Bcp[0].device != device:
            for ii in range(len(Bcp)):
                Bcp[ii] = Bcp[ii].to(device)

        logit = model(X, Bcp, self.weights, self.non_negative, softplus_kwargs=self.softplus_kwargs).detach().cpu().numpy()
        pred = np.argmax(logit, axis=1)
        # pred_onehot = idx_to_oneHot(pred, self.n_classes)

        # cm, acc = self.make_confusion_matrix(prob_or_pred='pred', pred=pred, y_true=y_true)

        # if plot_pref:
        #     fig, axs = plt.subplots(2)
        #     axs[0].imshow(idx_to_oneHot(pred, self.n_classes), aspect='auto', interpolation='none')
        #     axs[1].imshow(idx_to_oneHot(y_true, self.n_classes), aspect='auto', interpolation='none')
        #     axs[1].set_xlabel('class')
        #     fig.suptitle('predictions')

        #     fig = plt.figure()
        #     plt.imshow(cm)
        #     plt.ylabel('true class')
        #     plt.xlabel('predicted class')
        #     plt.title('confusion matrix (predictions)')
            
        return logit, pred

    
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
        Bcp = list(non_neg_fn(self.Bcp, self.non_negative, softplus_kwargs=self.softplus_kwargs))
        Bcp_nonNeg = [Bcp[ii].detach().cpu().numpy() for ii in range(len(Bcp))]
        return Bcp_nonNeg

    def make_confusion_matrix(self, prob_or_pred='pred', prob=None, pred=None, y_true=None):
        """
        Make a confusion matrix.
        If 'prob' and 'pred' are None, then they will
         be calculated using self.predict().
        RH 2021

        Args:
            prob_or_pred (str):
                'prob' or 'pred'. If 'prob', then use the
                 probabilities of the model. If 'pred', then
                 use the predicted class labels.
            prob (np.ndarray):
                Probabilities of the model.
            pred (np.ndarray):
                Predicted class labels.

        Returns:
            cm (np.ndarray):
                Confusion matrix.
            acc (float):
                Accuracy of the model.
        """
        if (prob is None) and (pred is None):
            prob, pred = self.predict()
            cm, acc = self.make_confusion_matrix(prob_or_pred='pred', pred=pred, y_true=y_true)

        if y_true is None:
            y_true = self.y.detach().cpu().numpy()
        
        if prob_or_pred == 'pred':
            cm = confusion_matrix(pred, y_true)
        elif prob_or_pred == 'prob':
            cm = confusion_matrix(prob, y_true)
        
        acc = np.sum(np.diag(cm))/np.sum(cm)

        return cm, acc
    
    def detach_Bcp(self):
        """
        Detach the Bcp Kruskal tensor list.
        RH 2021

        Returns:
            Bcp_detached (list of np.ndarray):
                Detached Bcp tensors.
        """
        Bcp_detached = [Bcp.detach().cpu().numpy() for Bcp in self.Bcp]
        return Bcp_detached

    def get_params(self):
        """
        Get the parameters of the model.
        RH 2021
        """
        return {'X': self.X.detach().cpu().numpy(),
                'y': self.y.detach().cpu().numpy(),
                'weights': self.weights.detach().cpu().numpy(),
                'Bcp': self.detach_Bcp(),
                'bias': self.bias.detach().cpu().numpy(),
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
        self.X = params['X']
        self.y = params['y']
        self.weights = params['weights']
        self.Bcp = params['Bcp']
        self.bias = params['bias']
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
        print('X:', self.X.shape)
        print('y:', self.y.shape)
        print('weights:', self.weights)
        print('Bcp:', self.Bcp)
        print('bias', self.bias)
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
        plt.plot(self.loss_running);
        plt.xlabel('logged iteration')
        plt.ylabel('loss')
        plt.title('loss')

        logit, pred = self.predict()
        fig, axs = plt.subplots(2)
        axs[0].imshow(idx_to_oneHot(pred, self.n_classes), aspect='auto', interpolation='none')
        axs[1].imshow(idx_to_oneHot(self.y.detach().cpu().numpy(), self.n_classes), aspect='auto', interpolation='none')
        axs[1].set_xlabel('class')
        fig.suptitle('predictions')

        cm, acc = self.make_confusion_matrix(prob_or_pred='pred')
        fig = plt.figure()
        plt.imshow(cm)
        plt.ylabel('true class')
        plt.xlabel('predicted class')
        plt.title('confusion matrix (predictions)')

        Bcp_final = self.return_Bcp_final()
        fig, axs = plt.subplots(len(Bcp_final))
        for ii, val in enumerate(Bcp_final):
            axs[ii].set_title(f'factor {ii}')
            axs[ii].plot(val)
        fig.suptitle('components');