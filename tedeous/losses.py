"""Module for losses calculation"""

from typing import Tuple, Union
import numpy as np
import torch

from tedeous.input_preprocessing import lambda_prepare
from tedeous.derivative import Derivative_autograd, Derivative
from tedeous.eval import Operator
import torch.nn as nn
from sklearn.kernel_ridge import KernelRidge

def svd_lstsq(AA, BB, tol=1e-5):
        U, S, Vh = torch.linalg.svd(AA, full_matrices=False)
        Spinv = torch.zeros_like(S)
        Spinv[S>tol] = 1/S[S>tol]
        UhBB = U.adjoint() @ BB
        if Spinv.ndim!=UhBB.ndim:
            Spinv = Spinv.unsqueeze(-1)
        SpinvUhBB = Spinv * UhBB
        return Vh.adjoint() @ SpinvUhBB

class KernelRidge_(nn.Module):
    """
    Kernel Ridge Regression model implemented in PyTorch with fit and predict methods.

    Args:
        kernel (str): The kernel function to use. Available options:
            - 'linear': Linear kernel
            - 'rbf': Radial basis function (Gaussian) kernel
        gamma (float, optional): Kernel parameter for RBF kernel. Defaults to None.
        alpha (float): Regularization parameter.
        device (str, optional): Device to use for computations. Defaults to 'cpu'.
    """
    def __init__(self, alpha=1e-3):
        super(KernelRidge_, self).__init__()
        self.alpha = alpha
        self.coeffs = None  # Store the learned coefficients

    def forward(self, X):
        """
        Computes the Kernel Ridge Regression prediction.

        Args:
            X (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predictions.
        """
        if self.coeffs is None:
            raise ValueError("Model must be fitted first.")
        

        # Predict using the learned coefficients and the kernel
        predictions = X @ self.coeffs
        # predictions = X @ self.coeffs

        return predictions

    def fit(self, X, y):
        """
        Fits the Kernel Ridge Regression model to the data.

        Args:
            X (torch.Tensor): Input features.
            y (torch.Tensor): Target values.
        """

        # Calculate the kernel matrix
        K = X

        # Add regularization term to the kernel matrix
        K_reg = K + self.alpha * torch.eye(K.shape[0])

        # Solve for the coefficients using the closed-form solution
        self.coeffs = svd_lstsq(K_reg, y)

    def predict(self, X):
        """
        Makes predictions on new data.

        Args:
            X (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predictions.
        """
        return self.forward(X)

class Losses():
    """
    Class which contains all losses.
    """
    def __init__(self,
                 mode: str,
                 weak_form: Union[None, list],
                 n_t: int,
                 tol: Union[int, float],
                 model: Union[torch.nn.Sequential, torch.Tensor],
                 grid: torch.Tensor,
                 use_kernel: bool):
        """
        Args:
            mode (str): calculation mode, *NN, autograd, mat*.
            weak_form (Union[None, list]): list of basis functions if form is weak.
            n_t (int): number of unique points in time dinension.
            tol (Union[int, float])): penalty in *casual loss*.
        """

        self.mode = mode
        self.weak_form = weak_form
        self.n_t = n_t
        self.tol = tol
        self.model = model
        self.grad = Derivative_autograd(self.model)
        self.grid = grid
        self.kernel_ready = False
        self.use_kernel = use_kernel
        # TODO: refactor loss_op, loss_bcs into one function, carefully figure out when bval
        # is None + fix causal_loss operator crutch (line 76).

    def _loss_op(self,
                operator: torch.Tensor,
                lambda_op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Operator term in loss calc-n.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().

            lambda_op (torch.Tensor): regularization parameter for operator term in loss.

        Returns:
            loss_operator (torch.Tensor): operator term in loss.
            op (torch.Tensor): MSE of operator on the whole grid.
        """
        if self.weak_form is not None and self.weak_form != []:
            op = operator
        else:
            op = torch.mean(operator**2, 0)

        loss_operator = op @ lambda_op.T
        return loss_operator, op


    def _loss_bcs(self,
                 bval: torch.Tensor,
                 true_bval: torch.Tensor,
                 lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes boundary loss for corresponding type.

        Args:
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss_bnd (torch.Tensor): boundary term in loss.
            bval_diff (torch.Tensor): MSE of all boundary con-s.
        """

        bval_diff = torch.mean((bval - true_bval)**2, 0)

        loss_bnd = bval_diff @ lambda_bound.T
        return loss_bnd, bval_diff


    def _default_loss(self,
                     operator: torch.Tensor,
                     bval: torch.Tensor,
                     true_bval: torch.Tensor,
                     lambda_op: torch.Tensor,
                     lambda_bound: torch.Tensor,
                     save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute l2 loss.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
            save_graph (bool, optional): saving computational graph. Defaults to True.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """

        if bval is None:
            return torch.sum(torch.mean((operator) ** 2, 0))

        loss_oper, op = self._loss_op(operator, lambda_op)
        dtype = op.dtype
        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1).to(dtype)
        lambda_bound_normalized = lambda_prepare(bval, 1).to(dtype)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T +\
                        bval_diff @ lambda_bound_normalized.T

        # TODO make decorator and apply it for all losses.
        if not save_graph:
            temp_loss = loss.detach()
            del loss
            torch.cuda.empty_cache()
            loss = temp_loss
        
        return loss, loss_normalized

    def _causal_loss(self,
                    operator: torch.Tensor,
                    bval: torch.Tensor,
                    true_bval: torch.Tensor,
                    lambda_op: torch.Tensor,
                    lambda_bound: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes causal loss, which is calculated with weights matrix:
        W = exp(-tol*(Loss_i)) where Loss_i is sum of the L2 loss from 0
        to t_i moment of time. This loss function should be used when one
        of the DE independent parameter is time.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """

        res = torch.sum(operator**2, dim=1).reshape(self.n_t, -1)
        res = torch.mean(res, axis=1).reshape(self.n_t, 1)
        m = torch.triu(torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1).T
        with torch.no_grad():
            w = torch.exp(- self.tol * (m @ res))
        loss_oper = torch.mean(w * res)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)

        loss = loss_oper + loss_bnd

        lambda_bound_normalized = lambda_prepare(bval, 1)
        with torch.no_grad():
            loss_normalized = loss_oper +\
                        lambda_bound_normalized @ bval_diff

        return loss, loss_normalized

    def _weak_loss(self,
                  operator: torch.Tensor,
                  bval: torch.Tensor,
                  true_bval: torch.Tensor,
                  lambda_op: torch.Tensor,
                  lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Weak solution of O/PDE problem.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """

        if bval is None:
            return sum(operator)

        loss_oper, op = self._loss_op(operator, lambda_op)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1)
        lambda_bound_normalized = lambda_prepare(bval, 1)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T +\
                        bval_diff @ lambda_bound_normalized.T
        
        return loss, loss_normalized
        
    def _kernel_loss_(self,
                  operator: torch.Tensor,
                  bval: torch.Tensor,
                  true_bval: torch.Tensor,
                  lambda_op: torch.Tensor,
                  lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Weak solution of O/PDE problem.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """
        f_x = lambda points: self.model(points)#self.grid
        d_dx_2_f_x = lambda points: self.grad._nn_autograd(
                    self.model, points, 0, [0,0]) #self.grid
        lambda_n, mu_n = 1, 1
        self.kernel_function = lambda points: lambda_n * d_dx_2_f_x(points) + mu_n * d_dx_2_f_x(points) + lambda_n * f_x(points)
        kernel_function_x = self.kernel_function(self.grid)
        kernel_function_x_orig_size = kernel_function_x.size()
        kernel_function_x = kernel_function_x.reshape(-1)
        kernel_x_matrix = torch.tile(kernel_function_x, (len(kernel_function_x), 1))
        operator_orig_size = operator.size()
        operator = operator.reshape(-1, 1)
        with torch.no_grad():
            if not self.kernel_ready:
                kernel_model = KernelRidge_(alpha=len(operator))
                kernel_model.fit(kernel_x_matrix, operator)
                self.kernel_ready = True
                self.kernel_model = kernel_model
        
        kernel_res = self.kernel_model.predict(kernel_x_matrix)
        kernel_res = kernel_res.reshape(kernel_function_x_orig_size)
        operator = operator.reshape(operator_orig_size)
        operator_diff = kernel_res - operator
        operator_diff = operator_diff * lambda_op
        loss_oper = torch.norm(operator_diff, p=2)
        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1)
        lambda_bound_normalized = lambda_prepare(bval, 1)
        with torch.no_grad():
            loss_normalized = torch.mean(operator_diff**2, 0) @ lambda_op_normalized.T +\
                        bval_diff @ lambda_bound_normalized.T
        dif = torch.mean(abs(kernel_res - 1))
        print(dif)

        return loss, loss_normalized
        

    def compute(self,
                operator: torch.Tensor,
                bval: torch.Tensor,
                true_bval: torch.Tensor,
                lambda_op: torch.Tensor,
                lambda_bound: torch.Tensor,
                save_graph: bool = True) -> Union[_default_loss, _weak_loss, _causal_loss]:
        """ Setting the required loss calculation method.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
            save_graph (bool, optional): saving computational graph. Defaults to True.

        Returns:
            Union[default_loss, weak_loss, causal_loss]: A given calculation method.
        """

        if self.mode in ('mat', 'autograd'):
            if bval is None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf
        inputs = [operator, bval, true_bval, lambda_op, lambda_bound]
        if self.use_kernel:
            return self._kernel_loss_(*inputs)
        if self.weak_form is not None and self.weak_form != []:
            return self._weak_loss(*inputs)
        elif self.tol != 0:
            return self._causal_loss(*inputs)
        else:
            return self._default_loss(*inputs, save_graph)
