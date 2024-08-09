"""Module for connecting *eval.py*, *losses.py*"""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple, Union, Any
import torch

from tedeous.derivative import Derivative
from tedeous.points_type import Points_type
from tedeous.eval import Operator, Bounds
from tedeous.losses import Losses
from tedeous.device import device_type, check_device
from tedeous.input_preprocessing import lambda_prepare, Equation_NN, Equation_mat, Equation_autograd


flatten_list = lambda t: [item for sublist in t for item in sublist]

class Solution():
    """
    class for different loss functions calculation.
    """
    def __init__(
        self,
        grid: torch.Tensor,
        equal_cls: Union[Equation_NN, Equation_mat, Equation_autograd],
        model: Union[torch.nn.Sequential, torch.Tensor],
        mode: str,
        weak_form: Union[None, list[callable]],
        lambda_operator,
        lambda_bound,
        tol: float = 0,
        derivative_points: int = 2,
        batch_size: int = None):
        """
        Args:
            grid (torch.Tensor): discretization of comp-l domain.
            equal_cls (Union[Equation_NN, Equation_mat, Equation_autograd]): Equation_{NN, mat, autograd} object.
            model (Union[torch.nn.Sequential, torch.Tensor]): model of *mat or NN or autograd* mode.
            mode (str): *mat or NN or autograd*
            weak_form (Union[None, list[callable]]): list with basis functions, if the form is *weak*.
            lambda_operator (_type_): regularization parameter for operator term in loss.
            lambda_bound (_type_): regularization parameter for boundary term in loss.
            tol (float, optional): penalty in *casual loss*. Defaults to 0.
            derivative_points (int, optional): points number for derivative calculation.
            batch_size (int): size of batch.
            For details to Derivative_mat class.. Defaults to 2.
        """

        self.grid = check_device(grid)
        if mode == 'NN':
            sorted_grid = Points_type(self.grid).grid_sort()
            self.n_t = len(sorted_grid['central'][:, 0].unique())
            self.n_t_operation = lambda sorted_grid: len(sorted_grid['central'][:, 0].unique())
        elif mode == 'autograd':
            self.n_t = len(self.grid[:, 0].unique())
            self.n_t_operation = lambda grid: len(grid[:, 0].unique())
        elif mode == 'mat':
            self.n_t = grid.shape[1]
            self.n_t_operation = lambda grid: grid.shape[1]
        
        equal_copy = deepcopy(equal_cls)
        prepared_operator = equal_copy.operator_prepare()
        self._operator_coeff(equal_cls, prepared_operator)
        self.prepared_bconds = equal_copy.bnd_prepare()
        self.model = model.to(device_type())
        self.mode = mode
        self.weak_form = weak_form
        self.lambda_operator = lambda_operator
        self.lambda_bound = lambda_bound
        self.tol = tol
        self.derivative_points = derivative_points
        self.batch_size = batch_size
        if self.batch_size is None:
            self.n_t_operation = None
        

        self.operator = Operator(self.grid, prepared_operator, self.model,
                                   self.mode, weak_form, derivative_points, 
                                   self.batch_size)
        self.boundary = Bounds(self.grid,self.prepared_bconds, self.model,
                                   self.mode, weak_form, derivative_points)

        self.loss_cls = Losses(self.mode, self.weak_form, self.n_t, self.tol, 
                               self.n_t_operation) # n_t calculate for each batch 
        self.op_list = []
        self.bval_list = []
        self.loss_list = []

    @staticmethod
    def _operator_coeff(equal_cls: Any, operator: list):
        """ Coefficient checking in operator.

        Args:
            equal_cls (Any): Equation_{NN, mat, autograd} object.
            operator (list): prepared operator (result of operator_prepare())
        """
        for i, _ in enumerate(equal_cls.operator):
            eq = equal_cls.operator[i]
            for key in eq.keys():
                if isinstance(eq[key]['coeff'], torch.nn.Parameter):
                    try:
                        operator[i][key]['coeff'] = eq[key]['coeff'].to(device_type())
                    except:
                        operator[key]['coeff'] = eq[key]['coeff'].to(device_type())
                elif isinstance(eq[key]['coeff'], torch.Tensor):
                    eq[key]['coeff'] = eq[key]['coeff'].to(device_type())

    def _model_change(self, new_model: torch.nn.Module) -> None:
        """Change self.model for class and *operator, boundary* object.
            It should be used in cache_lookup and cache_retrain method.

        Args:
            new_model (torch.nn.Module): new self model.
        """
        self.model = new_model
        self.operator.model = new_model
        self.operator.derivative = Derivative(new_model, self.derivative_points).set_strategy(
            self.mode).take_derivative
        self.boundary.model = new_model
        self.boundary.operator = Operator(self.grid,
                                          self.prepared_bconds,
                                          new_model,
                                          self.mode,
                                          self.weak_form,
                                          self.derivative_points,
                                          self.batch_size)

    def evaluate(self,
                 save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes loss.

        Args:
            second_order_interactions (bool, optional): optimizer iteration
            (serves only for computing adaptive lambdas). Defaults to True.
            sampling_N (int, optional): parameter for accumulation of
            solutions (op, bcs). The more sampling_N, the more accurate the
            estimation of the variance (only for computing adaptive lambdas). Defaults to 1.
            lambda_update (bool, optional): update lambda or not. Defaults to False.
            save_graph (bool, optional): responsible for saving the computational graph. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: loss
        """
        self.op = self.operator.operator_compute()
        self.bval, self.true_bval,\
            self.bval_keys, self.bval_length = self.boundary.apply_bcs()
        dtype = self.op.dtype
        self.lambda_operator = lambda_prepare(self.op, self.lambda_operator).to(dtype)
        self.lambda_bound = lambda_prepare(self.bval, self.lambda_bound).to(dtype)

        self.loss, self.loss_normalized = self.loss_cls.compute(
            self.op,
            self.bval,
            self.true_bval,
            self.lambda_operator,
            self.lambda_bound,
            save_graph)
        if self.batch_size is not None: 
            if self.operator.new_init: # if first batch in epoch
                self.save_op = self.op
                self.operator.new_init = False
            else:
                self.save_op = torch.cat((self.save_op, self.operator.operator_compute()), 0) # cat curent losses to previous
            del self.op
            torch.cuda.empty_cache()

        return self.loss, self.loss_normalized
