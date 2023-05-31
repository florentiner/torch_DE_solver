import torch
import numpy as np
from copy import deepcopy
from typing import Tuple, Union

from tedeous.points_type import Points_type
from tedeous.derivative import Derivative
from tedeous.device import device_type, check_device
import tedeous.input_preprocessing
from tedeous.utils import *

flatten_list = lambda t: [item for sublist in t for item in sublist]


def integration(func: torch.tensor, grid, pow: Union[int, float] = 2) -> Union[
    Tuple[float, float], Tuple[list, torch.Tensor]]:
    """
    Function realize 1-space integrands,
    where func=(L(u)-f)*weak_form subintegrands function and
    definite integral parameter is grid.

    Args:
        func: operator multiplied on test function
        grid: array of a n-D points.
        pow: string (sqr ar abs) power of func points

    Returns:
        tuple(result, grid)
        'result' is integration result through one grid axis
        'grid' is initial grid without last column or zero (if grid.shape[N,1])
    """
    if grid.shape[-1] == 1:
        column = -1
    else:
        column = -2
    marker = grid[0][column]
    index = [0]
    result = []
    U = 0.
    for i in range(1, len(grid)):
        if grid[i][column] == marker or column == -1:
            U += (grid[i][-1] - grid[i - 1][-1]).item() * \
                 (func[i] ** pow + func[i - 1] ** pow) / 2
        else:
            result.append(U)
            marker = grid[i][column]
            index.append(i)
            U = 0.
    if column == -1:
        return U, 0.
    else:
        result.append(U)
        grid = grid[index, :-1]
        return result, grid


class Operator():
    def __init__(self, operator, grid, model, mode, weak_form):
        self.operator = operator
        self.grid = grid
        self.model = model
        self.mode = mode
        self.weak_form = weak_form
        if self.mode == 'NN':
            self.grid_dict = Points_type(self.grid).grid_sort()
            self.sorted_grid = torch.cat(list(self.grid_dict.values()))
        elif self.mode == 'autograd' or self.mode == 'mat':
            self.sorted_grid = self.grid

    def apply_op(self, operator, grid_points) -> torch.Tensor:
        """
        Deciphers equation in a single grid subset to a field.
        Args:
            operator: Single (len(subset)==1) operator in input form. See
            input_preprocessing.operator_prepare()
            grid_points: grid points
        Returns:
            smth
        """
        derivative = Derivative(self.model).set_strategy(
            self.mode).take_derivative

        for term in operator:
            term = operator[term]
            dif = derivative(term, grid_points)
            try:
                total += dif
            except NameError:
                total = dif
        return total

    def pde_compute(self) -> torch.Tensor:
        """
        Computes PDE residual.

        Returns:
            PDE residual.
        """
        num_of_eq = len(self.operator)
        if num_of_eq == 1:
            op = self.apply_op(
                self.operator[0], self.sorted_grid)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_op(
                    self.operator[i], self.sorted_grid))
            op = torch.cat(op_list, 1)
        return op

    def weak_pde_compute(self, weak_form) -> torch.Tensor:
        """
        Computes PDE residual in weak form.

        Args:
            weak_form: list of basis functions
        Returns:
            weak PDE residual.
        """
        device = device_type()
        if self.mode == 'NN':
            grid_central = self.grid_dict['central']
        elif self.mode == 'autograd':
            grid_central = self.grid

        op = self.pde_compute()
        sol_list = []
        for i in range(op.shape[-1]):
            sol = op[:, i]
            for func in weak_form:
                sol = sol * func(grid_central).to(device).reshape(-1)
            grid_central1 = torch.clone(grid_central)
            for k in range(grid_central.shape[-1]):
                sol, grid_central1 = integration(sol, grid_central1)
            sol_list.append(sol.reshape(-1, 1))
        if len(sol_list) == 1:
            return sol_list[0]
        else:
            return torch.cat(sol_list)

    @counter
    def operator_compute(self):
        if self.weak_form == None or self.weak_form == []:
            return self.pde_compute()
        else:
            return self.weak_pde_compute(self.weak_form)


class Bounds(Operator):
    def __init__(self, bconds, grid, model, mode, weak_form):
        super().__init__(bconds, grid, model, mode, weak_form)
        self.bconds = bconds

    def apply_bconds_set(self, operator_set: list) -> torch.Tensor:
        """
        Deciphers equation in a whole grid to a field.
        Args:
            operator_set: Multiple (len(subset)>=1) operators in input form. See
            input_preprocessing.operator_prepare().
        Returns:
            smth
        """
        field_part = []
        for operator in operator_set:
            field_part.append(self.apply_op(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def apply_dirichlet(self, bnd, var):
        if self.mode == 'NN' or self.mode == 'autograd':
            b_op_val = self.model(bnd)[:, var].reshape(-1, 1)
        elif self.mode == 'mat':
            b_op_val = []
            for position in bnd:
                if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                    b_op_val.append(self.model[:, position])
                else:
                    b_op_val.append(self.model[position])
            b_op_val = torch.cat(b_op_val).reshape(-1, 1)
        return b_op_val

    def apply_neumann(self, bnd, bop):
        if self.mode == 'NN':
            b_op_val = self.apply_bconds_set(bop)
        elif self.mode == 'autograd':
            b_op_val = self.apply_op(bop, bnd)
        elif self.mode == 'mat':
            b_op_val = self.apply_op(operator=bop, grid_points=self.grid)
            b_val = []
            for position in bnd:
                if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                    b_val.append(b_op_val[:, position])
                else:
                    b_val.append(b_op_val[position])
            b_op_val = torch.cat(b_val).reshape(-1, 1)
        return b_op_val

    def apply_periodic(self, bnd, bop, var):
        if bop is None:
            b_op_val = self.apply_dirichlet(bnd[0], var).reshape(-1, 1)
            for i in range(1, len(bnd)):
                b_op_val -= self.apply_dirichlet(bnd[i], var).reshape(-1, 1)
        else:
            if self.mode == 'NN':
                b_op_val = self.apply_neumann(bnd, bop[0]).reshape(-1, 1)
                for i in range(1, len(bop)):
                    b_op_val -= self.apply_neumann(bnd, bop[i]).reshape(-1, 1)
            elif self.mode == 'autograd' or self.mode == 'mat':
                b_op_val = self.apply_neumann(bnd[0], bop).reshape(-1, 1)
                for i in range(1, len(bnd)):
                    b_op_val -= self.apply_neumann(bnd[i], bop).reshape(-1, 1)
        return b_op_val

    def b_op_val_calc(self, bcond) -> torch.Tensor:
        """
        Auxiliary function. Serves only to evaluate operator on the boundary.
        Args:
            bcond:  terms of prepared boundary conditions (see input_preprocessing.bnd_prepare) in input form.
        Returns:
            calculated operator on the boundary.
        """

        if bcond['type'] == 'dirichlet':
            b_op_val = self.apply_dirichlet(bcond['bnd'], bcond['var'])
        elif bcond['type'] == 'operator':
            b_op_val = self.apply_neumann(bcond['bnd'], bcond['bop'])
        elif bcond['type'] == 'periodic':
            b_op_val = self.apply_periodic(bcond['bnd'], bcond['bop'],
                                           bcond['var'])
        return b_op_val

    def compute_bconds(self, bcond):
        truebval = bcond['bval'].reshape(-1, 1)
        b_op_val = self.b_op_val_calc(bcond)
        return b_op_val, truebval

    def bcs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Auxiliary function. Serves only to evaluate boundary values and true boundary values.
        Returns:
            * **b_val** -- calculated model boundary values.\n
            * **true_b_val** -- true grid boundary values.
        """

        true_b_val_list = []
        b_val_list = []

        for bcond in self.bconds:
            b_op_val, truebval = self.compute_bconds(bcond)
            b_val_list.append(b_op_val)
            true_b_val_list.append(truebval)

        true_b_val = torch.cat(true_b_val_list)
        b_val = torch.cat(b_val_list).reshape(-1, 1)
        return b_val, true_b_val

    def bcs_ics(self):
        true_ics_list = []
        ics_list = []

        bcs_list = []
        true_bcs_list = []

        for bcond in self.bconds:
            if bcond['condition'] == 'initial':
                ics_pred, true_ics = self.compute_bconds(bcond)
                ics_list.append(ics_pred)
                true_ics_list.append(true_ics)
            else:
                bcs_pred, true_bcs = self.compute_bconds(bcond)
                bcs_list.append(bcs_pred)
                true_bcs_list.append(true_bcs)

        true_bcs = torch.cat(true_bcs_list)
        bcs_pred = torch.cat(bcs_list)

        true_ics = torch.cat(true_ics_list)
        ics_pred = torch.cat(ics_list)

        b_val = [bcs_pred, ics_pred]
        true_b_val = [true_bcs, true_ics]
        return b_val, true_b_val

    def apply_bnd(self, num_of_terms):
        if num_of_terms == 3:
            return self.bcs_ics()
        else:
            return self.bcs()


class Loss():
    def __init__(self, operator, bval, true_bval, lambda_op, lambda_bcs, lambda_ics, mode, weak_form, num_of_loss_term):
        self.operator = operator
        self.mode = mode
        self.weak_form = weak_form

        if num_of_loss_term == 2:
            self.lambda_op = 1
            self.lambda_bound = lambda_bcs
            self.lambda_initial = 0
            self.bval = [bval, torch.zeros_like(bval)]
            self.true_bval = [true_bval, torch.zeros_like(true_bval)]

        elif num_of_loss_term == 3:
            self.lambda_op = lambda_op
            self.lambda_bound = lambda_bcs
            self.lambda_initial = lambda_ics
            self.bval = bval
            self.true_bval = true_bval

    def l2_loss(self) -> torch.Tensor:
        """
        Computes l2 loss.
        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
        Returns:
            model loss.
        """
        if self.bval == None:
            return torch.sum(torch.mean((self.operator) ** 2, 0))

        if self.mode == 'mat':
            loss = torch.mean((self.operator) ** 2) + \
                   self.lambda_bound * torch.mean((self.bval - self.true_bval) ** 2)
        else:
            loss = self.lambda_op * torch.sum(torch.mean((self.operator) ** 2), 0) + \
                   self.lambda_bound * torch.sum(torch.mean((self.bval[0] - self.true_bval[0]) ** 2, 0)) + \
                   self.lambda_initial * torch.sum(torch.mean((self.bval[1] - self.true_bval[1]) ** 2, 0))
        return loss

    def weak_loss(self) -> torch.Tensor:
        """
        Weak solution of O/PDE problem.
        Args:
            weak_form: list of basis functions.
            lambda_bound: const regularization parameter.
        Returns:
            model loss.
        """
        if self.bval == None:
            return torch.sum(self.operator)

        # we apply no  boundary conditions operators if they are all None

        loss = self.lambda_op * torch.sum(self.operator) + \
               self.lambda_bound * torch.sum(torch.mean((self.bval - self.true_bval) ** 2, 0)) + \
               self.lambda_initial * torch.sum(torch.mean((self.bval[1] - self.true_bval[1]) ** 2, 0))
        return loss

    def compute(self) -> Union[l2_loss, weak_loss]:
        """
        Setting the required loss calculation method.
        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
            weak_form: list of basis functions.
        Returns:
            A given calculation method.
        """

        if self.mode == 'mat' or self.mode == 'autograd':
            if self.bval == None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf

        if self.weak_form == None or self.weak_form == []:
            return self.l2_loss()
        else:
            return self.weak_loss()


class Solution():
    def __init__(self, grid: torch.Tensor, equal_cls: Union[tedeous.input_preprocessing.Equation_NN,
                                                            tedeous.input_preprocessing.Equation_mat, tedeous.input_preprocessing.Equation_autograd],
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str, weak_form, update_every_lambdas,
                 loss_term, lambda_op,lambda_bcs, lambda_ics):

        grid = check_device(grid)
        equal_copy = deepcopy(equal_cls)
        prepared_operator = equal_copy.operator_prepare()
        prepared_bconds = equal_copy.bnd_prepare()

        self.model = model.to(device_type())
        self.mode = mode
        self.update_every_lambdas = update_every_lambdas
        self.weak_form = weak_form
        self.loss_term = loss_term

        self.lambda_op = lambda_op
        self.lambda_bcs = lambda_bcs
        self.lambda_ics = lambda_ics

        self.operator = Operator(prepared_operator, grid, self.model, self.mode, self.weak_form)
        self.boundary = Bounds(prepared_bconds, grid, self.model, self.mode, self.weak_form)

    def evaluate(self, iter):
        op = self.operator.operator_compute()
        bval, true_bval = self.boundary.apply_bnd(num_of_terms=self.loss_term)

        if self.update_every_lambdas is not None and iter % self.update_every_lambdas == 0:
            self.lambda_bcs, self.lambda_ics, self.lambda_op = LambdaCompute(bval, true_bval, op, self.model).update()

        loss = Loss(operator=op, bval=bval, true_bval=true_bval, mode=self.mode,
                    weak_form=self.weak_form, lambda_op=self.lambda_op, lambda_bcs=self.lambda_bcs, lambda_ics=self.lambda_ics,
                    num_of_loss_term=self.loss_term)

        return loss.compute()
