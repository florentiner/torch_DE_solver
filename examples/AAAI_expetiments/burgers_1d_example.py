
import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time
from scipy.integrate import quad

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('AAAI_expetiments'))))


from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.models import mat_model, Fourier_embedding
from tedeous.callbacks import plot, early_stopping, adaptive_lambda
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.eval import integration



import pandas as pd

solver_device('cuda')

mu = 0.01 / np.pi



def u(grid):
    
    def f(y):
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * mu))

    def integrand1(m, x, t):
        return np.sin(np.pi * (x - m)) * f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def integrand2(m, x, t):
        return f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def u(x, t):
        if t == 0:
            return -np.sin(np.pi * x)
        else:
            return -quad(integrand1, -np.inf, np.inf, args=(x, t))[0] / quad(integrand2, -np.inf, np.inf, args=(x, t))[
                0]

    solution = []
    for point in grid:
        solution.append(u(point[0].item(), point[1].item()))

    return torch.tensor(solution)


def u_net(net, x):
    net = net.to('cpu')
    x = x.to('cpu')
    return net(x).detach()


def l2_norm(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()

def l2_norm_mat(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net.detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()

def l2_norm_fourier(net, x):
    x = x.to(torch.device('cuda:0'))
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()




def burgers1d_problem_formulation(grid_res):
    
    domain = Domain()
    domain.variable('x', [-1, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)

    boundaries = Conditions()
    x = domain.variable_dict['x']
    boundaries.dirichlet({'x': [-1, 1], 't': 0}, value=-torch.sin(np.pi * x))

    boundaries.dirichlet({'x': -1, 't': [0, 1]}, value=0)

    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=0)

    equation = Equation()

    burgers_eq = {
        'du/dt**1':
            {
                'coeff': 1.,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '+u*du/dx':
            {
                'coeff': 1,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(burgers_eq)

    grid = domain.build('autograd')

    return grid,domain,equation,boundaries




def experiment_data_amount_burgers_1d_PINN(grid_res,exp_name='burgers1d_PINN',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = burgers1d_problem_formulation(grid_res)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1/2, lambda_bound=1/2)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=100,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 2e5, callbacks=[cb_es])
    end = time.time()

    run_time = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f, _ = integration(lu_f, gr)


    exp_dict={'grid_res': grid_res,
                          'error_train': error_train.item(),
                          'error_test': error_test.item(),
                          'loss': loss.item(),
                          "lu_f_adam": lu_f.item(),
                          'time_adam': run_time,
                          'type': exp_name}

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_test))

    exp_dict_list.append(exp_dict)

    #if save_plot:
    #    if not os.path.isdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name)):
    #        os.mkdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name))
    #    plt.figure()
    #    plt.plot(grid.detach().cpu().numpy(), u(grid).detach().cpu().numpy(), label='Exact')
    #    plt.plot(grid.detach().cpu().numpy(), net(grid.cpu()).detach().cpu().numpy(), '--', label='Predicted')
    #    plt.xlabel('x')
    #    plt.ylabel('y')
    #    plt.legend(loc='upper right')
    #    plt.savefig('examples\\AAAI_expetiments\\figures_{}\\img_{}.png'.format(exp_name,N))

    return exp_dict_list


def experiment_data_amount_burgers_1d_PSO(grid_res,exp_name='burgers1d_PSO',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = burgers1d_problem_formulation(grid_res)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1/2, lambda_bound=1/2)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=100,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 2e5, callbacks=[cb_es])
    end = time.time()

    time_adam = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    error_adam_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_adam_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss_adam = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f_adam, _ = integration(lu_f, gr)

    ########

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=100,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    optim = Optimizer('PSO', {'pop_size': 50, #30
                                  'b': 0.4, #0.5
                                  'c2': 0.5, #0.05
                                  'c1': 0.5, 
                                  'variance': 5e-2,
                                  'lr': 1e-4})
    start = time.time()
    model.train(optim, 2e4, save_model=False, callbacks=[cb_es])
    end = time.time()
    time_pso = end - start

    error_pso_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_pso_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss_pso = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    grid = domain.build('autograd')

    lu_f, gr = integration(lu_f, grid)

    lu_f_pso, _ = integration(lu_f, gr)

    #########

    

    exp_dict={'grid_res': grid_res,
                          'error_adam_train': error_adam_train.item(),
                          'error_adam_test': error_adam_test.item(),
                          'error_PSO_train': error_pso_train.item(),
                          'error_PSO_test': error_pso_test.item(),
                          'loss_adam': loss_adam.item(),
                          'loss_pso': loss_pso.item(),
                          "lu_f_adam": lu_f_adam.item(),
                          "lu_f_pso": lu_f_pso.item(),
                          'time_adam': time_adam,
                          'time_pso': time_pso,
                          'type': exp_name}

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE_adam {}= {}'.format(grid_res, error_adam_test))
    print('RMSE_pso {}= {}'.format(grid_res, error_pso_test))

    exp_dict_list.append(exp_dict)

    return exp_dict_list




def experiment_data_amount_burgers_1d_lam(grid_res,exp_name='burgers1d_lam',save_plot=True):

    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = burgers1d_problem_formulation(grid_res)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1/2, lambda_bound=1/2)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=100,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    cb_lam = adaptive_lambda.AdaptiveLambda()

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 2e5, callbacks=[cb_es,cb_lam])
    end = time.time()

    run_time = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f, _ = integration(lu_f, gr)


    exp_dict={'grid_res': grid_res,
                          'error_train': error_train.item(),
                          'error_test': error_test.item(),
                          'loss': loss.item(),
                          "lu_f_adam": lu_f.item(),
                          'time_adam': run_time,
                          'type': exp_name}

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_test))

    exp_dict_list.append(exp_dict)

    return exp_dict_list




def experiment_data_amount_burgers_1d_fourier(grid_res,exp_name='burgers1d_fourier',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = burgers1d_problem_formulation(grid_res)

    FFL = Fourier_embedding(L=[2,2], M=[1,1])

    out = FFL.out_features


    net = torch.nn.Sequential(
        FFL,
        torch.nn.Linear(out, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1/2, lambda_bound=1/2)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=100,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 2e5, callbacks=[cb_es])
    end = time.time()

    run_time = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f, _ = integration(lu_f, gr)


    exp_dict={'grid_res': grid_res,
                          'error_train': error_train.item(),
                          'error_test': error_test.item(),
                          'loss': loss.item(),
                          "lu_f_adam": lu_f.item(),
                          'time_adam': run_time,
                          'type': exp_name}

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_test))


    exp_dict_list.append(exp_dict)

    return exp_dict_list



def replace_none_with_zero(tuple_data):
    if isinstance(tuple_data, torch.Tensor):
        tuple_data[tuple_data == None] = 0
    elif tuple_data is None:
        tuple_data = torch.tensor([0.])
    elif isinstance(tuple_data, tuple):
        new_tuple = tuple(replace_none_with_zero(item) for item in tuple_data)
        return new_tuple
    return tuple_data

def gramian(net, residuals):
        # Compute the jacobian on batched data
    def jacobian():
        jac = []
        loss = residuals
        for l in loss:
            j = torch.autograd.grad(l, net.parameters(), retain_graph=True, allow_unused=True)
            j = replace_none_with_zero(j)
            j = parameters_to_vector(j).reshape(1, -1)
            jac.append(j)
        return torch.cat(jac)

    J = jacobian()
    return 1.0 / len(residuals) * J.T @ J

def grid_line_search_factory(loss, steps):

    def loss_at_step(step, model, tangent_params):
        params = parameters_to_vector(model.parameters())
        new_params = params - step*tangent_params
        vector_to_parameters(new_params, model.parameters())
        loss_val, _ = loss()
        vector_to_parameters(params, model.parameters())
        return loss_val

    def grid_line_search_update(model, tangent_params):

        losses = []
        for step in steps:
            losses.append(loss_at_step(step, model, tangent_params).reshape(1))
        losses = torch.cat(losses)
        step_size = steps[torch.argmin(losses)]

        params = parameters_to_vector(model.parameters())
        new_params = params - step_size*tangent_params
        vector_to_parameters(new_params, model.parameters())

        return step_size

    return grid_line_search_update


def experiment_data_amount_burgers_1d_NGD(grid_res,NGD_info_string=True,exp_name='burgers1d_NGD',save_plot=True):

    exp_dict_list = []

    
    l_op = 1
    l_bound = 1
    grid_steps = torch.linspace(0, 30, 31)
    steps = 0.5**grid_steps

    grid,domain,equation,boundaries = burgers1d_problem_formulation(grid_res)
    
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=l_op, lambda_bound=l_bound)


    ls_update = grid_line_search_factory(model.solution_cls.evaluate, steps)

    loss, _ = model.solution_cls.evaluate()


    start = time.time()

    iteration=0
    #for iteration in range(100):
    while loss.item()>1e-6:
        
        loss, _ = model.solution_cls.evaluate()
        grads = torch.autograd.grad(loss, model.net.parameters(), retain_graph=True, allow_unused=True)
        grads = replace_none_with_zero(grads)
        f_grads = parameters_to_vector(grads)

        int_res = model.solution_cls.operator._pde_compute()
        bval, true_bval, _, _ = model.solution_cls.boundary.apply_bcs()
        bound_res = bval-true_bval

        # assemble gramian
        G_int  = gramian(model.net, int_res)

        G_bdry = gramian(model.net, bound_res)
        G      = G_int + G_bdry

        # Marquardt-Levenberg
        Id = torch.eye(len(G))
        G = torch.min(torch.tensor([loss, 0.0])) * Id + G
        # compute natural gradient

        #G = G.detach().cpu().numpy()
        #f_grads =f_grads.detach().cpu().numpy()
        #f_nat_grad = np.linalg.lstsq(G, f_grads)[0] 
        #f_nat_grad = torch.from_numpy(np.array(f_nat_grad)).to(torch.float64).to("cuda:0")


        G = G.detach().cpu()
        f_grads =f_grads.detach().cpu()
        f_nat_grad=torch.linalg.lstsq(G, f_grads,driver='gelsd')[0] 
        
        f_nat_grad = f_nat_grad.to("cuda:0")

        

        # one step of NGD
        actual_step = ls_update(model.net, f_nat_grad)
        iteration+=1
        if iteration>1000:
            break
        #if iteration%10 == 0 and NGD_info_string:
        ##if NGD_info_string:
        #    print('iteration= ', iteration)
        #    print('step= ', actual_step.item())
        #    print('loss=' , model.solution_cls.evaluate()[0].item())
    if NGD_info_string:
        print('iteration= ', iteration)
        print('step= ', actual_step.item())
        print('loss=' , model.solution_cls.evaluate()[0].item())
    end = time.time()
    run_time = end-start
    
    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f, _ = integration(lu_f, gr)


    exp_dict={'grid_res': grid_res,
                          'error_train': error_train.item(),
                          'error_test': error_test.item(),
                          'loss': loss.item(),
                          "lu_f_adam": lu_f.item(),
                          'time_adam': run_time,
                          'type': exp_name}

    exp_dict_list.append(exp_dict)



    print('grid_res=', grid_res)
    print('l2_norm = ', error_train.item())
    print('lu_f = ', lu_f.item())


    return exp_dict_list




if __name__ == '__main__':

    if not os.path.isdir('examples\\AAAI_expetiments\\results'):
        os.mkdir('examples\\AAAI_expetiments\\results')
    

    exp_dict_list=[]

    nruns = 1

    #for grid_res in range(10, 101, 10):
    #    for _ in range(nruns):
    #        exp_dict_list.append(experiment_data_amount_poisson_1d_PINN(N))

    

    #exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    #df = pd.DataFrame(exp_dict_list_flatten)
    #df.to_csv('examples\\AAAI_expetiments\\results\\burgers_PINN.csv')

    #exp_dict_list=[]


    #for grid_res in range(10, 101, 10):
    #    for _ in range(nruns):
    #        exp_dict_list.append(experiment_data_amount_burgers_1d_PSO(grid_res))

    

    #exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    #df = pd.DataFrame(exp_dict_list_flatten)
    #df.to_csv('examples\\AAAI_expetiments\\results\\burgers_PSO.csv')


    exp_dict_list=[]

    for grid_res in range(10, 101, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_burgers_1d_lam(grid_res))


    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\burgers_lam.csv')

    exp_dict_list=[]

    for grid_res in range(10, 101, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_burgers_1d_fourier(grid_res))

    

    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\burgers_fourier.csv')

    exp_dict_list=[]

    for grid_res in range(10, 101, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_burgers_1d_NGD(grid_res,NGD_info_string=True))

  
    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\burgers_NGD.csv')
