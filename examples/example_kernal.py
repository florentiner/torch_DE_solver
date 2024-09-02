import time
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model, KenelModel
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.models import FourierNN
import numpy as np 
import torch


np.random.seed(seed=3016)

solver_device('cpu')

domain = Domain()
domain.variable('x', [-1, 1], 1000)


#initial conditions
boundaries = Conditions()
boundaries.data(bnd={'x': -1}, operator=None, value=torch.tensor(np.array([[1]])))
boundaries.data(bnd={'x':  1}, operator=None, value=torch.tensor(np.array([[1]])))


equation = Equation()

eq1 = {
    'du/dx':
        {
        'coeff': 1,
        'term': [0],
        'pow': 1,
        }
}


equation.add(eq1)

net = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 2)
)

model = Model(net, domain, equation, boundaries, use_kernel=True)

model.compile("NN", lambda_operator=1, lambda_bound=10, tol=0.01)

img_dir=os.path.join(os.path.dirname( __file__ ), 'kernel')

start = time.time()

cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                    loss_window=100,
                                    no_improvement_patience=100,
                                    patience=5,
                                    randomize_parameter=1e-5,
                                    info_string_every=500)

cb_plots = plot.Plots(save_every=500, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-3})
# optimizer = Optimizer('PSO', {'pop_size': 100,
#                                   'b': 0.5,
#                                   'c2': 0.05,
#                                   'variance': 5e-2,
#                                   'c_decrease': True,
#                                   'lr': 5e-3})

model.train(optimizer, 4e4, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])

km = KenelModel(model)
import numpy as np
tr = torch.Tensor([np.random.uniform(-1, 1, 1) for _ in range(100)])
tr1 = torch.Tensor([0.1])
tr2 = torch.Tensor([0.1, 0.2])

res = km(tr)
end = time.time()