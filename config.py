# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:13:37 2022

@author: Sashka
"""

import json

def read_config(name):
    with open(name, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data


DEFAULT_CONFIG="""
{
"Optimizer": {
"learning_rate":1e-4,
"lambda_bound":10,
"optimizer":"Adam"
},
"Cache":{
"use_cache":true,
"cache_dir":"../cache/",
"cache_verbose":false,
"save_always":false,
"model_randomize_parameter":0
},
"NN":{
"batch_size":null,
"lp_par":null,
"grid_point_subset":["central"],
"h":0.001
},
"Verbose":{
	"verbose":false,
	"print_every":null
},
"StopCriterion":{
"eps":1e-5,
"tmin":1000,
"tmax":1e5 ,
"patience":5,
"loss_oscillation_window":100,
"no_improvement_patience":1000   	
},
"Matrix":{
"lp_par":null,
"cache_model":null
}
}
"""

default_config = json.loads(DEFAULT_CONFIG)

def check_module_name(module_name:str) -> bool:
    """
    check correctness of the 'first' level of config parameter name
    we call it module

    Parameters
    ----------
    module_name: str
        
        first level of a parameter of a custom config

    Returns
    -------
    module_correctness : bool

        true if module presents in 'default' config

    """
    if module_name in default_config.keys():
        return True
    else:
        return False
    


def check_param_name(module_name:str, param_name:str) -> bool:
    """
    check correctness of the 'first' level of config parameter name
    we call it module

    Parameters
    ----------
    module_name: str
        
        first level of a parameter of a custom config

    Returns
    -------
    module_correctness : bool

        true if module presents in 'default' config

    """
    if param_name in default_config[module_name].keys():
        return True
    else:
        return False

'''
We can use old json load. However, it is good to check if the parameters named correctly
So, we make the full 'default' version of the config and load 'non-default' parameters
either from file or using method 'change_parameter'
'''
class Config:
  def __init__(self, *args):
    '''
    We init config with default one

    If there is passed path to a custon config, we try to load it and change
    default parameters


    Parameters
    ----------
    config_path: str, optional
        
        path to a custom config

    Returns
    -------
    self: Config 

        config used in solver.optimization_solver function
    '''



    self.params=default_config
    if len(args)==1:
        custom_config=read_config(args[0])
        for module_name in custom_config.keys():
            if check_module_name(module_name) and check_param_name(module_name,custom_config[module_name].keys()):
                for param in custom_config[module_name].keys():
                    self.params[module_name][param]=custom_config[module_name][param]
    else:
        print('Too much initialization args, using default config')



