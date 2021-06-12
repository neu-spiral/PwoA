import os
from .path import *
import glob

def code_name(task, ttype, dtype, idx):
    if idx:
        filename = "{}-{}-{}-{:04d}.npy".format(task, ttype, dtype, idx)
    else:
        filename = "{}-{}-{}.npy".format(task, ttype, dtype)
    return filename

def get_log_filepath(filename, idx=None):
    filepath = "{}/assets/logs/{}".format(os.getcwd(), filename)
    return filepath

def get_model_path(filename, idx=None):
    if idx:
        filepath = "{}/assets/models/{}-{:04d}.pt".format(
            os.getcwd(), os.path.splitext(filename)[0], idx)
    else:
        filepath = "{}/assets/models/{}".format(os.getcwd(), filename)
    return filepath

def get_pr_path(config_dict):
    '''
    get path of pruning ratio
    '''
    # check if there is pre-defined config files
    if 'v' in config_dict['prune_ratio']:
        filepath = "{}/assets/configs/{}_{}.yaml".format(os.getcwd(), config_dict['model'], config_dict['prune_ratio'])
    else:
        config_dict['prune_ratio'] = float(config_dict['prune_ratio'])
        filepath = "{}/assets/configs/{}.yaml".format(os.getcwd(), config_dict['model'])
    return filepath
