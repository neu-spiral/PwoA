from .. import *
from .misc  import *
from .path  import *
import yaml

def load_yaml(filepath):

    with open(filepath, 'r') as stream:
        try:
            data = yaml.load(stream, yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return data
    
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    
def load_model(filepath):
    model = torch.load(filepath)
    return model

def load_state_dict(model, filepath):
    device = next(model.parameters()).device
    own_state = model.state_dict()
    
    state_dict = torch.load(filepath, map_location=device)
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    
    for name, param in state_dict.items():
        name = name.replace("module.model.", "") # for madry loaded model
        if name not in own_state:
            print('not found: ',name)
            continue
        #if 'conv' in name and 'weight' in name:
        #    print(name+': '+'0.5')
        param = param.data
        own_state[name].copy_(param)
    return model

def save_logs(logs, filepath):
    np.save(filepath, logs)

def load_logs(filepath):
    logs = np.load(filepath, allow_pickle=True)[()]
    return logs

