import os
import pickle
import yaml
from . import utils

def create_scalers_from_yaml(file):
    with open(file, 'r') as f:
        lookup = yaml.load(f, Loader=yaml.FullLoader)
    scalers = {}
    for k, v in lookup.items():
        scalers[k] = (float(v['mean']), float(v['std']))
    return scalers

HERE = os.path.dirname(os.path.abspath(__file__))
#DEFAULT_SCALER_PATH = f'{HERE}/default_scalers_adjusted_pressure.yaml'
#DEFAULT_SCALER_PATH = f'{HERE}/CONUS21_scalers_delta_manual.yaml'
DEFAULT_SCALER_PATH = f'{HERE}/CONUS21_scalers_parflow_names.yaml'
DEFAULT_SCALER_PATH = f'{HERE}/PF_CLMTB_34335_scalers.yaml'
DEFAULT_SCALER_PATH = f'{HERE}/PF_CLMTB_34335_scalers_until_6000.yaml'
DEFAULT_SCALER_PATH = f'{HERE}/PF_CLMTB_34335_scalers_sigma_1_7.yaml'
DEFAULT_SCALERS = create_scalers_from_yaml(DEFAULT_SCALER_PATH)
