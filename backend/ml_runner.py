# backend/ml_runner.py
import os
import sys # 
import argparse
import torch
import json
import traceback
import numpy as np

# --- FIX: Add the SIDDHI directory to the Python path ---
# This ensures that all sub-modules within SIDDHI can find each other.
siddhi_path = os.path.join(os.path.dirname(__file__), 'SIDDHI')
if siddhi_path not in sys.path:
    sys.path.insert(0, siddhi_path)

# Now, we can safely import from the SIDDHI package
from exp.exp_classification import Exp_Classification
from utils.tools import dotdict
from config import OUTPUT_JSON_PATH

# --- Global variable to hold the initialized model and experiment ---
EXP_INSTANCE = None

def _initialize_model_and_exp():
    """
    This function is called only once per worker process to load the model
    and experiment setup into memory.
    """
    global EXP_INSTANCE
    if EXP_INSTANCE is not None:
        return

    print("ML Runner: Initializing model for the first time...")

    args = dotdict()
    args.task_name = 'classification'
    args.is_training = 0
    args.model_id = 'ADSZ-Indep'
    args.model = 'ADformer'
    args.data = 'ADSZIndep'
    args.e_layers = 6
    args.batch_size = 1
    args.d_model = 128
    args.d_ff = 256
    args.enc_in = 19
    args.num_class = 2
    args.seq_len = 128
    args.use_gpu = False
    args.features = 'M'
    args.label_len = 48
    args.pred_len = 96
    args.n_heads = 8
    args.d_layers = 1
    args.factor = 1
    args.embed = 'timeF'
    args.des = 'Exp'
    args.patch_len_list = [4]
    args.up_dim_list = [19]
    args.root_path = './'
    args.data_path = ''
    args.checkpoints = './checkpoints'
    args.device = torch.device('cpu')

    original_cwd = os.getcwd()
    # The siddhi_path is already defined above
    os.chdir(siddhi_path)

    try:
        exp = Exp_Classification(args)

        setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor,
            args.embed, args.des, 0, 0
        )
        path = os.path.join(args.checkpoints, setting)
        best_model_path = os.path.join(path, 'checkpoint.pth')

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Checkpoint file not found at {best_model_path}")

        exp.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
        exp.model.eval()
        print("ML Runner: Model weights loaded successfully.")

        EXP_INSTANCE = exp

    finally:
        os.chdir(original_cwd)


def run_model(filepath_to_process: str):
    """
    Executes a prediction using the pre-loaded ML model.
    """
    global EXP_INSTANCE
    
    if EXP_INSTANCE is None:
        _initialize_model_and_exp()

    print(f"ML Runner: Running prediction for: {filepath_to_process}")
    absolute_filepath_for_ml = os.path.abspath(filepath_to_process)
    EXP_INSTANCE.args.input_file = absolute_filepath_for_ml

    original_cwd = os.getcwd()
    # The siddhi_path is already defined above
    os.chdir(siddhi_path)

    try:
        EXP_INSTANCE.predict(setting='dummy_setting', load=False)
        output_file_in_siddhi = os.path.join(siddhi_path, 'output.json')

        if not os.path.exists(output_file_in_siddhi):
            raise FileNotFoundError("ML Runner Error: 'output.json' was not created.")

        with open(output_file_in_siddhi, 'r') as f_in:
            data = json.load(f_in)
        with open(OUTPUT_JSON_PATH, 'w') as f_out:
            json.dump(data, f_out)
        
        os.remove(output_file_in_siddhi)

        print("ML Runner: Prediction successful.")
        return OUTPUT_JSON_PATH

    except Exception as e:
        print(f"ML Runner Error: An error occurred during prediction: {e}")
        traceback.print_exc()
        raise
    finally:
        os.chdir(original_cwd)
