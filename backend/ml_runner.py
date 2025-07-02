# backend/ml_runner.py
import os
import argparse
import torch
import json
import traceback
import numpy as np

# Import from the SIDDHI folder and your config
from SIDDHI.exp.exp_classification import Exp_Classification
from SIDDHI.utils.tools import dotdict
from config import OUTPUT_JSON_PATH

# --- Global variable to hold the initialized model and experiment ---
# This will keep the model in memory after it's loaded once.
EXP_INSTANCE = None

def _initialize_model_and_exp():
    """
    This function is called only once per worker process to load the model
    and experiment setup into memory.
    """
    global EXP_INSTANCE
    # If the model is already loaded, do nothing.
    if EXP_INSTANCE is not None:
        return

    print("ML Runner: Initializing model for the first time...")

    # 1. Create an 'args' object with all the required parameters
    # These are the same parameters previously passed via command line
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

    # The SIDDHI code uses relative paths, so we need to temporarily
    # change the directory to the SIDDHI folder for initialization.
    original_cwd = os.getcwd()
    siddhi_path = os.path.join(os.path.dirname(__file__), 'SIDDHI')
    os.chdir(siddhi_path)

    try:
        # 2. Create the Experiment instance, which builds the model structure
        exp = Exp_Classification(args)

        # 3. Load the pre-trained model weights into the model structure
        setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor,
            args.embed, args.des, 0, 0
        )
        path = os.path.join(args.checkpoints, setting)
        best_model_path = os.path.join(path, 'checkpoint.pth')

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Checkpoint file not found at {best_model_path}")

        # Load the model weights onto the CPU
        exp.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
        exp.model.eval() # Set the model to evaluation mode
        print("ML Runner: Model weights loaded successfully and set to evaluation mode.")

        # Store the fully initialized experiment object in our global variable
        EXP_INSTANCE = exp

    finally:
        # Always change back to the original directory
        os.chdir(original_cwd)


def run_model(filepath_to_process: str):
    """
    Executes a prediction using the pre-loaded ML model. This function replaces
    the old subprocess-based approach.
    """
    global EXP_INSTANCE
    
    # 1. Ensure the model is loaded into memory. If not, initialize it.
    if EXP_INSTANCE is None:
        _initialize_model_and_exp()

    print(f"ML Runner: Running prediction for: {filepath_to_process}")

    # The prediction logic needs the absolute path to the file
    absolute_filepath_for_ml = os.path.abspath(filepath_to_process)

    # 2. Set the input file on the existing args object of our loaded experiment
    EXP_INSTANCE.args.input_file = absolute_filepath_for_ml

    # The SIDDHI code uses relative paths, so we must change the directory again
    original_cwd = os.getcwd()
    siddhi_path = os.path.join(os.path.dirname(__file__), 'SIDDHI')
    os.chdir(siddhi_path)

    try:
        # 3. Run the prediction. `load=False` is critical because it tells the
        # function to use the already-loaded model weights.
        EXP_INSTANCE.predict(setting='dummy_setting', load=False)

        # 4. The `predict` method creates 'output.json' in the current directory (SIDDHI)
        output_file_in_siddhi = os.path.join(siddhi_path, 'output.json')

        if not os.path.exists(output_file_in_siddhi):
            raise FileNotFoundError("ML Runner Error: 'output.json' was not created after prediction.")

        # 5. Read the result and write it to the final destination defined in config
        with open(output_file_in_siddhi, 'r') as f_in:
            data = json.load(f_in)
        with open(OUTPUT_JSON_PATH, 'w') as f_out:
            json.dump(data, f_out)
        
        os.remove(output_file_in_siddhi) # Clean up the intermediate file

        print("ML Runner: Prediction successful.")
        return OUTPUT_JSON_PATH # Return the path to the final output file

    except Exception as e:
        print(f"ML Runner Error: An error occurred during prediction: {e}")
        traceback.print_exc()
        raise
    finally:
        os.chdir(original_cwd)

