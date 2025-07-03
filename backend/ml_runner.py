import os
import subprocess
import traceback
from config import SIDDHI_FOLDER, BACKEND_DIR, OUTPUT_JSON_PATH

def run_model(filepath_to_process: str):
    """
    Executes the SIDDHI ML model script as a subprocess.
    """
    print(f"ML Runner: Executing ML model for: {filepath_to_process}")
    
    siddhi_absolute_path = os.path.join(BACKEND_DIR, SIDDHI_FOLDER)
    absolute_filepath_for_ml = os.path.abspath(filepath_to_process) # Ensure input path is absolute for the script
    
    # The output.json path is now defined in config relative to BACKEND_DIR
    # For the script running inside SIDDHI_FOLDER, it will output to its CWD
    expected_output_json_in_siddhi = os.path.join(siddhi_absolute_path, 'output.json')


    if not os.path.isdir(siddhi_absolute_path):
        raise FileNotFoundError(f"ML Runner Error: SIDDHI directory not found at: {siddhi_absolute_path}")
    if not os.path.isfile(absolute_filepath_for_ml):
        raise FileNotFoundError(f"ML Runner Error: Input EEG file not found at: {absolute_filepath_for_ml}")

    # Clean up previous output if it exists
    if os.path.exists(expected_output_json_in_siddhi):
        try:
            os.remove(expected_output_json_in_siddhi)
            print(f"ML Runner: Removed existing ML output file: {expected_output_json_in_siddhi}")
        except Exception as rem_e:
            print(f"ML Runner Warning: Could not remove existing {expected_output_json_in_siddhi}: {rem_e}")

    original_cwd = os.getcwd()
    print(f"ML Runner: Temporarily changing CWD from '{original_cwd}' to '{siddhi_absolute_path}'")
    os.chdir(siddhi_absolute_path)

    try:
        # These arguments should match those expected by your SIDDHI/run.py and how they were used in the original app.py
        cmd = [
            'python', 'run.py', 
            '--task_name', 'classification', 
            '--is_training', '0', 
            '--model_id', 'ADSZ-Indep', 
            '--model', 'ADformer', 
            '--data', 'ADSZIndep', 
            '--e_layers', '6', 
            '--batch_size', '1', # Often 1 for single prediction
            '--d_model', '128', 
            '--d_ff', '256', 
            '--enc_in', '19', 
            '--num_class', '2', 
            '--seq_len', '128', 
            '--input_file', absolute_filepath_for_ml, # Pass absolute path
            '--use_gpu', 'False', # As per original call
            '--features', 'M', 
            '--label_len', '48',
            '--pred_len', '96', 
            '--n_heads', '8', 
            '--d_layers', '1', 
            '--factor', '1', 
            '--embed', 'timeF',
            '--des', "'Exp'", # Ensure Exp is not quoted in final command if SIDDHI expects it so
            # Add other ADformer specific args if they were hardcoded or derived in original app.py:
            "--patch_len_list", "4",
            "--up_dim_list", "19",
            # "--augmentations", "none", # if needed by run.py for model loading
            # "--no_inter_attn", # if applicable
            # "--no_temporal_block", # if applicable
            # "--no_channel_block", # if applicable
        ]
        
        # Handle boolean flags like --distil (action='store_false', default=True)
        # If the original app.py logic implies args.distil would be True, then don't pass --distil.
        # If it would be False, pass '--distil'. The current setup implies it's True by default.

        print(f"ML Runner: Running ML command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=360) # Increased timeout
        
        print(f"ML Runner: ML Model STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"ML Runner: ML Model STDERR:\n{result.stderr}")
        
        if not os.path.exists('output.json'): # Check in current (SIDDHI) directory
            raise FileNotFoundError(f"ML Runner Error: 'output.json' not created in {siddhi_absolute_path} after script execution.")
        
        print("ML Runner: ML model script executed successfully.")
        # The output.json is expected to be in siddhi_absolute_path now
        return expected_output_json_in_siddhi # Return path to the output file

    except subprocess.CalledProcessError as proc_error:
        print(f"ML Runner Error: ML script execution failed (Return Code {proc_error.returncode})\n--- ML STDERR ---\n{proc_error.stderr}\n--- End ML STDERR ---")
        traceback.print_exc()
        raise
    except subprocess.TimeoutExpired:
        print("ML Runner Error: ML script execution timed out.")
        raise TimeoutError("ML model execution timed out.")
    except FileNotFoundError as fnf_error:
        print(f"ML Runner File System Error: {fnf_error}")
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"ML Runner Error: An unexpected error occurred: {e}")
        traceback.print_exc()
        raise
    finally:
        print(f"ML Runner: Changing CWD back to original: {original_cwd}")
        os.chdir(original_cwd)