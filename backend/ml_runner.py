import os
import subprocess
import traceback
from config import SIDDHI_FOLDER, OUTPUT_JSON_PATH

# --- FIX: Define paths relative to this file's location ---
# This is more robust for running inside a Docker container.
ML_RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))

def run_model(filepath_to_process: str):
    """
    Executes the SIDDHI ML model script as a subprocess.
    """
    print(f"ML Runner: Executing ML model for: {filepath_to_process}")
    
    # --- FIX: Use the new robust path definition ---
    siddhi_absolute_path = os.path.join(ML_RUNNER_DIR, SIDDHI_FOLDER)
    absolute_filepath_for_ml = os.path.abspath(filepath_to_process)
    
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
        cmd = [
            'python', 'run.py', 
            '--task_name', 'classification', 
            '--is_training', '0', 
            '--model_id', 'ADSZ-Indep', 
            '--model', 'ADformer', 
            '--data', 'ADSZIndep', 
            '--e_layers', '6', 
            '--batch_size', '1',
            '--d_model', '128', 
            '--d_ff', '256', 
            '--enc_in', '19', 
            '--num_class', '2', 
            '--seq_len', '128', 
            '--input_file', absolute_filepath_for_ml,
            '--use_gpu', 'False',
            '--features', 'M', 
            '--label_len', '48',
            '--pred_len', '96', 
            '--n_heads', '8', 
            '--d_layers', '1', 
            '--factor', '1', 
            '--embed', 'timeF',
            '--des', "'Exp'",
            "--patch_len_list", "4",
            "--up_dim_list", "19",
        ]
        
        print(f"ML Runner: Running ML command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=360)
        
        print(f"ML Runner: ML Model STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"ML Runner: ML Model STDERR:\n{result.stderr}")
        
        if not os.path.exists('output.json'):
            raise FileNotFoundError(f"ML Runner Error: 'output.json' not created in {siddhi_absolute_path} after script execution.")
        
        print("ML Runner: ML model script executed successfully.")
        return expected_output_json_in_siddhi

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
