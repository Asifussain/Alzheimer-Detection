# backend/SIDDHI/exp/exp_classification.py
from copy import deepcopy
from exp.exp_basic import Exp_Basic
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import Counter
import json
import scipy.stats # Ensure scipy is imported
import traceback # For detailed error logging

# --- Import sklearn metrics ---
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# --- End Import ---

# Import your model definitions (adjust path/names if necessary)
# Assuming models are in a 'models' sibling directory to 'exp'
# Make sure the necessary model classes like ADformer are imported
try:
    # Example imports - adjust based on your actual model file structure
    from models.ADformer import Model as ADformer_Model
    # from models.OtherModel import Model as OtherModel_Model
    # Add other models your project uses
except ImportError as e:
    print(f"Warning: Could not import default model classes: {e}. Ensure models are discoverable.")
    # Define placeholders if imports fail but allow code structure to be checked
    class PlaceholderModel:
        class Model:
            def __init__(self, args): pass
    ADformer_Model = PlaceholderModel.Model


warnings.filterwarnings("ignore")

# Helper class to encode NumPy types to JSON if needed
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj): return None
            if np.isinf(obj): return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)): # Handle numpy bool
            return bool(obj)
        return super(NpEncoder, self).default(obj)

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        # Ensure SWA model parts are initialized even if swa arg is missing/false
        if not hasattr(args, 'swa') or not args.swa:
            self.swa = False
            # Initialize swa_model to None if SWA is not used
            self.swa_model = None
        else:
            # Only initialize AveragedModel if SWA is enabled AND self.model exists
            if self.model is not None:
                self.swa = args.swa
                self.swa_model = optim.swa_utils.AveragedModel(self.model)
                print("SWA model wrapper initialized.")
            else:
                print("Warning: SWA enabled but self.model is not yet built. SWA wrapper not initialized.")
                self.swa = False # Disable SWA if model isn't ready
                self.swa_model = None


    def _build_model(self):
        print("Building model...")
        # Example model dictionary - UPDATE THIS WITH YOUR ACTUAL MODELS
        model_dict = {
            'ADformer': ADformer_Model, # Use the imported class
            # 'OtherModel': OtherModel_Model, # Add other models here
        }
        # Ensure self.args has model attribute
        model_name = getattr(self.args, 'model', None)
        if model_name is None or model_name not in model_dict:
            raise ValueError(f"Model name '{model_name}' not found in model_dict or args. Available: {list(model_dict.keys())}")

        print(f"Using model class: {model_dict[model_name]}")
        model = model_dict[model_name](self.args).float() # Instantiate the model

        # Ensure necessary GPU args exist before checking them
        use_gpu = hasattr(self.args, 'use_gpu') and self.args.use_gpu
        use_multi_gpu = hasattr(self.args, 'use_multi_gpu') and self.args.use_multi_gpu
        device_ids = getattr(self.args, 'device_ids', []) # Default to empty list

        if use_multi_gpu and use_gpu and device_ids and torch.cuda.device_count() > 1:
            try:
                # Ensure device_ids are valid integers
                valid_device_ids = [int(d) for d in device_ids]
                model = nn.DataParallel(model, device_ids=valid_device_ids)
                print(f"Using DataParallel on devices: {valid_device_ids}")
            except ValueError:
                 print(f"Warning: Invalid device_ids format: {device_ids}. Using default device instead.")
            except AssertionError as e:
                 print(f"Warning: DataParallel assertion error (often indicates issues with device IDs): {e}. Using default device instead.")

        # Note: Model is moved to the primary device (self.device) in Exp_Basic's __init__
        # DataParallel handles distributing to other specified GPUs if used.
        print("Model built successfully.")
        return model

    def predict_unlabeled_sample(self, npy_file_path, setting, device):
        """
        Loads the trained model checkpoint and predicts the class(es) for an unlabeled NPY file.

        Args:
            npy_file_path (str): Absolute path to the input .npy file.
            setting (str): The setting string used for training (e.g., ADSZ-Indep_ftM_sl128_...)
                           This should define the specific checkpoint directory name.
            device (torch.device): The device to run prediction on (e.g., 'cpu' or 'cuda:0').
        """
        # --- Start: Dynamic Relative Path Construction ---

        # 1. Construct the path dynamically relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__)) # .../backend/SIDDHI/exp

        # Use the 'setting' variable passed into the function.
        # This 'setting' string is crucial and should uniquely identify the trained model run folder.
        if not setting or not isinstance(setting, str):
            # If 'setting' is missing, we cannot reliably find the checkpoint.
            # It should be generated and passed correctly from run.py
            print("Error: 'setting' argument (specific experiment directory name) is missing or invalid.")
            # You could attempt a fallback construction here IF you know the exact format AND
            # have ALL necessary self.args fields, but it's less reliable than passing 'setting'.
            # Example Fallback (USE WITH CAUTION - ensure all args exist and format is correct):
            # try:
            #     setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(...) # Construct from self.args
            #     print(f"Warning: Constructed setting fallback: {setting}")
            # except AttributeError as e:
            #     print(f"Error constructing setting fallback from self.args: {e}")
            #     raise ValueError("Could not determine the specific experiment setting directory for checkpoint loading.") from e
            raise ValueError("Missing 'setting' argument required to locate the correct checkpoint directory.")


        # Build path components relative to the script directory
        # Go UP ('..') to SIDDHI, then DOWN into checkpoints/...
        relative_checkpoint_dir_path = os.path.join(
            '..',  # Go up from 'exp' to 'SIDDHI'
            'checkpoints',
            self.args.task_name, # e.g., "classification"
            self.args.model_id,  # e.g., "ADSZ-Indep"
            self.args.model,     # e.g., "ADformer"
            setting              # The specific experiment run directory name passed in
        )

        # Combine the script's directory with the relative path
        model_path = os.path.join(script_dir, relative_checkpoint_dir_path, 'checkpoint.pth')

        # Normalize the path (handles '..', ensures correct OS-specific slashes)
        model_path = os.path.normpath(model_path)

        print(f"Looking for model checkpoint at calculated path: {model_path}") # Debugging print

        # Check existence
        if not os.path.exists(model_path):
            print(f"Error: Model checkpoint not found.")
            print(f"Expected location based on calculated path: {model_path}")
            print(f"Current working directory: {os.getcwd()}") # Still useful context
            # Check if intermediate dirs exist for more detailed error
            parent_dir = os.path.dirname(model_path)
            if not os.path.exists(parent_dir):
                 print(f"Parent directory '{parent_dir}' also does not exist.")
                 print("Verify arguments: task_name, model_id, model, and the passed 'setting' string.")
            else:
                 print(f"Parent directory '{parent_dir}' exists, but 'checkpoint.pth' is missing.")
            raise FileNotFoundError("Model checkpoint not found at calculated path: %s" % model_path)

        # --- End: Dynamic Relative Path Construction ---


        # Load Model State (using your existing SWA logic)
        print(f"Loading model state from {model_path} onto device: {device}")
        # Determine if SWA should be used (check if initialized and enabled)
        use_swa = hasattr(self, 'swa') and self.swa and self.swa_model is not None

        try:
            # Choose the correct model instance (base model or SWA wrapper)
            if use_swa:
                # Ensure the SWA model instance exists if use_swa is True
                if self.swa_model is None:
                     raise RuntimeError("SWA is enabled but swa_model is None. Cannot load weights.")
                self.swa_model.load_state_dict(torch.load(model_path, map_location=device))
                self.swa_model = self.swa_model.to(device) # Ensure SWA model is on the correct device
                self.swa_model.eval()
                model_to_use = self.swa_model # Use the SWA model for prediction
                print("Using SWA model weights for prediction.")
            else:
                # Ensure the base model instance exists
                if self.model is None:
                    raise RuntimeError("Base model (self.model) is None. Cannot load weights.")
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                # self.model should already be on the correct device from Exp_Basic
                self.model.eval()
                model_to_use = self.model # Use the standard model for prediction
                print("Using standard model weights for prediction.")

        except Exception as load_err:
            print(f"Error loading model state dict from {model_path}: {load_err}")
            traceback.print_exc()
            raise


        # Load and Prepare Data (using your existing logic)
        try:
            print(f"Loading input EEG data from: {npy_file_path}")
            X_orig = np.load(npy_file_path, allow_pickle=True)
            print(f"Original input data shape: {X_orig.shape}")
        except Exception as e:
            print(f"Error loading .npy file {npy_file_path}: {e}")
            traceback.print_exc()
            raise

        # Handle multiple trials/segments (your existing logic)
        if X_orig.ndim == 3:
            num_trials, seq_len_data, channels_data = X_orig.shape
            print(f"Input is 3D ({X_orig.shape}), processing {num_trials} trials/segments.")
            X_batch = X_orig
        elif X_orig.ndim == 2:
            num_trials = 1
            seq_len_data, channels_data = X_orig.shape
            print(f"Input is 2D ({X_orig.shape}), processing as 1 trial/segment.")
            X_batch = np.expand_dims(X_orig, axis=0) # Add batch dimension
        else:
            raise ValueError(f"Unsupported input data dimension: {X_orig.ndim}. Expected 2 or 3.")

        # Validate shape against model expectations (your existing logic)
        expected_seq_len = getattr(self.args, 'seq_len', None) # Get expected len from args
        expected_channels = getattr(self.args, 'enc_in', None) # Get expected channels from args

        if expected_seq_len is None or expected_channels is None:
             print("Warning: Model's expected seq_len or enc_in not found in args. Skipping shape validation.")
        else:
            if seq_len_data != expected_seq_len:
                print(f"Warning: Input sequence length {seq_len_data} != expected {expected_seq_len}.")
                # !! IMPORTANT: Add truncation/padding logic here if sequences can vary !!
                # Example: Truncate if too long
                # if seq_len_data > expected_seq_len:
                #     print(f"Truncating sequence length from {seq_len_data} to {expected_seq_len}")
                #     X_batch = X_batch[:, :expected_seq_len, :]
                #     seq_len_data = expected_seq_len # Update length after truncation
                # Example: Pad if too short (requires knowing padding value)
                # elif seq_len_data < expected_seq_len:
                #     pad_width = ((0, 0), (0, expected_seq_len - seq_len_data), (0, 0)) # Pad sequence dim
                #     X_batch = np.pad(X_batch, pad_width, mode='constant', constant_values=0) # Pad with 0
                #     print(f"Padding sequence length from {seq_len_data} to {expected_seq_len}")
                #     seq_len_data = expected_seq_len # Update length after padding
                # else: pass # Length matches

            if channels_data != expected_channels:
                raise ValueError(f"Input data has {channels_data} channels, model expects {expected_channels}.")

        print(f"Data shape for model processing: {X_batch.shape}") # Should be (num_trials, seq_len, channels)

        # Convert to Tensor
        X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)

        # --- Perform Inference on all trials/segments ---
        all_predictions = []
        all_probabilities = []
        print(f"Running inference on {num_trials} trial(s) using device: {device}...")
        try:
            with torch.no_grad(): # Essential for inference
                # Process data - assuming model_to_use is the correct model instance (SWA or standard)
                batch_size, current_seq_len, _ = X_tensor.shape

                # Create necessary model inputs (adapt based on your model's forward signature)
                # Often includes input tensor (x_enc) and potentially masks.
                # Example assumes padding_mask is needed and is all True (no actual padding)
                # If you implemented padding above, this mask MUST reflect the padded elements.
                padding_mask = torch.ones((batch_size, current_seq_len), dtype=torch.bool).to(device) # Use boolean mask if model expects it

                # Example call: Assumes model takes x_enc, padding_mask, x_mark_enc, x_mark_dec
                # Set mark tensors to None if not used by your classification model.
                outputs = model_to_use(X_tensor, padding_mask, None, None) # Adjust arguments as needed!

                # If model returns tuple (e.g., output, attention_weights), take the first element
                if isinstance(outputs, tuple):
                     outputs = outputs[0]

                # Process outputs (e.g., softmax for probabilities, argmax for predictions)
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                predictions = torch.argmax(probs, dim=-1)

                all_predictions = predictions.cpu().numpy()
                all_probabilities = probs.cpu().numpy()
        except Exception as inference_err:
            print(f"Error during model inference: {inference_err}")
            print(f"Input tensor shape during error: {X_tensor.shape}")
            traceback.print_exc()
            raise

        print(f"Individual trial predictions (raw): {all_predictions}")

        # --- Calculate Final Prediction (Majority Vote) & Metrics (using your existing logic) ---
        majority_prediction = -1 # Default error value
        consistency_metrics = {"error": "Calculation failed or not applicable"} # Default error

        if num_trials > 0 and len(all_predictions) == num_trials:
            try:
                # Final prediction is the majority vote
                count = Counter(all_predictions)
                # Handle tie-breaking (default to 0 = Normal)
                if len(count) > 1 and len(count.most_common(2)) > 1 and count.most_common(2)[0][1] == count.most_common(2)[1][1]:
                    print("Warning: Tie in majority vote. Defaulting to 0 (Normal).")
                    majority_prediction = 0
                else:
                    majority_prediction = count.most_common(1)[0][0]

                print(f"Majority Prediction (0=Normal, 1=Alz): {majority_prediction}")

                # Calculate consistency metrics IF multiple trials exist
                if num_trials > 1:
                    print("Calculating internal consistency metrics...")
                    # Create 'true' labels based on the majority prediction
                    y_true = np.full(num_trials, majority_prediction)
                    y_pred = all_predictions

                    accuracy = accuracy_score(y_true, y_pred)
                    # Calculate precision, recall, f1 for the positive class (label 1 = Alzheimer's)
                    # Use zero_division=0 for safety
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', pos_label=1, zero_division=0
                    )
                    # Specificity (Recall for label 0)
                    _, specificity, _, _ = precision_recall_fscore_support(
                         y_true, y_pred, average='binary', pos_label=0, zero_division=0
                    )

                    # Confusion Matrix handling (your logic seems robust)
                    unique_labels_in_preds = np.unique(y_pred)
                    if len(unique_labels_in_preds) == 1:
                         print(f"Warning: Only one class ({unique_labels_in_preds[0]}) predicted across trials.")
                         # Simplified CM calculation for single class prediction
                         if unique_labels_in_preds[0] == 0: # All Normal
                             tn = len(y_true) if majority_prediction == 0 else 0
                             fp = 0 if majority_prediction == 0 else len(y_true)
                             fn, tp = 0, 0
                         else: # All Alzheimer's
                             tn, fp = 0, 0
                             fn = len(y_true) if majority_prediction == 0 else 0
                             tp = 0 if majority_prediction == 0 else len(y_true)
                         # Recalculate metrics based on these counts
                         precision = 0 if tp + fp == 0 else tp / (tp + fp)
                         recall = 0 if tp + fn == 0 else tp / (tp + fn)
                         specificity = 0 if tn + fp == 0 else tn / (tn + fp)
                         f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
                    else: # Both classes predicted
                         cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                         if cm.shape == (2, 2):
                             tn, fp, fn, tp = cm.ravel()
                         else: # Fallback
                             print(f"Warning: Unexpected confusion matrix shape: {cm.shape}. Setting counts/metrics to 0.")
                             tn, fp, fn, tp = 0, 0, 0, 0
                             precision, recall, specificity, f1 = 0, 0, 0, 0

                    consistency_metrics = {
                        "num_trials": int(num_trials),
                        "num_normal_pred": int(np.sum(y_pred == 0)),
                        "num_alz_pred": int(np.sum(y_pred == 1)),
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall_sensitivity": float(recall),
                        "specificity": float(specificity),
                        "f1_score": float(f1),
                        "true_positives": int(tp),
                        "true_negatives": int(tn),
                        "false_positives": int(fp),
                        "false_negatives": int(fn),
                        "majority_label_used_as_reference": int(majority_prediction)
                    }
                    print(f"Consistency Metrics: {json.dumps(consistency_metrics, cls=NpEncoder, indent=4)}")
                else: # num_trials == 1
                    print("Only one trial/segment found, consistency metrics are not applicable.")
                    consistency_metrics = {"num_trials": 1, "message": "Metrics not applicable for single segment input"}

            except Exception as metrics_err:
                print(f"Error calculating metrics: {metrics_err}")
                traceback.print_exc()
                consistency_metrics = {"error": f"Metrics calculation failed: {metrics_err}"}

        else: # Handle case where inference yielded empty results
            print("Warning: No predictions available to calculate majority or metrics.")
            majority_prediction = -1
            consistency_metrics = {"error": "No predictions generated"}
            all_predictions = []


        # --- Prepare final output dictionary (using your existing logic) ---
        # Use probabilities from the first trial if available
        first_trial_probabilities = all_probabilities[0].tolist() if len(all_probabilities) > 0 else None

        results = {
            "majority_prediction": int(majority_prediction),
            "probabilities": first_trial_probabilities, # Probabilities for the first trial
            "trial_predictions": all_predictions.tolist() if isinstance(all_predictions, np.ndarray) else all_predictions, # Ensure it's a list
            "consistency_metrics": consistency_metrics # Dict with calculated metrics
        }

        # --- Write result to output.json (using your existing logic) ---
        # This will be written in the CWD, which is .../backend/SIDDHI/
        output_file = 'output.json'
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, cls=NpEncoder, indent=4)
            print(f"Prediction results and metrics saved to {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"An error occurred saving results to {output_file}: {e}")
            traceback.print_exc()

        # --- Return results ---
        # The calling script (run.py) should read output.json to get the results
        # This function itself might not need to return anything if run.py reads the file,
        # but returning the dict can be useful for direct calls.
        return results

    # Include other methods from Exp_Classification if they exist (e.g., train, test, vali)
    # ...