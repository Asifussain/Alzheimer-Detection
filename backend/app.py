# backend/app.py
# FINAL VERSION Incorporating Similarity Analysis, Channel Selection, Consistency Metrics & Syntax Fixes v3

import os
import uuid
import json
import subprocess
import io
import base64
import traceback
import pandas as pd
from datetime import datetime, timezone

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import numpy as np
from fpdf import FPDF, XPos, YPos # Keep extras

# Helper for JSON serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            # Handle NaN/Inf which are not valid JSON
            if np.isnan(obj) or np.isinf(obj): return None
            return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        # Handle numpy bool_ type
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super(NpEncoder, self).default(obj)

# Import visualization functions (existing)
try:
    from visualization import (
        generate_stacked_timeseries_image,
        generate_average_psd_image,
        generate_descriptive_stats
    )
    print("Successfully imported visualization functions.")
except ImportError as import_err:
    print(f"CRITICAL ERROR importing visualization.py: {import_err}")
    # Define dummy functions if import fails
    def generate_stacked_timeseries_image(*args, **kwargs): return None
    def generate_average_psd_image(*args, **kwargs): return None
    def generate_descriptive_stats(*args, **kwargs): return {"error": "Visualization module not loaded"}

# --- Import the Similarity Analyzer ---
try:
    from similarity_analyzer import run_similarity_analysis
    print("Successfully imported similarity_analyzer.")
except ImportError as sim_import_err:
    print(f"CRITICAL ERROR importing similarity_analyzer.py: {sim_import_err}")
    # Define dummy function if import fails
    def run_similarity_analysis(*args, **kwargs):
        return {"error": "Similarity analyzer module not loaded", 'interpretation': 'N/A', 'plot_base64': None, 'consistency_metrics': None}
# --- End Import ---

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
# Configure CORS properly for production - allowing specific origins is safer
# For development, "*" might be okay, but restrict in production.
CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_URL", "*")}})

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY: raise ValueError("Supabase environment variables not set.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase client initialized.")

UPLOAD_FOLDER = 'uploads'
SIDDHI_FOLDER = 'SIDDHI' # Assumed relative to backend folder
OUTPUT_JSON_PATH = os.path.join(SIDDHI_FOLDER, 'output.json')
# Define reference file paths relative to the backend script location
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure these filenames match the files placed in your backend directory
ALZ_REF_PATH = os.path.join(BACKEND_DIR, 'feature_07.npy')
NORM_REF_PATH = os.path.join(BACKEND_DIR, 'feature_35.npy')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_FS = 128
RAW_EEG_BUCKET = 'eeg-data'
REPORT_ASSET_BUCKET = 'report-assets' # Bucket for ALL generated assets

# --- Custom PDF Class (Updated with Styling) ---
class PDFReport(FPDF):
    def header(self):
        try:
            self.set_font('Helvetica', 'B', 15)
            title = "EEG Analysis Report"
            title_w = self.get_string_width(title) + 6
            doc_w = self.w
            self.set_x((doc_w - title_w) / 2)
            self.set_text_color(74, 144, 226) # Primary Blue for header text
            self.cell(title_w, 10, title, border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(0) # Reset color
            self.ln(5) # Reduce space after header
            # Optional: Add a thin line below header
            self.set_draw_color(200, 200, 200)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(8)
        except Exception as e: print(f"PDF Header Error: {e}")

    def footer(self):
        try:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
            self.set_text_color(0) # Reset color
        except Exception as e: print(f"PDF Footer Error: {e}")

    def section_title(self, title):
        try:
            self.set_font('Helvetica', 'B', 13)
            # Use a teal color for section title background
            self.set_fill_color(80, 227, 194) # Approx --accent-teal
            self.set_text_color(10, 15, 26) # Approx --background-start (dark text for contrast)
            self.cell(0, 8, " " + title, border='B', align='L', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(0) # Reset text color
            self.ln(6) # Increased space after title
        except Exception as e: print(f"PDF Section Title Error: {e}")

    def key_value_pair(self, key, value, key_width=45):
        try:
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(50, 50, 50) # Darker text for keys
            key_start_y = self.get_y()
            self.multi_cell(key_width, 6, str(key)+":", align='L', new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=self.font_size)
            self.set_y(key_start_y)
            self.set_x(self.l_margin + key_width + 2)
            self.set_font('Helvetica', '', 10)
            self.set_text_color(0) # Black for values
            self.multi_cell(0, 6, str(value), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(1) # Minimal gap between pairs
        except Exception as e: print(f"PDF Key/Value Error: {e}")

    def write_multiline(self, text, height=5, indent=5):
        try:
             self.set_font('Helvetica', '', 10)
             self.set_text_color(80, 80, 80) # Greyish text for multiline blocks
             self.set_left_margin(self.l_margin + indent) # Apply indentation
             self.multi_cell(0, height, text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
             self.set_left_margin(self.l_margin) # Reset left margin
             self.ln(height / 2)
             self.set_text_color(0) # Reset color
        except Exception as e: print(f"PDF Multiline Error: {e}")

    def metric_card(self, title, value, unit="", description=""):
        try:
            start_x = self.get_x(); start_y = self.get_y()
            card_width = (self.w - self.l_margin - self.r_margin - 5) / 2 # Two cards per row with 5mm gap
            card_height = 25

            # Card background and border
            self.set_fill_color(240, 245, 250) # Light blue-grey background
            self.set_draw_color(80, 227, 194) # Teal border
            self.set_line_width(0.3)
            self.rect(start_x, start_y, card_width, card_height, 'DF') # Draw filled rectangle

            # Title
            self.set_xy(start_x + 3, start_y + 3)
            self.set_font('Helvetica', 'B', 9) # Smaller title font
            self.set_text_color(80, 80, 80) # Dark grey title
            self.cell(card_width - 6, 5, title.upper(), align='L') # Uppercase title

            # Value
            self.set_xy(start_x + 3, start_y + 9)
            self.set_font('Helvetica', 'B', 16) # Larger value font
            self.set_text_color(74, 144, 226) # Primary blue value
            value_str = f"{value}{unit}"
            self.cell(card_width - 6, 8, value_str, align='R')

             # Description (Optional)
            if description:
                 self.set_xy(start_x + 3, start_y + 18)
                 self.set_font('Helvetica', 'I', 8)
                 self.set_text_color(100, 100, 100) # Medium grey italic
                 self.cell(card_width - 6, 5, description, align='L')

            self.set_y(start_y) # Reset Y for potential next card
            self.set_x(start_x + card_width + 5) # Move X for next card
            self.set_text_color(0) # Reset text color
            self.set_line_width(0.2) # Reset line width
        except Exception as e: print(f"PDF Metric Card Error: {e}")


# --- Helper Functions ---
def _cleanup_storage_on_error(bucket_name, path):
    """Attempt to remove a file from storage when an error occurs"""
    try:
        if bucket_name and path:
            print(f"Cleaning up storage: Removing {path} from {bucket_name}")
            supabase.storage.from_(bucket_name).remove([path])
            print(f"Successfully removed {path} from storage")
        else: print("Skipping storage cleanup: bucket_name or path is empty")
    except Exception as e: print(f"Error during storage cleanup: {e}")

def get_prediction_and_eeg(prediction_id):
    """Fetches prediction record and downloads/prepares EEG data."""
    print(f"Helper: Fetching record ID: {prediction_id}")
    prediction = None
    try:
        # Fetching all columns needed for the report
        prediction_res = supabase.table('predictions').select('*').eq('id', prediction_id).maybe_single().execute()
        if not prediction_res.data:
            print(f"Helper: Prediction ID {prediction_id} not found.")
            return None, None, "Prediction not found"
        prediction = prediction_res.data
        eeg_url_path = prediction.get('eeg_data_url')
        if not eeg_url_path:
            print(f"Helper: EEG URL missing for {prediction_id}.")
            return prediction, None, "EEG data URL not found"

        print(f"Helper: Downloading EEG from {eeg_url_path}...")
        eeg_file_response = supabase.storage.from_(RAW_EEG_BUCKET).download(eeg_url_path)

        if not isinstance(eeg_file_response, bytes):
             error_message = f"Failed raw EEG download for {eeg_url_path}: {getattr(eeg_file_response, 'message', str(eeg_file_response))}"
             print(f"Helper Error: {error_message}"); return prediction, None, error_message

        with io.BytesIO(eeg_file_response) as f: eeg_data = np.load(f, allow_pickle=True)
        print(f"Helper: Loaded EEG shape: {eeg_data.shape}")
        if eeg_data.ndim == 3: eeg_data = eeg_data[0, :, :]
        elif eeg_data.ndim != 2: raise ValueError(f"Unsupported EEG dim: {eeg_data.ndim}")
        if eeg_data.shape[0] < eeg_data.shape[1]: eeg_data = eeg_data.T
        if eeg_data.ndim != 2: raise ValueError(f"Final EEG not 2D: {eeg_data.shape}")
        print(f"Helper: Final EEG shape: {eeg_data.shape}")
        return prediction, eeg_data.astype(np.double), None # Ensure float64

    except Exception as e:
        print(f"Helper Error for {prediction_id}: {e}"); traceback.print_exc()
        return (prediction if prediction else None), None, f"Error accessing/processing data: {str(e)}"

def run_model(filepath_to_process):
    """Runs the SIDDHI ML model script with all necessary arguments."""
    print(f"Executing run_model for: {filepath_to_process}")

    # --- Path Setup ---
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Should be backend/
    siddhi_path = os.path.join(current_dir, SIDDHI_FOLDER) # backend/SIDDHI/
    absolute_filepath = os.path.abspath(filepath_to_process) # Get full path to input .npy file
    expected_output_json = os.path.join(siddhi_path, 'output.json') # Expected output location

    # --- Pre-checks ---
    if not os.path.isdir(siddhi_path):
        raise FileNotFoundError(f"SIDDHI directory not found: {siddhi_path}")
    if not os.path.isfile(absolute_filepath):
        raise FileNotFoundError(f"Input file not found: {absolute_filepath}")

    # Remove old output file if it exists
    if os.path.exists(expected_output_json):
        try:
            os.remove(expected_output_json)
            print(f"Removed existing ML output: {expected_output_json}")
        except Exception as rem_e:
            print(f"Warning: Could not remove existing {expected_output_json}: {rem_e}")

    # --- Execute ML Script ---
    original_cwd = os.getcwd()
    print(f"Changing CWD to: {siddhi_path}")
    os.chdir(siddhi_path) # Change CWD to where run.py is

    try:
        # --- Define the command with ALL required arguments ---
        cmd = [
            'python', 'run.py',

            # --- Arguments needed for general setup & model ---
            '--task_name', 'classification',
            '--is_training', '0',          # Set to prediction mode
            '--model_id', 'ADSZ-Indep',    # Matches directory structure
            '--model', 'ADformer',         # Matches directory structure
            '--data', 'ADSZIndep',         # Argument for run.py (check if used for path)
            '--e_layers', '6',             # Model hyperparameter
            '--batch_size', '1',           # Prediction batch size
            '--d_model', '128',            # Model hyperparameter
            '--d_ff', '256',               # Model hyperparameter
            '--enc_in', '19',              # Input feature dimension
            '--num_class', '2',            # Number of output classes
            '--seq_len', '128',            # Input sequence length

            # --- Input File & GPU ---
            '--input_file', absolute_filepath, # Pass the absolute path to the input .npy file
            '--use_gpu', 'False',          # Set based on your environment

            # --- *** NEWLY ADDED ARGUMENTS required for 'setting' string *** ---
            # These values MUST match the specifics of the checkpoint directory you want to load
            '--features', 'M',             # For the 'ftM' part
            '--label_len', '48',           # For the 'll48' part
            '--pred_len', '96',            # For the 'pl96' part
            '--n_heads', '8',              # For the 'nh8' part
            '--d_layers', '1',             # For the 'dl1' part
            '--factor', '1',               # For the 'fc1' part (Check if 'factor' or 'f_layers' or similar arg name)
            '--embed', 'timeF',            # For the 'ebtimeF' part
            '--des',  "'Exp'"                 # For the '_Exp' part

            # --- Optional arguments from previous command (Verify if needed by run.py) ---
            # '--patch_len_list', '4', # Included previously, check if needed by run.py or model init
            # '--up_dim_list', '19',   # Included previously, check if needed by run.py or model init
        ]

        print(f"Running command: {' '.join(cmd)}") # Log the command being run

        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,    # Capture STDOUT/STDERR
            text=True,              # Decode output as text
            check=True,             # Raise CalledProcessError if exit code != 0
            encoding='utf-8',       # Specify encoding
            timeout=300             # Add a timeout (e.g., 5 minutes)
        )

        # --- Post-execution ---
        print(f"ML Model STDOUT:\n{result.stdout}") # Print standard output
        if result.stderr: # Print standard error only if it's not empty
             print(f"ML Model STDERR:\n{result.stderr}")

        # Check if the expected output file was created
        if not os.path.exists('output.json'): # Checks in the CWD (which is SIDDHI)
            raise FileNotFoundError(f"'output.json' not created in {siddhi_path} after script execution.")

        print("ML model script finished successfully.")

        # --- Read Results (Optional here, might be done in predict() endpoint) ---
        # try:
        #     with open('output.json', 'r') as f:
        #         prediction_results = json.load(f)
        #     print("Successfully read output.json")
        #     # You can return prediction_results here if needed by the caller
        #     # return prediction_results
        # except Exception as read_err:
        #     print(f"Error reading output.json: {read_err}")
        #     raise # Or handle appropriately

    except subprocess.CalledProcessError as proc_error:
        # Log details including stderr from the failed process
        print(f"ML script failed (Code {proc_error.returncode})\n--- ML STDERR ---\n{proc_error.stderr}\n--- End ML STDERR ---")
        # Re-raise the error so the endpoint knows something went wrong
        raise proc_error
    except subprocess.TimeoutExpired:
        print("ML script timed out.")
        raise TimeoutError("ML model execution timed out.")
    except FileNotFoundError as fnf_error: # Catch specific FileNotFoundError
         print(f"File system error: {fnf_error}")
         raise # Re-raise
    except Exception as e: # Catch any other unexpected errors
         print(f"An unexpected error occurred in run_model: {e}")
         traceback.print_exc() # Print full traceback for unexpected errors
         raise # Re-raise
    finally:
        # --- Cleanup ---
        # Change CWD back to the original directory REQUIRED
        print(f"Changing CWD back to: {original_cwd}")
        os.chdir(original_cwd)


# --- PDF Content Builder (Updated with metrics) ---
def _build_simple_pdf_content(pdf_obj, prediction_data, stats_data, similarity_data, consistency_metrics, ts_img_data, psd_img_data, similarity_plot_data):
    """Builds the PDF content including similarity and consistency metrics."""
    pdf = pdf_obj; page_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_auto_page_break(auto=True, margin=15)

    try:
        # Page 1: Details, Prediction, Consistency
        pdf.add_page()
        # Header is called automatically

        pdf.section_title("Analysis Details")
        pdf.key_value_pair("Filename", prediction_data.get('filename', 'N/A'))
        created_at = prediction_data.get('created_at'); date_str = 'N/A'
        if created_at:
            try: dt_obj = pd.to_datetime(created_at); date_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S UTC') if dt_obj.tzinfo else dt_obj.strftime('%Y-%m-%d %H:%M:%S (?)')
            except Exception: date_str = str(created_at)
        pdf.key_value_pair("Analyzed On", date_str)
        pdf.ln(5)

        pdf.section_title("ML Prediction & Internal Consistency")
        prediction_label = prediction_data.get('prediction', 'N/A')
        pdf.key_value_pair("Overall Prediction", prediction_label)
        probabilities = prediction_data.get('probabilities'); prob_str = 'N/A'
        if isinstance(probabilities, list) and len(probabilities) == 2:
            try: prob_str = f"Normal: {probabilities[0]*100:.1f}%, Alzheimer's: {probabilities[1]*100:.1f}%"
            except Exception: prob_str = str(probabilities)
        elif probabilities is not None: prob_str = str(probabilities)
        pdf.key_value_pair("Confidence (first trial)", prob_str)
        pdf.ln(5)

        # Consistency Metrics Section
        if consistency_metrics and not consistency_metrics.get('error') and consistency_metrics.get('num_trials', 0) > 1:
            pdf.set_font('Helvetica', 'B', 11); pdf.cell(0, 6, "Internal Consistency Metrics:", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(2)
            pdf.set_font('Helvetica', 'I', 9); pdf.set_text_color(100, 100, 100); pdf.cell(0, 5, "(Compares segment predictions against the overall prediction for this file)", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(4); pdf.set_text_color(0)
            metrics = consistency_metrics
            pdf.metric_card("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}", unit="%", description="Overall segment agreement")
            pdf.metric_card("Precision (Alz)", f"{metrics.get('precision', 0):.3f}", unit="", description="Alz predictions correct")
            pdf.ln(28) # Move down past the cards + gap
            pdf.metric_card("Sensitivity (Alz)", f"{metrics.get('recall_sensitivity', 0):.3f}", unit="", description="Alz segments found")
            pdf.metric_card("Specificity (Norm)", f"{metrics.get('specificity', 0):.3f}", unit="", description="Normal segments found")
            pdf.ln(28)
            pdf.metric_card("F1-Score (Alz)", f"{metrics.get('f1_score', 0):.3f}", unit="", description="Precision/Sensitivity balance")
            pdf.metric_card("Trials Analyzed", f"{metrics.get('num_trials', 'N/A')}", unit="", description="Segments in file")
            pdf.ln(28)
            pdf.set_font('Helvetica', '', 9); pdf.set_text_color(100, 100, 100)
            conf_matrix_str = (f"(Ref Label: {metrics.get('majority_label_used_as_reference', '?')}) " f"TP:{metrics.get('true_positives','?')} | TN:{metrics.get('true_negatives','?')} | FP:{metrics.get('false_positives','?')} | FN:{metrics.get('false_negatives','?')}")
            pdf.cell(0, 5, conf_matrix_str, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT) ; pdf.set_text_color(0)
        elif consistency_metrics and consistency_metrics.get('message'): pdf.set_font('Helvetica', 'I', 10); pdf.write_multiline(f"({consistency_metrics['message']})", indent=5)
        else: pdf.set_font('Helvetica', 'I', 10); pdf.write_multiline("(Internal consistency metrics not calculated.)", indent=5)
        pdf.ln(5)

        # Page 2: Similarity, Stats, Standard Plots
        pdf.add_page()
        pdf.section_title("Signal Shape Similarity Analysis (DTW)")
        if similarity_data and not similarity_data.get('error'):
            pdf.write_multiline(similarity_data.get('interpretation', 'No similarity interpretation available.'), indent=5)
            pdf.ln(2)
            if similarity_plot_data and isinstance(similarity_plot_data, str) and similarity_plot_data.startswith('data:image/png;base64,'):
                plotted_ch = similarity_data.get('plotted_channel_index', None); plot_title = f"Channel {plotted_ch + 1} Comparison Plot:" if plotted_ch is not None else "Comparison Plot:"
                pdf.set_font("Helvetica",'B',11); pdf.cell(0, 8, plot_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                try: img_bytes = base64.b64decode(similarity_plot_data.split(',')[1]); img_file = io.BytesIO(img_bytes); img_width_mm = page_width * 0.9; x_pos = pdf.l_margin + (page_width - img_width_mm) / 2; pdf.image(img_file, x=x_pos, w=img_width_mm); img_file.close(); pdf.ln(5)
                except Exception as e: pdf.set_font("Helvetica",'I',10); pdf.set_text_color(255,0,0); pdf.cell(0,10,f"(Err embedding Sim Plot: {e})", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.set_text_color(0); print(f"PDF Sim Embed Err: {e}")
            else: pdf.set_font("Helvetica",'I',10); pdf.cell(0,10,"(Sim plot not generated)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else: pdf.set_font("Helvetica",'I',10); err_msg = similarity_data.get('error', 'Unknown') if similarity_data else 'N/A'; pdf.write_multiline(f"(Similarity Analysis Error: {err_msg})", indent=5)
        pdf.ln(5)

        pdf.section_title("Descriptive Statistics")
        if stats_data and not stats_data.get('error'):
            pdf.set_font("Helvetica",'B',11); pdf.cell(0,6,"Avg Relative Band Power (%):", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(1); pdf.set_font("Helvetica",size=10); avg_power = stats_data.get('avg_band_power',{}); band_found=False
            if avg_power:
                 for band, powers in avg_power.items(): rel_power = powers.get('relative', None); band_found |= (rel_power is not None); rel_power_str = f"{rel_power * 100:.2f}%" if isinstance(rel_power, (int, float)) else 'N/A'; pdf.cell(10); pdf.cell(0,5,f"- {band.capitalize()}: {rel_power_str}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if not band_found: pdf.set_font("Helvetica",'I',10); pdf.cell(10); pdf.cell(0,5,"(No band power data)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
        else: pdf.set_font("Helvetica",'I',10); err_msg = stats_data.get('error', 'Unknown') if stats_data else 'N/A'; pdf.write_multiline(f"(Stats Error: {err_msg})", indent=5)
        pdf.ln(5)

        # Page 3: Standard Visualizations
        pdf.add_page()
        pdf.section_title("Standard Visualizations")
        pdf.set_font("Helvetica",'B',12); pdf.cell(0,8,"Stacked Time Series", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(2)
        if ts_img_data and isinstance(ts_img_data, str) and ts_img_data.startswith('data:image/png;base64,'):
            try: img_bytes=base64.b64decode(ts_img_data.split(',')[1]); img_file=io.BytesIO(img_bytes); pdf.image(img_file, x=pdf.l_margin, w=page_width); img_file.close(); pdf.ln(5)
            except Exception as e: pdf.set_font("Helvetica",'I',10); pdf.set_text_color(255,0,0); pdf.cell(0,10,f"(Err embedding TS Plot: {e})", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.set_text_color(0); print(f"PDF TS Embed Err: {e}")
        else: pdf.set_font("Helvetica",'I',10); pdf.cell(0,10,"(TS plot not generated)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(10)

        pdf.set_font("Helvetica",'B',12); pdf.cell(0,8,"Average Power Spectral Density (PSD)", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(2)
        if psd_img_data and isinstance(psd_img_data, str) and psd_img_data.startswith('data:image/png;base64,'):
            try: img_bytes=base64.b64decode(psd_img_data.split(',')[1]); img_file=io.BytesIO(img_bytes); img_width_mm=page_width*0.9; x_pos=pdf.l_margin+(page_width-img_width_mm)/2; pdf.image(img_file, x=x_pos, w=img_width_mm); img_file.close(); pdf.ln(5)
            except Exception as e: pdf.set_font("Helvetica",'I',10); pdf.set_text_color(255,0,0); pdf.cell(0,10,f"(Err embedding PSD Plot: {e})", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.set_text_color(0); print(f"PDF PSD Embed Err: {e}")
        else: pdf.set_font("Helvetica",'I',10); pdf.cell(0,10,"(PSD plot not generated)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    except Exception as pdf_build_e:
        print(f"Error building PDF content: {pdf_build_e}"); traceback.print_exc()
        try:
            if pdf.page_no() == 0: pdf.add_page()
            elif pdf.get_y() > pdf.h - 30 : pdf.add_page()
            pdf.set_font("Helvetica",'B',12); pdf.set_text_color(255,0,0); pdf.multi_cell(0,10,f"Crit Err Building PDF:\n{pdf_build_e}",align='C'); pdf.set_text_color(0)
        except: pass


# --- Predict Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files.get('file'); user_id = request.form.get('user_id')
    try: channel_index_str = request.form.get('channel_index', '0'); channel_index_for_plot = int(channel_index_str); assert 0 <= channel_index_for_plot <= 18
    except (ValueError, TypeError, AssertionError): channel_index_for_plot = 0; print(f"Warning: Invalid channel index '{channel_index_str}'. Defaulting to 0.")

    if not file or not user_id: return jsonify({'error': "Missing 'file' or 'user_id'"}), 400
    if not file.filename or not file.filename.lower().endswith('.npy'): return jsonify({'error': 'Invalid/Missing filename or type (.npy required).'}), 400

    filename_base, file_extension = os.path.splitext(file.filename); unique_id = str(uuid.uuid4())
    save_filename = f"{filename_base}_{unique_id}{file_extension}"; temp_filepath = os.path.join(UPLOAD_FOLDER, save_filename)
    raw_eeg_storage_path = f'raw_eeg/{user_id}/{save_filename}'; prediction_id = None
    report_generation_errors = []; similarity_analysis_results = None; consistency_metrics_results = None; similarity_plot_url = None

    try:
        # Steps 1 & 2: Save & Upload Raw EEG
        print(f"Step 1/2: Processing '{file.filename}'..."); file.save(temp_filepath)
        with open(temp_filepath, 'rb') as f_upload: supabase.storage.from_(RAW_EEG_BUCKET).upload(path=raw_eeg_storage_path, file=f_upload, file_options={"content-type": "application/octet-stream", "upsert": "false"})
        print("Step 1/2: Raw EEG upload successful.")

        # Step 3: Run ML Model
        print(f"Step 3: Running ML model..."); run_model(temp_filepath)
        ml_output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_JSON_PATH)
        if not os.path.exists(ml_output_file_path): raise FileNotFoundError(f"ML output missing: {ml_output_file_path}")
        with open(ml_output_file_path, 'r') as f: output_data = json.load(f)
        prediction_label = "Alzheimer's" if output_data.get('majority_prediction') == 1 else "Normal"; probabilities = output_data.get('probabilities'); consistency_metrics_results = output_data.get('consistency_metrics'); trial_predictions = output_data.get('trial_predictions')
        print(f"Step 3: ML prediction: {prediction_label}"); print(f"Step 3: Consistency Metrics: {consistency_metrics_results}")

        # Step 4: Insert Initial DB Record (with new fields)
        insert_data = {"user_id": user_id, "filename": file.filename, "prediction": prediction_label, "eeg_data_url": raw_eeg_storage_path, "probabilities": probabilities, "status": "Processing", "trial_predictions": trial_predictions, "consistency_metrics": consistency_metrics_results}
        print(f"Step 4: Inserting prediction record..."); insert_payload = json.loads(json.dumps(insert_data, cls=NpEncoder, allow_nan=False)); insert_res = supabase.table('predictions').insert(insert_payload).execute()
        if insert_res.data and len(insert_res.data) > 0: prediction_id = insert_res.data[0].get('id'); print(f"DB Insert successful. ID: {prediction_id}")
        else: _cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path); raise Exception(f"DB insert failed: {insert_res}")

        # Step 5: Generate Report Assets
        print(f"--- Step 5: Generating Report Assets for ID: {prediction_id} ---")
        stats_json, ts_img_data, psd_img_data = None, None, None; ts_url, psd_url, pdf_url = None, None, None
        report_generation_status = "Pending"; similarity_plot_filename = None

        try:
            print("Step 5a: Fetching data for report..."); prediction_data_for_report, eeg_data, error_msg = get_prediction_and_eeg(prediction_id)
            if error_msg or eeg_data is None: raise Exception(f"Cannot load data for report: {error_msg or 'No EEG data'}")
            print(f"Step 5a: Data fetched. EEG shape: {eeg_data.shape}"); report_generation_status = "Generating Assets"

            # Step 5b: Run Similarity Analysis
            print(f"Step 5b: Running Similarity Analysis (Plot Channel: {channel_index_for_plot+1})...")
            similarity_analysis_results = run_similarity_analysis(temp_filepath, ALZ_REF_PATH, NORM_REF_PATH, channel_index_for_plot)
            if isinstance(similarity_analysis_results, dict): similarity_analysis_results['plotted_channel_index'] = channel_index_for_plot
            if similarity_analysis_results.get('error'): print(f"Warn: Sim analysis failed: {similarity_analysis_results['error']}"); report_generation_errors.append("Sim Analysis")
            else:
                print("Sim analysis successful.")
                sim_plot_base64 = similarity_analysis_results.get('plot_base64')
                if sim_plot_base64 and isinstance(sim_plot_base64, str) and sim_plot_base64.startswith('data:image/png;base64,'):
                    try:
                        similarity_plot_filename = f"report_assets/{prediction_id}/similarity_plot_ch{channel_index_for_plot + 1}.png"; sim_plot_bytes = base64.b64decode(sim_plot_base64.split(',')[1])
                        supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=similarity_plot_filename, file=sim_plot_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                        similarity_plot_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(similarity_plot_filename)
                        if not isinstance(similarity_plot_url, str) or not similarity_plot_url.startswith('http'): print(f"Warn: Invalid Sim Plot URL: {similarity_plot_url}"); similarity_plot_url = None; report_generation_errors.append("Sim Plot URL")
                        else: print(f"Sim Plot URL: {similarity_plot_url}")
                    except Exception as e: print(f"ERR uploading Sim plot: {e}"); traceback.print_exc(); report_generation_errors.append("Sim Plot Upload")
                else: print("Warn: Sim plot not generated/invalid."); report_generation_errors.append("Sim Plot Gen")

            # Step 5c: Generate Stats
            print("Step 5c: Generating stats..."); stats_json = generate_descriptive_stats(eeg_data, DEFAULT_FS)
            if isinstance(stats_json, dict) and 'error' in stats_json: print(f"Warn: Stats failed: {stats_json['error']}"); report_generation_errors.append("Stats")
            elif stats_json is None: print(f"Warn: Stats None."); stats_json = {'error': 'Stats None'}; report_generation_errors.append("Stats")
            else: print("Stats successful.")

            # Step 5d: Generate TS Plot
            print("Step 5d: Generating TS plot..."); ts_img_data = generate_stacked_timeseries_image(eeg_data, DEFAULT_FS); ts_filename = f"report_assets/{prediction_id}/timeseries.png"
            if ts_img_data and isinstance(ts_img_data, str) and ts_img_data.startswith('data:image/png;base64,'):
                try:
                    ts_bytes = base64.b64decode(ts_img_data.split(',')[1])
                    supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=ts_filename, file=ts_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                    ts_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(ts_filename)
                    # --- CORRECTED SYNTAX ---
                    if not isinstance(ts_url, str) or not ts_url.startswith('http'):
                        print(f"Warn: Invalid TS URL: {ts_url}")
                        ts_url=None
                        report_generation_errors.append("TS URL")
                    else:
                        print(f"TS URL: {ts_url}")
                    # --- END CORRECTION ---
                except Exception as e: print(f"ERR uploading TS plot: {e}"); traceback.print_exc(); report_generation_errors.append("TS Upload")
            else: print("Warn: TS gen failed/invalid."); report_generation_errors.append("TS Generation")

            # Step 5e: Generate PSD Plot
            print("Step 5e: Generating PSD plot..."); psd_img_data = generate_average_psd_image(eeg_data, DEFAULT_FS); psd_filename = f"report_assets/{prediction_id}/psd.png"
            if psd_img_data and isinstance(psd_img_data, str) and psd_img_data.startswith('data:image/png;base64,'):
                try:
                    psd_bytes = base64.b64decode(psd_img_data.split(',')[1])
                    supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=psd_filename, file=psd_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                    psd_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(psd_filename)
                    # --- CORRECTED SYNTAX ---
                    if not isinstance(psd_url, str) or not psd_url.startswith('http'):
                        print(f"Warn: Invalid PSD URL: {psd_url}")
                        psd_url=None
                        report_generation_errors.append("PSD URL")
                    else:
                        print(f"PSD URL: {psd_url}")
                    # --- END CORRECTION ---
                except Exception as e: print(f"ERR uploading PSD plot: {e}"); traceback.print_exc(); report_generation_errors.append("PSD Upload")
            else: print("Warn: PSD gen failed/invalid."); report_generation_errors.append("PSD Generation")

            # Step 5f: Generate PDF
            print("Step 5f: Generating PDF report..."); pdf = PDFReport(); pdf.alias_nb_pages()
            # Pass consistency_metrics to PDF builder
            _build_simple_pdf_content(pdf, prediction_data_for_report, stats_json, similarity_analysis_results, consistency_metrics_results, ts_img_data, psd_img_data, similarity_analysis_results.get('plot_base64') if similarity_analysis_results else None)
            print("Step 5f: Calling pdf.output()..."); pdf_output_bytearray = pdf.output(); pdf_filename = f"report_assets/{prediction_id}/report.pdf"
            if pdf_output_bytearray:
                pdf_output_as_bytes = bytes(pdf_output_bytearray); print(f"PDF generated ({len(pdf_output_as_bytes)} bytes). Uploading...")
                try:
                    supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=pdf_filename, file=pdf_output_as_bytes, file_options={"content-type": "application/pdf", "upsert": "true"})
                    pdf_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(pdf_filename)
                    # --- CORRECTED SYNTAX ---
                    if not isinstance(pdf_url, str) or not pdf_url.startswith('http'):
                        print(f"Warn: Invalid PDF URL: {pdf_url}")
                        pdf_url=None
                        report_generation_errors.append("PDF URL")
                    else:
                        print(f"Step 5f: Retrieved PDF URL: {pdf_url}")
                    # --- END CORRECTION ---
                except Exception as pdf_e: print(f"ERR uploading PDF: {pdf_e}"); traceback.print_exc(); report_generation_errors.append("PDF Upload")
            else: print(f"Warn: PDF generation failed."); report_generation_errors.append("PDF Generation")

            if not report_generation_errors: report_generation_status = "Completed"
            else: report_generation_status = f"Completed with errors ({', '.join(report_generation_errors)})"

        except Exception as report_gen_e:
            print(f"CRIT ERR Report Gen (Step 5): {report_gen_e}"); traceback.print_exc(); report_generation_status = f"Failed: {type(report_gen_e).__name__}"
            if 'ts_filename' in locals(): _cleanup_storage_on_error(REPORT_ASSET_BUCKET, ts_filename)
            if 'psd_filename' in locals(): _cleanup_storage_on_error(REPORT_ASSET_BUCKET, psd_filename)
            if similarity_plot_filename: _cleanup_storage_on_error(REPORT_ASSET_BUCKET, similarity_plot_filename)
            if 'pdf_filename' in locals(): _cleanup_storage_on_error(REPORT_ASSET_BUCKET, pdf_filename)

        # Step 6: Update DB (Final state)
        db_similarity_data = None
        if isinstance(similarity_analysis_results, dict) and not similarity_analysis_results.get('error'): db_similarity_data = {k: v for k, v in similarity_analysis_results.items() if k != 'plot_base64'}
        # Include consistency metrics and trial predictions in the final update
        update_data = {"stats_data": stats_json, "timeseries_plot_url": ts_url, "psd_plot_url": psd_url, "pdf_report_url": pdf_url, "report_generated_at": datetime.now(timezone.utc).isoformat(), "status": report_generation_status, "similarity_results": db_similarity_data, "similarity_plot_url": similarity_plot_url, "consistency_metrics": consistency_metrics_results, "trial_predictions": trial_predictions}
        print(f"Step 6: Updating DB for {prediction_id} with status '{report_generation_status}'...")
        try: update_payload_str = json.dumps(update_data, cls=NpEncoder, allow_nan=False); update_payload = json.loads(update_payload_str); supabase.table('predictions').update(update_payload).eq('id', prediction_id).execute(); print("DB update successful.")
        except Exception as db_update_e: print(f"Exception final DB update: {db_update_e}"); traceback.print_exc()

        # Step 7: Return Response
        print(f"Step 7: Process finished for {prediction_id}. Returning results.")
        return jsonify({"filename": file.filename, "prediction": prediction_label, "prediction_id": prediction_id})

    except Exception as e: # Main error handler
        print(f"ERROR in /api/predict endpoint: {e}"); traceback.print_exc()
        _cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path)
        if prediction_id:
            try: fail_status = f"Failed: {type(e).__name__}"; supabase.table('predictions').update({"status": fail_status }).eq('id', prediction_id).execute()
            except Exception as final_update_e: print(f"Failed to update fail status: {final_update_e}")
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500
    finally: # Cleanup temp files
        print("Executing finally block: Cleaning up temp files...")
        if os.path.exists(temp_filepath):
            try: os.remove(temp_filepath); print(f"Removed temp: {temp_filepath}")
            except Exception as e: print(f"Error removing temp {temp_filepath}: {e}")
        # Use ml_output_file_path defined earlier
        if 'ml_output_file_path' in locals() and os.path.exists(ml_output_file_path):
            try: os.remove(ml_output_file_path); print(f"Removed ML output: {ml_output_file_path}")
            except Exception as e: print(f"Error removing ML output {ml_output_file_path}: {e}")


# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.exists(ALZ_REF_PATH): print(f"WARNING: Alzheimer's ref file missing: {ALZ_REF_PATH}")
    if not os.path.exists(NORM_REF_PATH): print(f"WARNING: Normal ref file missing: {NORM_REF_PATH}")
    try: import supabase as sc; print(f"--- Supabase version: {getattr(sc, '__version__', 'unknown')} ---")
    except Exception as e: print(f"--- Supabase version check failed: {e} ---")
    print("--- Starting Flask Server ---")
    # Use environment variable for debug mode, default to False for safety
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"--- Debug Mode: {debug_mode} ---")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
