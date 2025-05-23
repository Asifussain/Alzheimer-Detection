# backend/app.py
# FINAL VERSION Incorporating Similarity Analysis, Channel Selection, Consistency Metrics & Syntax Fixes v3
# Technical PDF section revamped as per user request.
# PDF generation error in section_title fixed.

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

# --- Sanitization Function (from existing target app.py) ---
def sanitize_for_helvetica(text_input):
    if not isinstance(text_input, str):
        text_input = str(text_input)
    replacements = {
        '•': '-', '◦': '-', '’': "'", '‘': "'", '“': '"', '”': '"',
        '–': '-', '—': '-', '…': '...', '€': 'EUR', '£': 'GBP',
    }
    for uni_char, ascii_char in replacements.items():
        text_input = text_input.replace(uni_char, ascii_char)
    return "".join(c if ord(c) < 128 else "?" for c in text_input)

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
CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_URL", "*")}})

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY: raise ValueError("Supabase environment variables not set.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase client initialized.")

UPLOAD_FOLDER = 'uploads'
SIDDHI_FOLDER = 'SIDDHI'
OUTPUT_JSON_PATH = os.path.join(SIDDHI_FOLDER, 'output.json')
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ALZ_REF_PATH = os.path.join(BACKEND_DIR, 'feature_07.npy')
NORM_REF_PATH = os.path.join(BACKEND_DIR, 'feature_35.npy')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_FS = 128
RAW_EEG_BUCKET = 'eeg-data'
REPORT_ASSET_BUCKET = 'report-assets'

# --- Base PDF Report Class (incorporating methods from user's PDFReport) ---
class BasePDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "EEG Analysis Report" # Default, can be overridden
        self.primary_color = (52, 73, 94); self.secondary_color = (74, 144, 226)
        self.text_color_dark = (30, 30, 30); self.text_color_light = (100, 100, 100)
        self.text_color_normal = (0,0,0) # Black for normal text
        self.line_color = (220, 220, 220); self.card_bg_color = (248, 249, 250)
        self.highlight_color_alz = (220, 60, 60); self.highlight_color_norm = (60, 179, 113)
        self.warning_bg_color = (255, 243, 205); self.warning_text_color = (133, 100, 4)
        self.page_margin = 15
        self.set_auto_page_break(auto=True, margin=self.page_margin)
        self.set_line_width(0.2)

    def _is_bold_font(self): # From target app.py
        return 'B' in self.font_style

    # Sanitized cell, multi_cell, write methods (from target app.py)
    def cell(self, w, h=0, txt="", border=0, ln=0, align="", fill=False, link=""):
        txt_to_render = sanitize_for_helvetica(txt)
        super().cell(w, h, txt_to_render, border, ln, align, fill, link)

    def multi_cell(self, w, h, txt="", border=0, align="J", fill=False, max_line_height=0, new_x=XPos.START, new_y=YPos.TOP):
        txt_to_render = sanitize_for_helvetica(txt)
        if max_line_height == 0: max_line_height = h
        super().multi_cell(w, h, txt_to_render, border, align, fill, max_line_height=max_line_height, new_x=new_x, new_y=new_y)

    def write(self, h, txt="", link=""):
        txt_to_render = sanitize_for_helvetica(txt)
        super().write(h, txt_to_render, link)

    def header(self): # Using header from user's PDFReport (simpler, for technical)
        try:
            self.set_font('Helvetica', 'B', 15)
            title = self.report_title # Use the instance's report_title
            title_w = self.get_string_width(title) + 6
            doc_w = self.w
            self.set_x((doc_w - title_w) / 2)
            self.set_text_color(74, 144, 226) # Primary Blue for header text
            self.cell(title_w, 10, title, border=0, align='C', ln=1) # Use ln=1
            self.set_text_color(0) # Reset color
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(8)
        except Exception as e: print(f"PDF Header Error: {e}")

    def footer(self): # Using footer from user's PDFReport
        try:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
            self.set_text_color(0) # Reset color
        except Exception as e: print(f"PDF Footer Error: {e}")

    # section_title from user's PDFReport (teal background) - CORRECTED
    def section_title(self, title):
        try:
            self.set_font('Helvetica', 'B', 13)
            self.set_fill_color(80, 227, 194) # Approx --accent-teal
            self.set_text_color(10, 15, 26)   # Approx --background-start (dark text)
            # Corrected call: Use ln=1. new_x and new_y are not valid for cell.
            self.cell(0, 8, " " + sanitize_for_helvetica(title), border='B', align='L', fill=True, ln=1)
            self.set_text_color(0) # Reset text color
            self.ln(6) # Additional space after title block
        except Exception as e: print(f"PDF Section Title Error: {e}")


    # key_value_pair from user's PDFReport
    def key_value_pair(self, key, value, key_width=45):
        try:
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(50, 50, 50)
            key_start_y = self.get_y()
            # multi_cell is fine with new_x, new_y
            self.multi_cell(key_width, 6, sanitize_for_helvetica(str(key))+":", align='L', new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=self.font_size)
            self.set_y(key_start_y)
            self.set_x(self.l_margin + key_width + 2)
            self.set_font('Helvetica', '', 10)
            self.set_text_color(0) # Black for values
            self.multi_cell(0, 6, sanitize_for_helvetica(str(value)), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(1) # Minimal gap between pairs
        except Exception as e: print(f"PDF Key/Value Error: {e}")

    # write_multiline from user's PDFReport, added to BasePDFReport
    def write_multiline(self, text, height=5, indent=5):
        try:
            self.set_font('Helvetica', '', 10)
            self.set_text_color(80, 80, 80)
            self.set_left_margin(self.l_margin + indent)
            # multi_cell is fine with new_x, new_y
            self.multi_cell(0, height, sanitize_for_helvetica(text), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_left_margin(self.l_margin)
            self.ln(height / 2)
            self.set_text_color(0)
        except Exception as e: print(f"PDF Multiline Error: {e}")

    # metric_card from user's PDFReport, added to BasePDFReport
    def metric_card(self, title, value, unit="", description=""):
        try:
            start_x = self.get_x()
            start_y = self.get_y()
            card_width = (self.w - self.l_margin - self.r_margin - 5) / 2
            card_height = 25

            self.set_fill_color(240, 245, 250)
            self.set_draw_color(80, 227, 194)
            self.set_line_width(0.3)
            self.rect(start_x, start_y, card_width, card_height, 'DF')

            self.set_xy(start_x + 3, start_y + 3)
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(80, 80, 80)
            self.cell(card_width - 6, 5, sanitize_for_helvetica(title.upper()), align='L')

            self.set_xy(start_x + 3, start_y + 9)
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(74, 144, 226)
            value_str = f"{sanitize_for_helvetica(str(value))}{sanitize_for_helvetica(str(unit))}"
            self.cell(card_width - 6, 8, value_str, align='R')

            if description:
                self.set_xy(start_x + 3, start_y + 18)
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(100, 100, 100)
                self.cell(card_width - 6, 5, sanitize_for_helvetica(description), align='L')

            self.set_y(start_y)
            self.set_x(start_x + card_width + 5)
            self.set_text_color(0)
            self.set_line_width(0.2)
        except Exception as e: print(f"PDF Metric Card Error: {e}")
        
    def write_paragraph(self, text, height=4, indent=0, font_style='', font_size=8.5, text_color=None, bullet_char_override=None):
        try:
             self.set_font('Helvetica', font_style, font_size)
             text_c = text_color if text_color else self.text_color_dark
             self.set_text_color(*text_c)
             current_x_start = self.l_margin + indent; self.set_x(current_x_start)
             sanitized_text = sanitize_for_helvetica(text)
             if bullet_char_override:
                 safe_bullet = sanitize_for_helvetica(bullet_char_override)
                 o_ff, o_fs, o_fsty = self.font_family, self.font_size, self.font_style
                 self.set_font('Helvetica', 'B', font_size);
                 self.cell(self.get_string_width(safe_bullet) + 0.5, height, safe_bullet)
                 self.set_font(o_ff, o_fsty, o_fs);
                 self.set_x(current_x_start + self.get_string_width(safe_bullet) + 1.5)
                 self.multi_cell(self.w - self.get_x() - self.r_margin, height, sanitized_text, align='L',new_x=XPos.LMARGIN, new_y=YPos.NEXT)
             else: self.multi_cell(0, height, sanitized_text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
             self.ln(height / 4); self.set_text_color(*self.text_color_normal)
        except Exception as e: print(f"PDF write_paragraph Error: {e}")

    def add_image_section(self, title, image_data_base64):
        image_height_estimate = 70
        title_height_estimate = 10 if title else 0
        if self.get_y() + title_height_estimate + image_height_estimate > self.h - self.b_margin:
            self.add_page()
        if title:
            self.set_font('Helvetica', 'B', 10); self.set_text_color(*self.text_color_dark)
            self.cell(0, 8, sanitize_for_helvetica(title), ln=1, align='L')
            self.ln(1)
        if image_data_base64 and isinstance(image_data_base64, str) and image_data_base64.startswith('data:image/png;base64,'):
            try:
                img_bytes = base64.b64decode(image_data_base64.split(',')[1]); img_file = io.BytesIO(img_bytes)
                page_content_width = self.w - 2 * self.page_margin
                img_display_width = page_content_width * 0.95
                x_pos = self.page_margin + (page_content_width - img_display_width) / 2
                self.image(img_file, x=x_pos, w=img_display_width); img_file.close(); self.ln(2)
            except Exception as e:
                self.write_paragraph(sanitize_for_helvetica(f"(Error embedding image '{title}': {str(e)[:50]})"), font_style='I')
                print(f"PDF Image Embed Error for {title}: {e}")
        else:
            if title: self.write_paragraph(sanitize_for_helvetica("(Image data not available)"), font_style='I', indent=5)
        self.ln(4)

    def add_explanation_box(self, title, text_lines, icon_char="[i]", bg_color=None, title_color=None, text_color_override=None, font_size_text=9, line_h=5):
        self.ln(1);
        bg_c = bg_color if bg_color else self.card_bg_color
        title_c = title_color if title_color else self.primary_color
        text_c = text_color_override if text_color_override else self.text_color_dark
        safe_icon = sanitize_for_helvetica(icon_char)
        self.set_font('Helvetica', 'B', 10); self.set_text_color(*title_c)
        title_text = f"{safe_icon} {sanitize_for_helvetica(title)}" if safe_icon else sanitize_for_helvetica(title)
        self.multi_cell(0, 7, title_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        y_before_text_content = self.get_y()
        estimated_box_height = 3
        for item in text_lines: estimated_box_height += line_h + 1
        self.set_fill_color(*bg_c); self.set_draw_color(*self.line_color)
        self.rect(self.l_margin, y_before_text_content, self.w - self.l_margin - self.r_margin, estimated_box_height, 'DF')
        self.set_y(y_before_text_content + 1.5)
        for item in text_lines:
            self.set_x(self.l_margin + 2)
            is_list = isinstance(item, tuple) and item[0] == "bullet"
            txt = item[1] if is_list else item
            is_sub_list = isinstance(txt, tuple) and txt[0] == "sub_bullet"
            final_txt = txt[1] if is_sub_list else txt
            if is_list:
                bullet_x = self.get_x()
                self.set_font('Helvetica', 'B', font_size_text); self.set_text_color(*text_c)
                bullet_char_to_use = "-"
                if is_sub_list: self.set_x(bullet_x + 5); bullet_char_to_use = "-"
                self.cell(5, line_h, bullet_char_to_use) # This cell call is fine
                self.set_x(bullet_x + 5 + (5 if is_sub_list else 0))
            bold_parts = sanitize_for_helvetica(final_txt).split("**")
            for i, part_text in enumerate(bold_parts):
                is_bold = i % 2 == 1
                self.set_font('Helvetica', 'B' if is_bold else '', font_size_text)
                self.set_text_color(*(self.primary_color if is_bold else text_c))
                self.write(line_h, part_text) # This write call is fine
            self.ln(line_h + 0.5)
        self.set_y(y_before_text_content + estimated_box_height); self.ln(3)
        self.set_text_color(*self.text_color_normal)

# --- Technical PDF Report Class ---
class TechnicalPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Technical EEG Analysis Report"

# --- Patient PDF Report Class ---
class PatientPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Your AI EEG Pattern Report"

# --- Helper Functions ---
def _cleanup_storage_on_error(bucket_name, path):
    try:
        if bucket_name and path: supabase.storage.from_(bucket_name).remove([path])
    except Exception as e: print(f"Error during storage cleanup: {e}")

def get_prediction_and_eeg(prediction_id):
    print(f"Helper: Fetching record ID: {prediction_id}"); prediction_rec = None
    try:
        prediction_res = supabase.table('predictions').select('*').eq('id', prediction_id).maybe_single().execute()
        if not prediction_res.data: return None, None, "Prediction record not found"
        prediction_rec = prediction_res.data; eeg_url_path = prediction_rec.get('eeg_data_url')
        if not eeg_url_path: return prediction_rec, None, "EEG data URL missing from record"
        eeg_file_response = supabase.storage.from_(RAW_EEG_BUCKET).download(eeg_url_path)
        if not isinstance(eeg_file_response, bytes):
             return prediction_rec, None, f"Failed to download raw EEG file from {eeg_url_path}: {getattr(eeg_file_response, 'message', str(eeg_file_response))}"
        with io.BytesIO(eeg_file_response) as f: eeg_data = np.load(f, allow_pickle=True)
        if eeg_data.ndim == 3: eeg_data = eeg_data[0, :, :]
        elif eeg_data.ndim != 2: raise ValueError(f"Unsupported EEG data dimension: {eeg_data.ndim}")
        if eeg_data.shape[0] < eeg_data.shape[1]: eeg_data = eeg_data.T
        if eeg_data.ndim != 2: raise ValueError(f"Final EEG data is not 2D: {eeg_data.shape}")
        return prediction_rec, eeg_data.astype(np.double), None
    except Exception as e:
        print(f"Helper Error for prediction ID {prediction_id}: {e}"); traceback.print_exc()
        return (prediction_rec if prediction_rec else None), None, f"Error accessing/processing data: {str(e)}"

def run_model(filepath_to_process):
    print(f"Executing ML model for: {filepath_to_process}"); current_dir = os.path.dirname(os.path.abspath(__file__))
    siddhi_path = os.path.join(current_dir, SIDDHI_FOLDER); absolute_filepath_ml = os.path.abspath(filepath_to_process)
    expected_output_json = os.path.join(siddhi_path, 'output.json')
    if not os.path.isdir(siddhi_path): raise FileNotFoundError(f"SIDDHI directory not found at: {siddhi_path}")
    if not os.path.isfile(absolute_filepath_ml): raise FileNotFoundError(f"Input EEG file not found at: {absolute_filepath_ml}")
    if os.path.exists(expected_output_json):
        try: os.remove(expected_output_json); print(f"Removed existing ML output file: {expected_output_json}")
        except Exception as rem_e: print(f"Warning: Could not remove existing {expected_output_json}: {rem_e}")
    original_cwd = os.getcwd(); print(f"Temporarily changing CWD to: {siddhi_path}"); os.chdir(siddhi_path)
    try:
        cmd = ['python', 'run.py', '--task_name', 'classification', '--is_training', '0', '--model_id', 'ADSZ-Indep', '--model', 'ADformer', '--data', 'ADSZIndep', '--e_layers', '6', '--batch_size', '1', '--d_model', '128', '--d_ff', '256', '--enc_in', '19', '--num_class', '2', '--seq_len', '128', '--input_file', absolute_filepath_ml, '--use_gpu', 'False', '--features', 'M', '--label_len', '48', '--pred_len', '96', '--n_heads', '8', '--d_layers', '1', '--factor', '1', '--embed', 'timeF', '--des', "'Exp'",]
        print(f"Running ML command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=300)
        print(f"ML Model STDOUT:\n{result.stdout}");
        if result.stderr: print(f"ML Model STDERR:\n{result.stderr}")
        if not os.path.exists('output.json'): raise FileNotFoundError(f"'output.json' not created in {siddhi_path} after script execution.")
        print("ML model script executed successfully.")
    except subprocess.CalledProcessError as proc_error: print(f"ML script execution failed (Return Code {proc_error.returncode})\n--- ML STDERR ---\n{proc_error.stderr}\n--- End ML STDERR ---"); raise proc_error
    except subprocess.TimeoutExpired: print("ML script execution timed out."); raise TimeoutError("ML model execution timed out.")
    except FileNotFoundError as fnf_error: print(f"File system error during ML execution: {fnf_error}"); raise
    except Exception as e: print(f"An unexpected error occurred in run_model: {e}"); traceback.print_exc(); raise
    finally: print(f"Changing CWD back to original: {original_cwd}"); os.chdir(original_cwd)


# --- Revamped Technical PDF Content Builder ---
def _build_technical_pdf_report_content(pdf: TechnicalPDFReport, prediction_data, stats_data, similarity_data, consistency_metrics, ts_img_data, psd_img_data, similarity_plot_data):
    """Builds the PDF content using the logic from the user's _build_simple_pdf_content."""
    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    
    try:
        pdf.add_page()
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

        if consistency_metrics and not consistency_metrics.get('error') and consistency_metrics.get('num_trials', 0) > 1:
            pdf.set_font('Helvetica', 'B', 11); pdf.cell(0, 6, "Internal Consistency Metrics:", ln=1); pdf.ln(2) # Use ln=1 for cell
            pdf.set_font('Helvetica', 'I', 9); pdf.set_text_color(100, 100, 100); pdf.cell(0, 5, "(Compares segment predictions against the overall prediction for this file)", ln=1); pdf.ln(4); pdf.set_text_color(0) # Use ln=1 for cell
            metrics = consistency_metrics
            
            current_y_after_card = pdf.get_y()
            pdf.metric_card("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}", unit="%", description="Overall segment agreement")
            pdf.set_y(current_y_after_card)
            pdf.metric_card("Precision (Alz)", f"{metrics.get('precision', 0):.3f}", unit="", description="Alz predictions correct")
            pdf.ln(28)

            current_y_after_card = pdf.get_y()
            pdf.metric_card("Sensitivity (Alz)", f"{metrics.get('recall_sensitivity', 0):.3f}", unit="", description="Alz segments found")
            pdf.set_y(current_y_after_card)
            pdf.metric_card("Specificity (Norm)", f"{metrics.get('specificity', 0):.3f}", unit="", description="Normal segments found")
            pdf.ln(28)

            current_y_after_card = pdf.get_y()
            pdf.metric_card("F1-Score (Alz)", f"{metrics.get('f1_score', 0):.3f}", unit="", description="Precision/Sensitivity balance")
            pdf.set_y(current_y_after_card)
            pdf.metric_card("Trials Analyzed", f"{metrics.get('num_trials', 'N/A')}", unit="", description="Segments in file")
            pdf.ln(28)
            
            pdf.set_font('Helvetica', '', 9); pdf.set_text_color(100, 100, 100)
            conf_matrix_str = (f"(Ref Label: {metrics.get('majority_label_used_as_reference', '?')}) " f"TP:{metrics.get('true_positives','?')} | TN:{metrics.get('true_negatives','?')} | FP:{metrics.get('false_positives','?')} | FN:{metrics.get('false_negatives','?')}")
            pdf.cell(0, 5, conf_matrix_str, align='C', ln=1) ; pdf.set_text_color(0) # Use ln=1 for cell
        elif consistency_metrics and consistency_metrics.get('message'):
            pdf.set_font('Helvetica', 'I', 10); pdf.write_multiline(f"({consistency_metrics['message']})", indent=5)
        else:
            pdf.set_font('Helvetica', 'I', 10); pdf.write_multiline("(Internal consistency metrics not calculated.)", indent=5)
        pdf.ln(5)

        if pdf.get_y() > pdf.h - 80 : pdf.add_page()

        pdf.section_title("Signal Shape Similarity Analysis (DTW)")
        if similarity_data and not similarity_data.get('error'):
            pdf.write_multiline(similarity_data.get('interpretation', 'No similarity interpretation available.'), indent=5)
            pdf.ln(2)
            if similarity_plot_data and isinstance(similarity_plot_data, str) and similarity_plot_data.startswith('data:image/png;base64,'):
                plotted_ch = similarity_data.get('plotted_channel_index', None); plot_title = f"Channel {plotted_ch + 1} Comparison Plot:" if plotted_ch is not None else "Comparison Plot:"
                pdf.set_font("Helvetica",'B',11); pdf.cell(0, 8, plot_title, ln=1); # Use ln=1 for cell
                try:
                    img_bytes = base64.b64decode(similarity_plot_data.split(',')[1]); img_file = io.BytesIO(img_bytes)
                    img_width_mm = page_width * 0.9; x_pos = pdf.l_margin + (page_width - img_width_mm) / 2
                    pdf.image(img_file, x=x_pos, w=img_width_mm); img_file.close(); pdf.ln(5)
                except Exception as e:
                    pdf.set_font("Helvetica",'I',10); pdf.set_text_color(255,0,0); pdf.cell(0,10,f"(Err embedding Sim Plot: {e})", ln=1); pdf.set_text_color(0); print(f"PDF Sim Embed Err: {e}") # Use ln=1 for cell
            else:
                pdf.set_font("Helvetica",'I',10); pdf.cell(0,10,"(Similarity plot not generated or invalid)", ln=1) # Use ln=1 for cell
        else:
            pdf.set_font("Helvetica",'I',10); err_msg = similarity_data.get('error', 'Unknown') if similarity_data else 'N/A'
            pdf.write_multiline(f"(Similarity Analysis Error: {err_msg})", indent=5)
        pdf.ln(5)

        if pdf.get_y() > pdf.h - 60 : pdf.add_page()

        pdf.section_title("Descriptive Statistics")
        if stats_data and not stats_data.get('error'):
            pdf.set_font("Helvetica",'B',11); pdf.cell(0,6,"Avg Relative Band Power (%):", ln=1); pdf.ln(1); pdf.set_font("Helvetica",size=10); avg_power = stats_data.get('avg_band_power',{}); band_found=False # Use ln=1 for cell
            if avg_power:
                for band, powers in avg_power.items():
                    rel_power = powers.get('relative', None); band_found |= (rel_power is not None)
                    rel_power_str = f"{rel_power * 100:.2f}%" if isinstance(rel_power, (int, float)) else 'N/A'
                    pdf.cell(10); pdf.cell(0,5,f"- {band.capitalize()}: {rel_power_str}", ln=1) # Use ln=1 for cell
            if not band_found:
                pdf.set_font("Helvetica",'I',10); pdf.cell(10); pdf.cell(0,5,"(No band power data)", ln=1) # Use ln=1 for cell
            pdf.ln(5)
        else:
            pdf.set_font("Helvetica",'I',10); err_msg = stats_data.get('error', 'Unknown') if stats_data else 'N/A'
            pdf.write_multiline(f"(Statistics Error: {err_msg})", indent=5)
        pdf.ln(5)

        if pdf.get_y() > pdf.h - 100 : pdf.add_page()

        pdf.section_title("Standard Visualizations")
        pdf.set_font("Helvetica",'B',12); pdf.cell(0,8,"Stacked Time Series", ln=1); pdf.ln(2) # Use ln=1 for cell
        if ts_img_data and isinstance(ts_img_data, str) and ts_img_data.startswith('data:image/png;base64,'):
            try:
                img_bytes=base64.b64decode(ts_img_data.split(',')[1]); img_file=io.BytesIO(img_bytes)
                pdf.image(img_file, x=pdf.l_margin, w=page_width); img_file.close(); pdf.ln(5)
            except Exception as e:
                pdf.set_font("Helvetica",'I',10); pdf.set_text_color(255,0,0); pdf.cell(0,10,f"(Err embedding TS Plot: {e})", ln=1); pdf.set_text_color(0); print(f"PDF TS Embed Err: {e}") # Use ln=1 for cell
        else:
            pdf.set_font("Helvetica",'I',10); pdf.cell(0,10,"(TS plot not generated or invalid)", ln=1) # Use ln=1 for cell
        pdf.ln(10)

        if pdf.get_y() > pdf.h - 100 : pdf.add_page()

        pdf.set_font("Helvetica",'B',12); pdf.cell(0,8,"Average Power Spectral Density (PSD)", ln=1); pdf.ln(2) # Use ln=1 for cell
        if psd_img_data and isinstance(psd_img_data, str) and psd_img_data.startswith('data:image/png;base64,'):
            try:
                img_bytes=base64.b64decode(psd_img_data.split(',')[1]); img_file=io.BytesIO(img_bytes)
                img_width_mm=page_width*0.9; x_pos=pdf.l_margin+(page_width-img_width_mm)/2
                pdf.image(img_file, x=x_pos, w=img_width_mm); img_file.close(); pdf.ln(5)
            except Exception as e:
                pdf.set_font("Helvetica",'I',10); pdf.set_text_color(255,0,0); pdf.cell(0,10,f"(Err embedding PSD Plot: {e})", ln=1); pdf.set_text_color(0); print(f"PDF PSD Embed Err: {e}") # Use ln=1 for cell
        else:
            pdf.set_font("Helvetica",'I',10); pdf.cell(0,10,"(PSD plot not generated or invalid)", ln=1) # Use ln=1 for cell

    except Exception as pdf_build_e:
        print(f"Error building PDF content: {pdf_build_e}"); traceback.print_exc()
        try:
            if pdf.page_no() == 0: pdf.add_page()
            elif pdf.get_y() > pdf.h - 30 : pdf.add_page()
            pdf.set_font("Helvetica",'B',12); pdf.set_text_color(255,0,0)
            pdf.multi_cell(0,10,f"Critical Error Building PDF Content:\n{sanitize_for_helvetica(str(pdf_build_e))}",align='C')
            pdf.set_text_color(0)
        except Exception as pdf_err_fallback:
            print(f"Fallback error writing to PDF failed: {pdf_err_fallback}")


# --- Patient PDF Content Builder (from target app.py, uses BasePDFReport methods) ---
def _build_patient_pdf_report_content(pdf: PatientPDFReport, prediction_data, similarity_data, consistency_metrics, similarity_plot_data):
    pdf.add_page()
    pdf.section_title("Analysis Summary")
    created_at_str = pd.to_datetime(prediction_data.get('created_at')).strftime('%B %d, %Y') if prediction_data.get('created_at') else 'N/A'
    pdf.key_value_pair("File Analyzed", prediction_data.get('filename', 'N/A')) # Uses BasePDFReport's k-v
    pdf.key_value_pair("Date of Analysis", created_at_str); pdf.ln(6)

    pdf.add_explanation_box(
        "About This Report",
        ["This report uses Artificial Intelligence (AI) to look for specific patterns in your brainwave (EEG) activity.",
         "The AI compares your EEG patterns to those it has learned from many examples.",
         ("bullet", "Important: This is an informational tool to help your doctor. **It is not a medical diagnosis.** Please discuss these results with them.")
        ], icon_char="[i]", font_size_text=9.5, line_h=5.5)

    pdf.section_title("AI's Main Finding: Pattern Assessment")
    prediction_label = prediction_data.get('prediction', 'Not Determined')
    pred_display_text = "Pattern assessment inconclusive"; pred_color = pdf.text_color_dark
    if prediction_label == "Alzheimer's": pred_display_text = "Patterns Suggestive of Alzheimer's Characteristics"; pred_color = pdf.highlight_color_alz
    elif prediction_label == "Normal": pred_display_text = "Normal Brainwave Patterns Observed"; pred_color = pdf.highlight_color_norm
    pdf.write_paragraph("The AI analyzed your EEG and found that the patterns are most similar to:", font_size=10, height=5.5)
    pdf.set_font('Helvetica', 'B', 14); pdf.set_text_color(*pred_color)
    pdf.multi_cell(0, 8, pred_display_text, border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.set_text_color(*pdf.text_color_normal); pdf.ln(3)
    probabilities = prediction_data.get('probabilities'); confidence_text = "Confidence score not available."
    if isinstance(probabilities, list) and len(probabilities) == 2:
        try: conf_val = probabilities[1]*100 if prediction_label == "Alzheimer's" else probabilities[0]*100; confidence_text = f"The AI is **{conf_val:.0f}%** confident that the patterns it found align with the finding above (based on the first segment of your EEG data)."
        except: pass
    pdf.add_explanation_box("AI's Confidence Level", [confidence_text], icon_char="[T]", font_size_text=9.5, line_h=5.5)

    pdf.section_title("AI's Internal Consistency Check")
    if consistency_metrics and not consistency_metrics.get('error') and isinstance(consistency_metrics.get('num_trials'), int) and consistency_metrics.get('num_trials', 0) > 0:
        num_segments = consistency_metrics.get('num_trials', 'multiple')
        intro_text = [f"To double-check its findings, the AI looked at your EEG data in **{num_segments} smaller pieces** (segments). This helps assess how stable the AI's finding was across your entire recording. Here's a simple breakdown of these checks:"]
        metric_items_for_patient = []
        accuracy_val = f"{consistency_metrics.get('accuracy', 0) * 100:.0f}%"
        metric_items_for_patient.append(("bullet", f"**Overall Consistency (Accuracy):** {accuracy_val}. This shows how often the AI's checks on the small pieces matched its main finding for your whole EEG sample."))
        if prediction_label == "Alzheimer's":
            sensitivity_val = f"{consistency_metrics.get('recall_sensitivity', 0) * 100:.0f}%"; precision_val = f"{consistency_metrics.get('precision', 0) * 100:.0f}%"; f1_val = f"{consistency_metrics.get('f1_score', 0):.2f}"
            metric_items_for_patient.extend([
                ("bullet", f"**Finding Alzheimer's-like Patterns (Sensitivity):** {sensitivity_val}. If segments showed Alzheimer's-like patterns (based on the main finding), the AI found them this often."),
                ("bullet", f"**Confirming Alzheimer's-like Patterns (Precision):** {precision_val}. When the AI said a segment was Alzheimer's-like, it was consistent with the main finding this often."),
                ("bullet", f"**Balanced Score for Alzheimer's Patterns (F1-Score):** {f1_val}. A combined score (0 to 1, higher is better) reflecting how well the AI balanced finding and confirming these patterns.")
            ])
        else:
            specificity_val = f"{consistency_metrics.get('specificity', 0) * 100:.0f}%"
            metric_items_for_patient.append(("bullet", f"**Finding Normal Patterns (Specificity):** {specificity_val}. If Normal patterns were present in segments (based on the main finding), the AI correctly identified them this often."))
        metric_items_for_patient.extend([("bullet", f"**Number of Segments Checked:** {num_segments}."), "Higher percentages and scores in these checks generally suggest the AI was consistent in what it observed throughout your EEG sample."])
        pdf.add_explanation_box("Understanding AI's Consistency", intro_text + metric_items_for_patient, icon_char="[M]", bg_color=(230,250,230), title_color=pdf.highlight_color_norm, font_size_text=9, line_h=5)
    else: pdf.write_paragraph("(Detailed internal consistency checks were not applicable or did not yield specific metrics for this sample.)", font_style='I', indent=3, font_size=9)
    pdf.ln(5)

    if pdf.get_y() > pdf.h - 120 : pdf.add_page()
    if similarity_data and not similarity_data.get('error') and similarity_plot_data:
        plotted_ch_idx = similarity_data.get('plotted_channel_index')
        plot_title = f"Comparing Your Brainwave Shape (from Channel {plotted_ch_idx + 1 if plotted_ch_idx is not None else 'Selected'})"
        pdf.add_image_section(plot_title, similarity_plot_data) # uses BasePDFReport.add_image_section
        sim_interp_text_main = "The AI found that your sample's brainwave shapes showed "
        overall_sim = similarity_data.get('overall_similarity', '')
        if "Higher Similarity to Alzheimer's Pattern" in overall_sim: sim_interp_text_main += "**more resemblance to the Alzheimer's reference patterns**."
        elif "Higher Similarity to Normal Pattern" in overall_sim: sim_interp_text_main += "**more resemblance to the Normal reference patterns**."
        else: sim_interp_text_main += "a mixed or inconclusive resemblance when compared to the reference patterns."
        sim_interpretation_from_data = similarity_data.get('interpretation', "").split("Disclaimer:")[0].replace("Similarity Analysis (DTW):", "").replace("Overall Assessment:", "").strip()
        pdf.add_explanation_box("What This Graph Shows", [sim_interp_text_main, f"More Details: \"{sim_interpretation_from_data}\""], icon_char="[D]", font_size_text=9.5, line_h=5.5)
    else:
        pdf.section_title("Comparing Your Brainwave Shape"); pdf.write_paragraph("(The brainwave shape comparison graph is not available for this report.)", font_style='I')
    pdf.ln(6)

    pdf.section_title("Important Information & Your Next Steps")
    pdf.add_explanation_box(
        "Please Discuss This Report With Your Doctor",
        [
            "This AI report is an informational tool based on EEG patterns. **It is NOT a medical diagnosis.**",
            "Only a qualified healthcare professional can diagnose medical conditions. They will consider this report along with your full medical history and other tests.",
            ("bullet", "Key Takeaway: The AI analysis suggests your EEG patterns are most similar to **" + sanitize_for_helvetica(pred_display_text) + "**."),
            ("bullet", "**Recommended Next Steps:**"),
            ("bullet", ("sub_bullet", "Share this entire report with your doctor or a neurologist.")),
            ("bullet", ("sub_bullet", "Discuss any health concerns and follow their medical advice."))
        ],
        icon_char="[!]", bg_color=pdf.warning_bg_color, title_color=(106, 63, 20), text_color_override=(85,60,10), font_size_text=9.5, line_h=5.5
    )


# --- Predict Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files.get('file'); user_id = request.form.get('user_id')
    try: channel_index_str = request.form.get('channel_index', '0'); channel_index_for_plot = int(channel_index_str); assert 0 <= channel_index_for_plot <= 18
    except (ValueError, TypeError, AssertionError): channel_index_for_plot = 0; print(f"Warning: Invalid channel index '{channel_index_str}'. Defaulting to 0.")

    if not file or not user_id: return jsonify({'error': "Missing 'file' or 'user_id'"}), 400
    if not file.filename or not file.filename.lower().endswith('.npy'): return jsonify({'error': 'Invalid/Missing filename or type (.npy required).'}), 400

    filename_base, file_extension = os.path.splitext(file.filename); unique_id = str(uuid.uuid4())
    save_filename = f"{filename_base}_{unique_id}{file_extension}"
    absolute_temp_filepath = os.path.abspath(os.path.join(UPLOAD_FOLDER, save_filename))
    raw_eeg_storage_path = f'raw_eeg/{user_id}/{save_filename}'; prediction_id = None
    report_generation_errors = []; similarity_analysis_results = None; consistency_metrics_results = None
    ts_img_data, psd_img_data, similarity_plot_base64_data = None, None, None
    ts_url, psd_url, similarity_plot_url, technical_pdf_url, patient_pdf_url = None, None, None, None, None
    asset_prefix = ""

    try:
        print(f"Step 1/2: Processing '{file.filename}'...");
        os.makedirs(os.path.dirname(absolute_temp_filepath), exist_ok=True)
        file.save(absolute_temp_filepath); print(f"File saved to: {absolute_temp_filepath}")
        with open(absolute_temp_filepath, 'rb') as f_upload:
            supabase.storage.from_(RAW_EEG_BUCKET).upload(path=raw_eeg_storage_path, file=f_upload, file_options={"content-type": "application/octet-stream", "upsert": "false"})
        print(f"Step 3: Running ML model on {absolute_temp_filepath}..."); run_model(absolute_temp_filepath)
        ml_output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_JSON_PATH)
        if not os.path.exists(ml_output_file_path): raise FileNotFoundError(f"ML output missing: {ml_output_file_path}")
        with open(ml_output_file_path, 'r') as f: output_data = json.load(f)
        prediction_label = "Alzheimer's" if output_data.get('majority_prediction') == 1 else "Normal"; probabilities = output_data.get('probabilities'); consistency_metrics_results = output_data.get('consistency_metrics'); trial_predictions = output_data.get('trial_predictions')
        insert_data = {"user_id": user_id, "filename": file.filename, "prediction": prediction_label, "eeg_data_url": raw_eeg_storage_path, "probabilities": probabilities, "status": "Processing Report Assets", "trial_predictions": trial_predictions, "consistency_metrics": consistency_metrics_results}
        print(f"Step 4: Inserting prediction record..."); insert_payload = json.loads(json.dumps(insert_data, cls=NpEncoder, allow_nan=False)); insert_res = supabase.table('predictions').insert(insert_payload).execute()
        if insert_res.data and len(insert_res.data) > 0: prediction_id = insert_res.data[0].get('id'); print(f"DB Insert successful. ID: {prediction_id}")
        else: _cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path); raise Exception(f"DB insert failed: {insert_res}")
        
        asset_prefix = f"report_assets/{prediction_id}"

        print(f"--- Step 5: Generating Report Assets for ID: {prediction_id} ---")
        prediction_data_for_report, eeg_data, error_msg = get_prediction_and_eeg(prediction_id)
        if error_msg or eeg_data is None: raise Exception(f"Cannot load data for report generation: {error_msg or 'No EEG data'}")
        
        similarity_analysis_results = run_similarity_analysis(absolute_temp_filepath, ALZ_REF_PATH, NORM_REF_PATH, channel_index_for_plot)
        similarity_plot_base64_data = similarity_analysis_results.get('plot_base64') if isinstance(similarity_analysis_results, dict) else None
        
        stats_json = generate_descriptive_stats(eeg_data, DEFAULT_FS); ts_img_data = generate_stacked_timeseries_image(eeg_data, DEFAULT_FS); psd_img_data = generate_average_psd_image(eeg_data, DEFAULT_FS)
        
        if similarity_plot_base64_data:
            try:
                sim_plot_filename = f"{asset_prefix}/similarity_plot_ch{channel_index_for_plot + 1}.png"; sim_plot_bytes = base64.b64decode(similarity_plot_base64_data.split(',')[1])
                supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=sim_plot_filename, file=sim_plot_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                similarity_plot_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(sim_plot_filename);
                if not isinstance(similarity_plot_url, str) or not similarity_plot_url.startswith('http'): similarity_plot_url=None; report_generation_errors.append("SimPlotURL")
            except Exception as e: print(f"ERR SimPlotUpload: {e}"); report_generation_errors.append("SimPlotUpload")
        if ts_img_data:
            try:
                ts_filename = f"{asset_prefix}/timeseries.png"; ts_bytes = base64.b64decode(ts_img_data.split(',')[1])
                supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=ts_filename, file=ts_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                ts_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(ts_filename);
                if not isinstance(ts_url, str) or not ts_url.startswith('http'): ts_url=None; report_generation_errors.append("TSPlotURL")
            except Exception as e: print(f"ERR TSPlotUpload: {e}"); report_generation_errors.append("TSPlotUpload")
        if psd_img_data:
            try:
                psd_filename = f"{asset_prefix}/psd.png"; psd_bytes = base64.b64decode(psd_img_data.split(',')[1])
                supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=psd_filename, file=psd_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                psd_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(psd_filename);
                if not isinstance(psd_url, str) or not psd_url.startswith('http'): psd_url=None; report_generation_errors.append("PSDPlotURL")
            except Exception as e: print(f"ERR PSDPlotUpload: {e}"); report_generation_errors.append("PSDPlotUpload")

        # Generate Technical PDF (revamped)
        print("Generating Technical PDF report..."); tech_pdf = TechnicalPDFReport(); tech_pdf.alias_nb_pages()
        # Updated call to include similarity_plot_base64_data
        _build_technical_pdf_report_content(tech_pdf, prediction_data_for_report, stats_json, similarity_analysis_results, consistency_metrics_results, ts_img_data, psd_img_data, similarity_plot_base64_data)
        tech_pdf_bytes = bytes(tech_pdf.output()); technical_pdf_filename = f"{asset_prefix}/technical_report.pdf"
        try:
            supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=technical_pdf_filename, file=tech_pdf_bytes, file_options={"content-type": "application/pdf", "upsert": "true"})
            technical_pdf_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(technical_pdf_filename)
            if not isinstance(technical_pdf_url, str) or not technical_pdf_url.startswith('http'): technical_pdf_url=None; report_generation_errors.append("TechPDFURL")
        except Exception as e: print(f"ERR TechPDFUpload: {e}"); report_generation_errors.append("TechPDFUpload")

        # Generate Patient PDF
        print("Generating Patient PDF report..."); patient_pdf = PatientPDFReport(); patient_pdf.alias_nb_pages()
        _build_patient_pdf_report_content(patient_pdf, prediction_data_for_report, similarity_analysis_results, consistency_metrics_results, similarity_plot_base64_data)
        patient_pdf_bytes = bytes(patient_pdf.output()); patient_pdf_filename = f"{asset_prefix}/patient_report.pdf"
        try:
            supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=patient_pdf_filename, file=patient_pdf_bytes, file_options={"content-type": "application/pdf", "upsert": "true"})
            patient_pdf_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(patient_pdf_filename)
            if not isinstance(patient_pdf_url, str) or not patient_pdf_url.startswith('http'): patient_pdf_url=None; report_generation_errors.append("PatientPDFURL")
        except Exception as e: print(f"ERR PatientPDFUpload: {e}"); report_generation_errors.append("PatientPDFUpload")

        report_generation_status = "Completed" if not report_generation_errors else f"Completed with errors ({', '.join(report_generation_errors)})"
        db_similarity_data = {k: v for k, v in similarity_analysis_results.items() if k != 'plot_base64'} if isinstance(similarity_analysis_results, dict) and not similarity_analysis_results.get('error') else similarity_analysis_results

        update_data = {
            "stats_data": stats_json, "timeseries_plot_url": ts_url, "psd_plot_url": psd_url,
            "technical_pdf_url": technical_pdf_url, "patient_pdf_url": patient_pdf_url,
            "report_generated_at": datetime.now(timezone.utc).isoformat(), "status": report_generation_status,
            "similarity_results": db_similarity_data, "similarity_plot_url": similarity_plot_url,
        }
        print(f"Step 6: Updating DB for {prediction_id} with status '{report_generation_status}'...")
        update_payload_str = json.dumps(update_data, cls=NpEncoder, allow_nan=False); update_payload = json.loads(update_payload_str)
        supabase.table('predictions').update(update_payload).eq('id', prediction_id).execute()
        return jsonify({"filename": file.filename, "prediction": prediction_label, "prediction_id": prediction_id})

    except Exception as e:
        print(f"ERROR in /api/predict: {e}"); traceback.print_exc()
        if prediction_id:
            try: supabase.table('predictions').update({"status": f"Failed: {type(e).__name__} - {str(e)[:100]}" }).eq('id', prediction_id).execute()
            except Exception as final_update_e: print(f"Failed to update DB on error: {final_update_e}")
        if asset_prefix:
            assets_to_clean = [
                f"{asset_prefix}/similarity_plot_ch{channel_index_for_plot + 1}.png",
                f"{asset_prefix}/timeseries.png", f"{asset_prefix}/psd.png",
                f"{asset_prefix}/technical_report.pdf", f"{asset_prefix}/patient_report.pdf"
            ]
            for asset_path in assets_to_clean: _cleanup_storage_on_error(REPORT_ASSET_BUCKET, asset_path)
        _cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path)
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500
    finally:
        if 'absolute_temp_filepath' in locals() and os.path.exists(absolute_temp_filepath):
            try: os.remove(absolute_temp_filepath); print(f"Removed temp file: {absolute_temp_filepath}")
            except Exception as e: print(f"Error removing temp {absolute_temp_filepath}: {e}")
        if 'ml_output_file_path' in locals() and os.path.exists(ml_output_file_path):
            try: os.remove(ml_output_file_path); print(f"Removed ML output file: {ml_output_file_path}")
            except Exception as e: print(f"Error removing ML output: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.exists(ALZ_REF_PATH): print(f"WARNING: Alzheimer's reference file missing: {ALZ_REF_PATH}")
    if not os.path.exists(NORM_REF_PATH): print(f"WARNING: Normal reference file missing: {NORM_REF_PATH}")
    print("--- Starting Flask Server ---")
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"--- Debug Mode: {debug_mode} ---")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)