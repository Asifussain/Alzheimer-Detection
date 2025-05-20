# backend/app.py
# FINAL VERSION - Incorporating DETAILED Patient Consistency Metrics, Shorter Disclaimer & Path Fixes

import os
import uuid
import json
import subprocess
import io
import base64
import traceback 
import pandas as pd
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import numpy as np
from fpdf import FPDF, XPos, YPos 

# Helper for JSON serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj): return None 
            return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj) 
        return super(NpEncoder, self).default(obj)

# Import visualization functions
try:
    from visualization import (
        generate_stacked_timeseries_image,
        generate_average_psd_image,
        generate_descriptive_stats
    )
    print("Successfully imported visualization functions.")
except ImportError as import_err:
    print(f"CRITICAL ERROR importing visualization.py: {import_err}")
    def generate_stacked_timeseries_image(*args, **kwargs): return None
    def generate_average_psd_image(*args, **kwargs): return None
    def generate_descriptive_stats(*args, **kwargs): return {"error": "Visualization module not loaded"}

# Import the Similarity Analyzer
try:
    from similarity_analyzer import run_similarity_analysis
    print("Successfully imported similarity_analyzer.")
except ImportError as sim_import_err:
    print(f"CRITICAL ERROR importing similarity_analyzer.py: {sim_import_err}")
    def run_similarity_analysis(*args, **kwargs):
        return {"error": "Similarity analyzer module not loaded", 'interpretation': 'N/A', 'plot_base64': None, 'consistency_metrics': None, 'plotted_channel_index': 0}

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_URL", "*")}})

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") 
if not SUPABASE_URL or not SUPABASE_KEY: 
    raise ValueError("Supabase URL or Service Role Key environment variables not set.")
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

# --- PDF Report Base Class (Enhanced Styling) ---
class BasePDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "EEG Analysis Report"; self.logo_path = os.path.join(BACKEND_DIR, "logo.png") 
        self.primary_color = (52, 73, 94); self.secondary_color = (74, 144, 226); self.accent_color = (80, 227, 194)   
        self.text_color_dark = (40, 40, 40); self.text_color_light = (100, 100, 100); self.text_color_normal = (0, 0, 0)
        self.line_color = (200, 200, 200); self.card_bg_color = (245, 249, 252) 
        self.highlight_color_alz = (200, 50, 50); self.highlight_color_norm = (30, 150, 80) 
        self.warning_bg_color = (255, 243, 205); self.warning_text_color = (133, 100, 4) 

    def header(self):
        try:
            if os.path.exists(self.logo_path): self.image(self.logo_path, x=10, y=8, h=12); self.set_x(10 + 30 + 5) 
            else: self.set_x(10) 
            self.set_font('Helvetica', 'B', 18); self.set_text_color(*self.primary_color)
            remaining_width = self.w - self.get_x() - self.r_margin 
            self.multi_cell(remaining_width, 10, self.report_title, border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*self.text_color_normal); self.ln(3) 
            self.set_draw_color(*self.accent_color); self.set_line_width(0.6)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y()); self.ln(7) 
        except Exception as e: print(f"PDF Header Error: {e}")

    def footer(self):
        try:
            self.set_y(-15); self.set_font('Helvetica', 'I', 8); self.set_text_color(*self.text_color_light)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', border=0, align='C')
            self.set_x(self.l_margin) 
            self.cell(0, 10, f'Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}', border=0, align='R')
            self.set_text_color(*self.text_color_normal)
        except Exception as e: print(f"PDF Footer Error: {e}")

    def section_title(self, title, icon_char=""):
        try:
            self.ln(2); self.set_font('Helvetica', 'B', 13); self.set_fill_color(*self.secondary_color) 
            self.set_text_color(255,255,255); icon_prefix = f"{icon_char} " if icon_char else " " 
            self.cell(0, 9, f"{icon_prefix}{title}", border=0, align='L', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*self.text_color_normal); self.ln(6) 
        except Exception as e: print(f"PDF Section Title Error: {e}")
    
    def key_value_pair(self, key, value_str, key_width=60, is_bold_value=False, value_color=None):
        self.set_font('Helvetica', 'B', 9.5); self.set_text_color(*self.text_color_dark)
        key_start_y = self.get_y()
        self.multi_cell(key_width, 6, str(key)+":", align='L', new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=self.font_size)
        self.set_y(key_start_y); self.set_x(self.l_margin + key_width + 2) 
        value_width = self.w - self.l_margin - self.r_margin - key_width - 2 
        font_style_val = 'B' if is_bold_value else ''; self.set_font('Helvetica', font_style_val, 9.5) 
        if value_color: self.set_text_color(*value_color) 
        else: self.set_text_color(*self.text_color_normal)
        self.multi_cell(value_width, 6, str(value_str), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*self.text_color_normal); self.ln(1.5) 

    def write_paragraph(self, text, height=5, indent=0, font_style='', font_size=9.5, text_color=None, bullet_char=None):
        try:
             self.set_font('Helvetica', font_style, font_size)
             if text_color: self.set_text_color(*text_color)
             else: self.set_text_color(*self.text_color_dark) 
             current_x_start = self.l_margin + indent; self.set_x(current_x_start)
             if bullet_char: 
                 self.set_font('Helvetica', 'B', font_size + 1) 
                 self.cell(self.get_string_width(bullet_char) + 1, height, bullet_char) 
                 self.set_x(current_x_start + self.get_string_width(bullet_char) + 2) 
                 self.set_font('Helvetica', font_style, font_size) 
                 self.multi_cell(self.w - self.get_x() - self.r_margin, height, text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
             else: self.multi_cell(0, height, text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
             self.ln(height / 3); self.set_text_color(*self.text_color_normal) 
        except Exception as e: print(f"PDF write_paragraph Error: {e}")

    def add_image_section(self, title, image_data_base64, caption="", icon_char="🖼️"):
        if image_data_base64 and isinstance(image_data_base64, str) and image_data_base64.startswith('data:image/png;base64,'):
            self.section_title(title, icon_char=icon_char)
            try:
                img_bytes = base64.b64decode(image_data_base64.split(',')[1]); img_file = io.BytesIO(img_bytes)
                page_content_width = self.w - self.l_margin - self.r_margin; img_display_width = page_content_width * 0.85
                x_pos = self.l_margin + (page_content_width - img_display_width) / 2 
                self.image(img_file, x=x_pos, w=img_display_width); img_file.close(); self.ln(4)
                if caption: 
                    self.set_font('Helvetica', 'I', 8.5); self.set_text_color(*self.text_color_light)
                    self.set_x(self.l_margin + (page_content_width - (page_content_width * 0.9)) / 2) 
                    self.multi_cell(page_content_width * 0.9, 5, caption, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_text_color(*self.text_color_normal)
            except Exception as e: 
                self.set_font("Helvetica",'I',9); self.set_text_color(200,0,0) 
                self.cell(0,8,f"(Error embedding image '{title}': {str(e)[:100]})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_text_color(*self.text_color_normal); print(f"PDF Image Embed Error for {title}: {e}")
        else: 
            self.section_title(title, icon_char=icon_char)
            self.write_paragraph("(Image data not available or in invalid format)", font_style='I', indent=5)
        self.ln(6)

    def add_explanation_box(self, title, text_lines, icon_char="💡", bg_color=(235, 245, 255), title_color=None, text_color_override=None, font_size_text=9):
        self.ln(2); title_color_actual = title_color if title_color else self.primary_color
        text_color_actual = text_color_override if text_color_override else self.text_color_dark
        self.set_font('Helvetica', 'B', 11); self.set_text_color(*title_color_actual)
        self.cell(0, 8, f"{icon_char} {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_fill_color(*bg_color); self.set_draw_color(*self.line_color); y_before = self.get_y()
        est_height = 3; line_h = 5 # Base line height for font_size_text
        
        for line_item in text_lines:
            is_list_item = isinstance(line_item, tuple) and len(line_item) == 2 and line_item[0] == "bullet"
            text_to_measure = line_item[1] if is_list_item else line_item
            is_sub_bullet = isinstance(text_to_measure, tuple) and len(text_to_measure) == 2 and text_to_measure[0] == "sub_bullet"
            final_text_to_measure = text_to_measure[1] if is_sub_bullet else text_to_measure

            self.set_font('Helvetica', 'BI' if "**" in final_text_to_measure else '', font_size_text) 
            # Calculate width for text, considering potential bullet indentation
            text_width = self.w - self.l_margin - self.r_margin - 8 # base padding
            if is_list_item: text_width -= (5 if not is_sub_bullet else 10) # reduce width for bullet space

            num_lines_for_this_text = len(self.multi_cell(text_width, line_h, final_text_to_measure.replace("**",""), split_only=True))
            est_height += num_lines_for_this_text * line_h + (1 if is_list_item else 0.5)
        est_height += 3
        
        self.rect(self.l_margin, y_before, self.w - self.l_margin - self.r_margin, est_height, 'DF') 
        self.set_y(y_before + 3) 
        
        for line_item in text_lines: 
            is_list_item = isinstance(line_item, tuple) and len(line_item) == 2 and line_item[0] == "bullet"
            text_content = line_item[1] if is_list_item else line_item
            is_sub_bullet = isinstance(text_content, tuple) and len(text_content) == 2 and text_content[0] == "sub_bullet"
            final_text_content = text_content[1] if is_sub_bullet else text_content
            
            parts = final_text_content.split("**")
            base_indent = 3
            bullet_indent = base_indent + (5 if is_list_item and not is_sub_bullet else (10 if is_sub_bullet else 0))
            
            self.set_x(self.l_margin + base_indent) # Start of line indent

            if is_list_item and not is_sub_bullet:
                self.set_font('Helvetica', 'B', font_size_text + 2); self.set_text_color(*title_color_actual) 
                self.cell(5, line_h, "•"); self.set_x(self.l_margin + bullet_indent) 
            elif is_sub_bullet:
                self.set_x(self.l_margin + base_indent + 5) # Indent for sub-bullet
                self.set_font('Helvetica', '', font_size_text + 1); self.set_text_color(*text_color_actual)
                self.cell(5, line_h, "◦"); self.set_x(self.l_margin + bullet_indent)

            for i, part in enumerate(parts): 
                is_bold_part = i % 2 == 1; self.set_font('Helvetica', 'B' if is_bold_part else '', font_size_text)
                self.set_text_color(*(self.primary_color if is_bold_part else text_color_actual)); self.write(line_h, part)
            
            self.ln(line_h + (0.5 if is_list_item else 0.2))
            if is_list_item: self.set_y(self.get_y() - (line_h * 0.3)) # Fine-tune spacing for lists
            
        self.set_y(y_before + est_height); self.ln(5)

# --- Technical PDF Report Class ---
class TechnicalPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.report_title = "Comprehensive EEG Analysis Report" 
    def metric_card(self, title, value, unit="", description="", card_height=23, value_font_size=14):
        page_content_width = self.w - self.l_margin - self.r_margin; num_cards_per_row = 2; gap = 5 
        card_width = (page_content_width - gap * (num_cards_per_row - 1)) / num_cards_per_row
        current_x = self.get_x(); current_y = self.get_y()
        if current_x + card_width > self.w - self.r_margin + 0.1 or (current_x > self.l_margin and (self.w - current_x) < card_width):
            self.ln(card_height + 3); current_x = self.l_margin; current_y = self.get_y()
        self.set_xy(current_x, current_y)
        self.set_fill_color(*self.card_bg_color); self.set_draw_color(*self.line_color); self.set_line_width(0.2)
        self.rect(current_x, current_y, card_width, card_height, 'DF')
        self.set_xy(current_x + 3, current_y + 2.5); self.set_font('Helvetica', 'B', 8.5); self.set_text_color(*self.text_color_dark)
        self.multi_cell(card_width - 6, 4, title.upper(), align='L')
        self.set_xy(current_x + 3, self.get_y() + 0.5); self.set_font('Helvetica', 'B', value_font_size); self.set_text_color(*self.primary_color)
        self.multi_cell(card_width - 6, 7, f"{value}{unit}", align='R')
        if description:
            self.set_xy(current_x + 3, current_y + card_height - 6); self.set_font('Helvetica', 'I', 7); self.set_text_color(*self.text_color_light)
            self.multi_cell(card_width - 6, 3, description, align='L')
        next_x_pos = current_x + card_width + gap
        if next_x_pos + card_width <= self.w - self.r_margin + 0.1 : self.set_x(next_x_pos); self.set_y(current_y) 
        else: self.set_x(self.l_margin); self.ln(card_height + 3) 
        self.set_text_color(*self.text_color_normal); self.set_line_width(0.2)

# --- Patient PDF Report Class ---
class PatientPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.report_title = "Your AI EEG Pattern Report"

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

# --- Technical PDF Content Builder (Content from previous response, minor layout tweaks) ---
def _build_technical_pdf_report_content(pdf: TechnicalPDFReport, prediction_data, stats_data, similarity_data, consistency_metrics, ts_img_data, psd_img_data):
    pdf.set_auto_page_break(auto=True, margin=20); pdf.add_page()
    pdf.section_title("Analysis Overview & AI Prediction", icon_char="📄")
    pdf.key_value_pair("Filename", prediction_data.get('filename', 'N/A'))
    created_at_str = pd.to_datetime(prediction_data.get('created_at')).strftime('%Y-%m-%d %H:%M:%S UTC') if prediction_data.get('created_at') else 'N/A'
    pdf.key_value_pair("Analysis Timestamp", created_at_str)
    prediction_label = prediction_data.get('prediction', 'N/A')
    pred_color = pdf.highlight_color_alz if prediction_label == "Alzheimer's" else (pdf.highlight_color_norm if prediction_label == "Normal" else pdf.text_color_dark)
    pdf.key_value_pair("Overall AI Prediction", prediction_label, is_bold_value=True, value_color=pred_color)
    probabilities = prediction_data.get('probabilities'); prob_str = 'N/A'
    if isinstance(probabilities, list) and len(probabilities) == 2:
        try: prob_str = f"Normal: {probabilities[0]*100:.1f}%, Alzheimer's Pattern: {probabilities[1]*100:.1f}%"
        except: prob_str = str(probabilities)
    pdf.key_value_pair("Prediction Confidence (Initial Segment)", prob_str); pdf.ln(5)
    pdf.section_title("Internal Consistency Assessment", icon_char="⚙️")
    pdf.write_paragraph("These metrics evaluate the AI model's predictive consistency across multiple segments of the input EEG sample. The model's prediction for each segment is compared against its overall prediction for the entire file. This serves as an internal validation of model stability on this specific sample, not as a measure of diagnostic accuracy against external ground truth. Calculations are based on the overall prediction for this file acting as the 'true' label for its segments.", height=4.5, font_size=9); pdf.ln(2)
    if consistency_metrics and not consistency_metrics.get('error') and consistency_metrics.get('num_trials', 0) > 1:
        metrics = consistency_metrics; num_trials = metrics.get('num_trials', 'N/A')
        pdf.key_value_pair("Segments Analyzed", str(num_trials), key_width=70)
        pdf.metric_card("Overall Agreement (Accuracy)", f"{metrics.get('accuracy', 0)*100:.1f}%", description="Segs matching overall prediction / Total segs")
        pdf.metric_card("Precision (Alz. Pattern)", f"{metrics.get('precision', 0):.3f}", description="TP / (TP + FP) for Alzheimer's class"); pdf.ln(26) 
        pdf.metric_card("Recall/Sensitivity (Alz. Pattern)", f"{metrics.get('recall_sensitivity', 0):.3f}", description="TP / (TP + FN) for Alzheimer's class")
        pdf.metric_card("Specificity (Normal Pattern)", f"{metrics.get('specificity', 0):.3f}", description="TN / (TN + FP) for Normal class"); pdf.ln(26)
        pdf.metric_card("F1-Score (Alz. Pattern)", f"{metrics.get('f1_score', 0):.3f}", description="2*(Prec*Rec)/(Prec+Rec) for Alz."); pdf.ln(26) 
        pdf.set_font('Helvetica', 'B', 9); pdf.write(5, "Confusion Matrix Details "); pdf.set_font('Helvetica', 'I', 8.5); pdf.write(5, f"(Reference: AI's overall prediction of '{prediction_data.get('prediction', 'N/A')}')") ; pdf.ln(4)
        cm_details = [f"  • True Positives (TP): {metrics.get('true_positives','?')}", f"  • True Negatives (TN): {metrics.get('true_negatives','?')}", f"  • False Positives (FP): {metrics.get('false_positives','?')}", f"  • False Negatives (FN): {metrics.get('false_negatives','?')}"]
        for detail in cm_details: pdf.write_paragraph(detail, font_size=8.5, height=4)
    else: pdf.write_paragraph(f"({consistency_metrics.get('message', 'Metrics not calculated or N/A.') if consistency_metrics else 'Metrics N/A.'})", font_style='I', indent=3, font_size=9)
    pdf.ln(6); pdf.add_page() 
    pdf.section_title("Signal Shape Similarity Analysis (DTW)", icon_char="🌊")
    if similarity_data and not similarity_data.get('error'):
        pdf.write_paragraph("Dynamic Time Warping (DTW) quantifies waveform similarity between the input EEG and predefined reference patterns (Normal and Alzheimer's), robust to temporal shifts. Lower DTW distances imply higher similarity. Reference patterns are derived from training data exemplars. The analysis is channel-wise, with aggregated results presented.", height=4.5, font_size=9); pdf.ln(1)
        pdf.write_paragraph(similarity_data.get('interpretation', 'No interpretation available.'), indent=3, font_style='I', font_size=9, height=4.5); pdf.ln(2)
        plotted_ch_idx = similarity_data.get('plotted_channel_index'); sim_plot_caption = (f"DTW Waveform Comparison: Channel {plotted_ch_idx + 1 if plotted_ch_idx is not None else 'Default'}. Normalized Z-scores of the sample EEG segment against reference signals.")
        pdf.add_image_section("Channel Waveform Comparison Plot", similarity_data.get('plot_base64'), caption=sim_plot_caption)
    else: pdf.write_paragraph(f"(Similarity Analysis Error: {similarity_data.get('error', 'Data N/A') if similarity_data else 'Data N/A'})", font_style='I', indent=3)
    pdf.ln(6); pdf.section_title("Descriptive EEG Statistics", icon_char="📊")
    if stats_data and not stats_data.get('error'):
        pdf.write_paragraph("Quantitative summary of the EEG signal's spectral characteristics. Relative band power indicates the proportion of total signal power within standard frequency bands.", height=4.5, font_size=9); pdf.ln(1)
        pdf.set_font("Helvetica",'B',10.5); pdf.cell(0,6,"Average Relative Band Power (%):", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(1); pdf.set_font("Helvetica",'',9.5)
        avg_power = stats_data.get('avg_band_power',{}); band_info = {"Delta": ("0.5-4 Hz", "Deep sleep, slow-wave activity."),"Theta": ("4-8 Hz", "Drowsiness, memory consolidation."),"Alpha": ("8-13 Hz", "Relaxed wakefulness; suppressed by mental effort."),"Beta": ("13-30 Hz", "Active mental engagement, concentration."),"Gamma": ("30-50 Hz", "Higher cognitive functions, perception.")}
        for band, powers_dict in avg_power.items():
            rel_power = powers_dict.get('relative'); rel_power_str = f"{rel_power * 100:.2f}%" if isinstance(rel_power, (float, int)) else "N/A"
            freq_range, typical_assoc = band_info.get(band.capitalize(), ("N/A", "N/A"))
            pdf.key_value_pair(f"  • {band.capitalize()} [{freq_range}]", f"{rel_power_str} - {typical_assoc}", key_width=55); pdf.ln(1)
        std_devs = stats_data.get('std_dev_per_channel')
        if std_devs and isinstance(std_devs, list):
            pdf.ln(3); pdf.set_font("Helvetica",'B',10.5); pdf.cell(0,6,"Std. Deviation per Channel (µV):", new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(1)
            pdf.write_paragraph("Indicates signal amplitude variability across channels.", font_style='I', font_size=8.5, indent=3)
            pdf.write_paragraph(", ".join([f"{s:.2f}" if isinstance(s, (float,int)) else "N/A" for s in std_devs]), height=4, indent=5, font_size=8.5)
    else: pdf.write_paragraph(f"(Statistics Error: {stats_data.get('error', 'Data N/A') if stats_data else 'Data N/A'})", font_style='I', indent=3)
    pdf.ln(6); pdf.add_page() 
    pdf.section_title("Standard EEG Visualizations", icon_char="📈")
    pdf.add_image_section("Stacked EEG Time Series", ts_img_data, caption="Raw voltage (µV) fluctuations over time (seconds) for all 19 EEG channels, stacked vertically. This view helps identify overall signal quality, presence of artifacts, and synchronous events across channels.")
    pdf.add_image_section("Average Power Spectral Density (PSD)", psd_img_data, caption="Average distribution of EEG signal power (µV²/Hz, log scale) across frequencies (0-50Hz). Shaded regions denote standard clinical bands. Peaks indicate dominant frequencies, providing insights into the brain's oscillatory activity.")
    pdf.ln(8); pdf.section_title("Report Disclaimer & Methodology Note", icon_char="❗")
    pdf.write_paragraph("This report is generated by an AI model (ADFormer) analyzing EEG data for pattern recognition potentially associated with Alzheimer's disease. It is intended for technical review and to supplement clinical assessment by qualified professionals. This report IS NOT A MEDICAL DIAGNOSIS. The 'Internal Consistency' metrics reflect the model's stability on this specific sample by comparing segment-wise predictions to the overall file prediction. DTW analysis compares signal morphology against reference patterns. Descriptive statistics summarize spectral power distribution. Visualizations provide qualitative signal overview. All interpretations should be made by qualified personnel considering the full clinical context.", height=4.5, font_size=9)

# --- Patient PDF Content Builder (Updated with detailed consistency metrics & shorter disclaimer) ---
def _build_patient_pdf_report_content(pdf: PatientPDFReport, prediction_data, similarity_data, consistency_metrics, similarity_plot_data):
    pdf.set_auto_page_break(auto=True, margin=20); pdf.add_page()
    pdf.section_title("Your AI EEG Pattern Report", icon_char="🧠")
    created_at_str = pd.to_datetime(prediction_data.get('created_at')).strftime('%B %d, %Y') if prediction_data.get('created_at') else 'N/A'
    pdf.key_value_pair("File Analyzed", prediction_data.get('filename', 'N/A'), key_width=45)
    pdf.key_value_pair("Date of Analysis", created_at_str, key_width=45); pdf.ln(6)
    pdf.add_explanation_box(
        "About This Report", 
        ["This report uses Artificial Intelligence (AI) to look for specific patterns in your brainwave (EEG) activity.",
         "The AI compares your EEG patterns to those it has learned from many examples.",
         ("bullet", "Important: This is an informational tool to help your doctor. **It is not a medical diagnosis.** Please discuss these results with them.")
        ], icon_char="ℹ️", font_size_text=9)
    
    pdf.section_title("AI's Main Finding: Pattern Assessment", icon_char="💡")
    prediction_label = prediction_data.get('prediction', 'Not Determined')
    pred_display_text = "Pattern assessment inconclusive"; pred_color = pdf.text_color_dark
    if prediction_label == "Alzheimer's": pred_display_text = "Patterns Suggestive of Alzheimer's Characteristics"; pred_color = pdf.highlight_color_alz
    elif prediction_label == "Normal": pred_display_text = "Normal Brainwave Patterns Observed"; pred_color = pdf.highlight_color_norm
    pdf.write_paragraph("The AI analyzed your EEG and found that the patterns are most similar to:", font_size=10, height=5)
    pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*pred_color) # Highlight color
    pdf.multi_cell(0, 7, pred_display_text, border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.set_text_color(*pdf.text_color_normal); pdf.ln(2)
    probabilities = prediction_data.get('probabilities'); confidence_text = "Confidence score not available for this analysis."
    if isinstance(probabilities, list) and len(probabilities) == 2:
        try: conf_val = probabilities[1]*100 if prediction_label == "Alzheimer's" else probabilities[0]*100; confidence_text = f"The AI is **{conf_val:.0f}%** confident that the patterns it found align with the finding above (based on the first segment of your EEG data)."
        except: pass 
    pdf.add_explanation_box("AI's Confidence Level", [confidence_text], icon_char="🎯", font_size_text=9)
    
    pdf.section_title("AI's Internal Consistency Check", icon_char="🤔")
    if consistency_metrics and not consistency_metrics.get('error') and isinstance(consistency_metrics.get('num_trials'), int) and consistency_metrics.get('num_trials', 0) > 0:
        num_segments = consistency_metrics.get('num_trials', 'multiple')
        
        intro_text = [f"To double-check its findings, the AI looked at your EEG data in **{num_segments} smaller pieces** (segments). This helps assess how stable the AI's finding was across your entire recording. Here's a simple breakdown of these checks:"]
        
        metric_items_for_patient = []
        accuracy_val = f"{consistency_metrics.get('accuracy', 0) * 100:.0f}%"
        metric_items_for_patient.append(("bullet", f"**Overall Consistency (Accuracy):** {accuracy_val}. This shows how often the AI's checks on the small pieces matched its main finding for your whole EEG sample."))
        
        if prediction_label == "Alzheimer's":
            sensitivity_val = f"{consistency_metrics.get('recall_sensitivity', 0) * 100:.0f}%"
            precision_val = f"{consistency_metrics.get('precision', 0) * 100:.0f}%"
            f1_val = f"{consistency_metrics.get('f1_score', 0):.2f}"
            metric_items_for_patient.append(("bullet", f"**Finding Alzheimer's-like Patterns (Sensitivity):** {sensitivity_val}. If segments showed Alzheimer's-like patterns (based on the main finding), the AI found them this often."))
            metric_items_for_patient.append(("bullet", f"**Confirming Alzheimer's-like Patterns (Precision):** {precision_val}. When the AI said a segment was Alzheimer's-like, it was consistent with the main finding this often."))
            metric_items_for_patient.append(("bullet", f"**Balanced Score for Alzheimer's Patterns (F1-Score):** {f1_val}. A combined score (0 to 1, higher is better) reflecting how well the AI balanced finding and confirming these patterns."))
        else: # Prediction is "Normal"
            specificity_val = f"{consistency_metrics.get('specificity', 0) * 100:.0f}%"
            metric_items_for_patient.append(("bullet", f"**Finding Normal Patterns (Specificity):** {specificity_val}. If Normal patterns were present in segments (based on the main finding), the AI correctly identified them this often."))
        
        metric_items_for_patient.append(("bullet", f"**Number of Segments Checked:** {num_segments}."))
        metric_items_for_patient.append("Higher percentages and scores in these checks generally suggest the AI was consistent in what it observed throughout your EEG sample.")
        
        pdf.add_explanation_box("Understanding AI's Consistency", intro_text + metric_items_for_patient, icon_char="🧐", bg_color=(230,250,230), title_color=pdf.highlight_color_norm, font_size_text=8.5) # Smaller font for dense info
    else: pdf.write_paragraph("(Detailed internal consistency checks were not applicable or did not yield specific metrics for this sample.)", font_style='I', indent=3, font_size=9)
    pdf.ln(5)

    if similarity_plot_data or (not similarity_plot_data and pdf.get_y() > pdf.h - 100): pdf.add_page()
    if similarity_data and not similarity_data.get('error') and similarity_plot_data:
        plotted_ch_idx = similarity_data.get('plotted_channel_index')
        plot_title = f"Comparing Your Brainwave Shape (from Channel {plotted_ch_idx + 1 if plotted_ch_idx is not None else 'Selected'})"
        sim_caption = (f"This graph (Channel {plotted_ch_idx + 1 if plotted_ch_idx is not None else 'Selected'}) shows one of your brainwave patterns (white line) compared to typical 'Normal' (blue dashed line) and 'Alzheimer's' (red dotted line) patterns that the AI has learned. The AI looks at how similar the overall shapes are.")
        pdf.add_image_section(plot_title, similarity_plot_data, caption=sim_caption, icon_char="📈")
        sim_interp_text_main = "The AI found that your sample's brainwave shapes showed "
        overall_sim = similarity_data.get('overall_similarity', '')
        if "Higher Similarity to Alzheimer's Pattern" in overall_sim: sim_interp_text_main += "**more resemblance to the Alzheimer's reference patterns**."
        elif "Higher Similarity to Normal Pattern" in overall_sim: sim_interp_text_main += "**more resemblance to the Normal reference patterns**."
        else: sim_interp_text_main += "a mixed or inconclusive resemblance when compared to the reference patterns."
        sim_interpretation_from_data = similarity_data.get('interpretation', "").split("Disclaimer:")[0].replace("Similarity Analysis (DTW):", "").replace("Overall Assessment:", "").strip()
        pdf.add_explanation_box("What This Graph Shows", [sim_interp_text_main, f"More Details: \"{sim_interpretation_from_data}\""], icon_char="📊", font_size_text=9)
    else:
        pdf.section_title("Comparing Your Brainwave Shape", icon_char="📈"); pdf.write_paragraph("(The brainwave shape comparison graph is not available for this report.)", font_style='I')
    pdf.ln(6)

    pdf.section_title("Important Information & Your Next Steps", icon_char="❗")
    # Shortened Disclaimer for Patient PDF
    pdf.add_explanation_box(
        "Please Discuss This Report With Your Doctor",
        [
            "This AI report is an informational tool based on EEG patterns. **It is NOT a medical diagnosis.**",
            "Only a qualified healthcare professional can diagnose medical conditions. They will consider this report along with your full medical history and other tests.",
            "**Key Takeaway:** The AI analysis suggests your EEG patterns are most similar to **" + pred_display_text + "**.",
            "**Recommended Next Steps:**",
            ("bullet", "Share this entire report with your doctor or a neurologist."),
            ("bullet", "Discuss any health concerns and follow their medical advice.")
        ],
        bg_color=pdf.warning_bg_color, title_color=(106, 63, 20), text_color_override=(85,60,10), font_size_text=9
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
        print("Generating Technical PDF report..."); tech_pdf = TechnicalPDFReport()
        _build_technical_pdf_report_content(tech_pdf, prediction_data_for_report, stats_json, similarity_analysis_results, consistency_metrics_results, ts_img_data, psd_img_data)
        tech_pdf_bytes = bytes(tech_pdf.output()); technical_pdf_filename = f"{asset_prefix}/technical_report.pdf"
        try:
            supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=technical_pdf_filename, file=tech_pdf_bytes, file_options={"content-type": "application/pdf", "upsert": "true"})
            technical_pdf_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(technical_pdf_filename)
            if not isinstance(technical_pdf_url, str) or not technical_pdf_url.startswith('http'): technical_pdf_url=None; report_generation_errors.append("TechPDFURL")
        except Exception as e: print(f"ERR TechPDFUpload: {e}"); report_generation_errors.append("TechPDFUpload")
        print("Generating Patient PDF report..."); patient_pdf = PatientPDFReport()
        _build_patient_pdf_report_content(patient_pdf, prediction_data_for_report, similarity_analysis_results, consistency_metrics_results, similarity_plot_base64_data)
        patient_pdf_bytes = bytes(patient_pdf.output()); patient_pdf_filename = f"{asset_prefix}/patient_report.pdf"
        try:
            supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=patient_pdf_filename, file=patient_pdf_bytes, file_options={"content-type": "application/pdf", "upsert": "true"})
            patient_pdf_url = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(patient_pdf_filename)
            if not isinstance(patient_pdf_url, str) or not patient_pdf_url.startswith('http'): patient_pdf_url=None; report_generation_errors.append("PatientPDFURL")
        except Exception as e: print(f"ERR PatientPDFUpload: {e}"); report_generation_errors.append("PatientPDFUpload")
        report_generation_status = "Completed" if not report_generation_errors else f"Completed with errors ({', '.join(report_generation_errors)})"
        db_similarity_data = {k: v for k, v in similarity_analysis_results.items() if k != 'plot_base64'} if isinstance(similarity_analysis_results, dict) and not similarity_analysis_results.get('error') else similarity_analysis_results
        update_data = {"stats_data": stats_json, "timeseries_plot_url": ts_url, "psd_plot_url": psd_url, "pdf_report_url": technical_pdf_url, "technical_pdf_url": technical_pdf_url, "patient_pdf_url": patient_pdf_url, "report_generated_at": datetime.now(timezone.utc).isoformat(), "status": report_generation_status, "similarity_results": db_similarity_data, "similarity_plot_url": similarity_plot_url,}
        print(f"Step 6: Updating DB for {prediction_id} with status '{report_generation_status}'...")
        update_payload_str = json.dumps(update_data, cls=NpEncoder, allow_nan=False); update_payload = json.loads(update_payload_str)
        supabase.table('predictions').update(update_payload).eq('id', prediction_id).execute()
        return jsonify({"filename": file.filename, "prediction": prediction_label, "prediction_id": prediction_id})
    except Exception as e: 
        print(f"ERROR in /api/predict: {e}"); traceback.print_exc()
        if prediction_id:
            try: supabase.table('predictions').update({"status": f"Failed: {type(e).__name__}" }).eq('id', prediction_id).execute()
            except Exception as final_update_e: print(f"Failed to update DB on error: {final_update_e}")
        if asset_prefix: 
            _cleanup_storage_on_error(REPORT_ASSET_BUCKET, f"{asset_prefix}/similarity_plot_ch{channel_index_for_plot + 1}.png"); _cleanup_storage_on_error(REPORT_ASSET_BUCKET, f"{asset_prefix}/timeseries.png"); _cleanup_storage_on_error(REPORT_ASSET_BUCKET, f"{asset_prefix}/psd.png"); _cleanup_storage_on_error(REPORT_ASSET_BUCKET, f"{asset_prefix}/technical_report.pdf"); _cleanup_storage_on_error(REPORT_ASSET_BUCKET, f"{asset_prefix}/patient_report.pdf")
        _cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path) 
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500
    finally: 
        if 'absolute_temp_filepath' in locals() and os.path.exists(absolute_temp_filepath): # type: ignore
            try: os.remove(absolute_temp_filepath); print(f"Removed temp file: {absolute_temp_filepath}") # type: ignore
            except Exception as e: print(f"Error removing temp {absolute_temp_filepath}: {e}") # type: ignore
        if 'ml_output_file_path' in locals() and os.path.exists(ml_output_file_path): # type: ignore
            try: os.remove(ml_output_file_path); print(f"Removed ML output file: {ml_output_file_path}") # type: ignore
            except Exception as e: print(f"Error removing ML output: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.exists(ALZ_REF_PATH): print(f"WARNING: Alzheimer's reference file missing: {ALZ_REF_PATH}")
    if not os.path.exists(NORM_REF_PATH): print(f"WARNING: Normal reference file missing: {NORM_REF_PATH}")
    print("--- Starting Flask Server ---")
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"--- Debug Mode: {debug_mode} ---")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)

