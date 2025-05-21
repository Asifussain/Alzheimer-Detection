# backend/app.py
# Version: Fix AttributeError, Major Technical Report Aesthetic & Layout Overhaul

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

# --- Sanitization Function ---
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

# Import visualization and similarity functions
try:
    from visualization import generate_stacked_timeseries_image, generate_average_psd_image, generate_descriptive_stats
    from similarity_analyzer import run_similarity_analysis
    print("Successfully imported helper modules.")
except ImportError as import_err:
    print(f"CRITICAL ERROR importing helper modules: {import_err}")
    def generate_stacked_timeseries_image(*args, **kwargs): return None
    def generate_average_psd_image(*args, **kwargs): return None
    def generate_descriptive_stats(*args, **kwargs): return {"error": "Visualization module not loaded"}
    def run_similarity_analysis(*args, **kwargs):
        return {"error": "Similarity module not loaded", 'interpretation': 'N/A', 'plot_base64': None}

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

UPLOAD_FOLDER = 'uploads'; SIDDHI_FOLDER = 'SIDDHI'
OUTPUT_JSON_PATH = os.path.join(SIDDHI_FOLDER, 'output.json')
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ALZ_REF_PATH = os.path.join(BACKEND_DIR, 'feature_07.npy')
NORM_REF_PATH = os.path.join(BACKEND_DIR, 'feature_35.npy')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_FS = 128
RAW_EEG_BUCKET = 'eeg-data'; REPORT_ASSET_BUCKET = 'report-assets'

# --- PDF Report Base Class (with auto-sanitization and add_explanation_box) ---
class BasePDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "EEG Analysis Report"
        self.primary_color = (52, 73, 94); self.secondary_color = (74, 144, 226)
        self.text_color_dark = (30, 30, 30); self.text_color_light = (100, 100, 100)
        self.text_color_normal = (0,0,0)
        self.line_color = (220, 220, 220); self.card_bg_color = (248, 249, 250)
        self.highlight_color_alz = (220, 60, 60); self.highlight_color_norm = (60, 179, 113)
        self.warning_bg_color = (255, 243, 205); self.warning_text_color = (133, 100, 4)
        self.page_margin = 15 # Using sample's implied margins
        self.set_auto_page_break(auto=True, margin=self.page_margin)
        self.set_line_width(0.2)

    def _is_bold_font(self):
        return 'B' in self.font_style

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

    def header(self):
        try:
            self.set_y(self.page_margin - 5) # Start title higher
            self.set_font('Helvetica', 'B', 18) # Larger title
            self.set_text_color(*self.text_color_dark)
            # Center title:
            title_text = sanitize_for_helvetica(self.report_title)
            title_w = self.get_string_width(title_text) + 6
            self.set_x((self.w - title_w) / 2)
            self.cell(title_w, 10, title_text, 0, 1, 'C')
            self.ln(4) # Space after title
        except Exception as e: print(f"PDF Header Error: {e}")

    def footer(self):
        try:
            self.set_y(-(self.page_margin)); self.set_font('Helvetica', 'I', 8); self.set_text_color(*self.text_color_light)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', border=0, align='C')
            self.set_text_color(*self.text_color_normal)
        except Exception as e: print(f"PDF Footer Error: {e}")

    def section_title(self, title): # For bold, underlined titles like sample
        try:
            self.ln(6); self.set_font('Helvetica', 'B', 11); self.set_text_color(*self.text_color_dark)
            # Underline the title by drawing a line after printing it
            title_text = sanitize_for_helvetica(title)
            self.cell(0, 7, title_text, border=0, ln=0, align='L') # ln=0 to draw line immediately after
            line_y = self.get_y() + 6.5 # Position line under text
            self.set_draw_color(180,180,180) # Grey underline
            self.line(self.l_margin, line_y, self.l_margin + self.get_string_width(title_text) + 1, line_y)
            self.ln(7) # Move down after title + line
            self.set_text_color(*self.text_color_normal)
            self.set_draw_color(0,0,0) # Reset draw color
        except Exception as e: print(f"PDF Section Title Error: {e}")

    def key_value_pair(self, key, value_str, key_width=45, is_bold_value=False, value_color=None):
        self.set_font('Helvetica', 'B', 9); self.set_text_color(*self.text_color_dark)
        key_start_y = self.get_y()
        self.multi_cell(key_width, 5, sanitize_for_helvetica(str(key)+":"), align='L', new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=5)
        self.set_y(key_start_y); self.set_x(self.l_margin + key_width + 2)
        value_width = self.w - self.l_margin - self.r_margin - key_width - 2
        font_style_val = 'B' if is_bold_value else ''; self.set_font('Helvetica', font_style_val, 9)
        text_c = value_color if value_color else self.text_color_normal
        self.set_text_color(*text_c)
        self.multi_cell(value_width, 5, sanitize_for_helvetica(str(value_str)), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=5)
        self.set_text_color(*self.text_color_normal); self.ln(1)

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
        image_height_estimate = 70 # Approximate height for EEG plots
        title_height_estimate = 10 if title else 0 # Space for title if provided
        
        if self.get_y() + title_height_estimate + image_height_estimate > self.h - self.b_margin:
            self.add_page()

        if title:
            self.set_font('Helvetica', 'B', 10); self.set_text_color(*self.text_color_dark)
            self.cell(0, 8, sanitize_for_helvetica(title), ln=1, align='L') # Left align title for images
            self.ln(1)

        if image_data_base64 and isinstance(image_data_base64, str) and image_data_base64.startswith('data:image/png;base64,'):
            try:
                img_bytes = base64.b64decode(image_data_base64.split(',')[1]); img_file = io.BytesIO(img_bytes)
                page_content_width = self.w - 2 * self.page_margin
                img_display_width = page_content_width * 0.95 # Slightly larger to fill space better
                x_pos = self.page_margin + (page_content_width - img_display_width) / 2
                self.image(img_file, x=x_pos, w=img_display_width); img_file.close(); self.ln(2)
            except Exception as e:
                self.write_paragraph(sanitize_for_helvetica(f"(Error embedding image '{title}': {str(e)[:50]})"), font_style='I')
                print(f"PDF Image Embed Error for {title}: {e}")
        else:
            if title: self.write_paragraph(sanitize_for_helvetica("(Image data not available)"), font_style='I', indent=5)
        self.ln(4) # More space after images

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
        # Simplified height estimation for the box
        estimated_box_height = 3 # Initial padding
        for item in text_lines:
            estimated_box_height += line_h + 1 # Add height for each line + spacing
        
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
                if is_sub_list:
                    self.set_x(bullet_x + 5)
                    bullet_char_to_use = "-" 
                self.cell(5, line_h, bullet_char_to_use)
                self.set_x(bullet_x + 5 + (5 if is_sub_list else 0))

            bold_parts = sanitize_for_helvetica(final_txt).split("**")
            for i, part_text in enumerate(bold_parts):
                is_bold = i % 2 == 1
                self.set_font('Helvetica', 'B' if is_bold else '', font_size_text)
                self.set_text_color(*(self.primary_color if is_bold else text_c))
                self.write(line_h, part_text)
            self.ln(line_h + 0.5)
            
        self.set_y(y_before_text_content + estimated_box_height); self.ln(3)
        self.set_text_color(*self.text_color_normal)


# --- Technical PDF Report Class ---
class TechnicalPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "EEG Analysis Report"

    def metric_card_item(self, title, value, description, width, cell_height=15): # Adjusted height from sample
        item_padding_x = 2; item_padding_y = 2
        line_h_title = 3.5; line_h_value = 4.5; line_h_desc = 3

        current_x = self.get_x(); current_y = self.get_y()
        
        inner_x = current_x + item_padding_x
        pos_y = current_y + item_padding_y

        self.set_xy(inner_x, pos_y)
        self.set_font('Helvetica', 'B', 7.5); self.set_text_color(*self.text_color_dark)
        self.multi_cell(width - 2 * item_padding_x, line_h_title, title.upper(), align='L', border=0) # Ensure no border from multi_cell
        pos_y += line_h_title

        self.set_xy(inner_x, pos_y)
        self.set_font('Helvetica', 'B', 11); self.set_text_color(*self.primary_color) # Value font larger
        self.multi_cell(width - 2 * item_padding_x, line_h_value, value, align='L', border=0)
        pos_y += line_h_value 
        
        if description:
            self.set_xy(inner_x, pos_y)
            self.set_font('Helvetica', '', 7); self.set_text_color(*self.text_color_light) # Description italic-like
            self.multi_cell(width - 2 * item_padding_x, line_h_desc, description, align='L', border=0)
        
        self.set_text_color(*self.text_color_normal)
        self.set_xy(current_x + width, current_y) # Set X for next cell in the row


# --- Patient PDF Report Class ---
class PatientPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Your AI EEG Pattern Report"
    # Inherits add_explanation_box from BasePDFReport


# (Helper functions: get_prediction_and_eeg, _cleanup_storage_on_error, run_model - remain unchanged)
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

def _cleanup_storage_on_error(bucket_name, path):
    try:
        if bucket_name and path: supabase.storage.from_(bucket_name).remove([path])
    except Exception as e: print(f"Error during storage cleanup: {e}")

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


# --- Technical PDF Content Builder (Revamped) ---
def _build_technical_pdf_report_content(pdf: TechnicalPDFReport, prediction_data, stats_data, similarity_data, consistency_metrics, ts_img_data, psd_img_data):
    pdf.add_page()

    pdf.section_title("Analysis Details")
    pdf.key_value_pair("Filename", prediction_data.get('filename', 'N/A'))
    created_at_str = pd.to_datetime(prediction_data.get('created_at')).strftime('%Y-%m-%d %H:%M:%S UTC') if prediction_data.get('created_at') else 'N/A'
    pdf.key_value_pair("Analyzed On", created_at_str); pdf.ln(4)

    pdf.section_title("ML Prediction & Internal Consistency")
    prediction_label = prediction_data.get('prediction', 'N/A')
    pred_color = pdf.highlight_color_alz if prediction_label == "Alzheimer's" else (pdf.highlight_color_norm if prediction_label == "Normal" else pdf.text_color_dark)
    pdf.key_value_pair("Overall Prediction", prediction_label, is_bold_value=True, value_color=pred_color)
    probabilities = prediction_data.get('probabilities'); prob_str = 'N/A'
    if isinstance(probabilities, list) and len(probabilities) == 2:
        try: prob_str = f"Normal: {probabilities[0]*100:.1f}%, Alzheimer's: {probabilities[1]*100:.1f}%"
        except: prob_str = str(probabilities)
    pdf.key_value_pair("Confidence (first trial)", prob_str); pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 9); pdf.write(5, "Internal Consistency Metrics:"); pdf.ln(3)
    pdf.set_font('Helvetica', '', 7.5)
    pdf.multi_cell(0, 3.5, "(Compares segment predictions against the overall prediction for this file)", align='L'); pdf.ln(1.5)

    if consistency_metrics and not consistency_metrics.get('error') and isinstance(consistency_metrics.get('num_trials'), int) and consistency_metrics.get('num_trials', 0) > 0:
        metrics = consistency_metrics
        metric_items_for_grid = [
            ("ACCURACY", f"{metrics.get('accuracy', 0)*100:.1f}%", "Overall segment agreement"),
            ("PRECISION (ALZ)", f"{metrics.get('precision', 0):.3f}", "Alz predictions correct"),
            ("SENSITIVITY (ALZ)", f"{metrics.get('recall_sensitivity', 0):.3f}", "Alz segments found"),
            ("F1-SCORE (ALZ)", f"{metrics.get('f1_score', 0):.3f}", "Precision/Sensitivity balance"),
            ("SPECIFICITY (NORM)", f"{metrics.get('specificity', 0):.3f}", "Normal segments found"),
            ("TRIALS ANALYZED", str(metrics.get('num_trials', 'N/A')), "Segments in file")
        ]
        
        page_width = pdf.w - 2 * pdf.page_margin
        num_cols = 3; col_spacing = 2
        metric_cell_width = (page_width - (num_cols - 1) * col_spacing) / num_cols
        metric_cell_height = 15 # Adjusted for content

        initial_y_for_grid = pdf.get_y()
        current_y_offset = 0
        for i, (title, value, desc) in enumerate(metric_items_for_grid):
            col_idx = i % num_cols
            if col_idx == 0 and i > 0: # New row
                current_y_offset += metric_cell_height + 1.5 # Add height of previous row + spacing
            
            x_pos = pdf.l_margin + col_idx * (metric_cell_width + col_spacing)
            y_pos = initial_y_for_grid + current_y_offset
            
            pdf.set_xy(x_pos, y_pos)
            pdf.metric_card_item(title, value, desc, metric_cell_width, metric_cell_height)
        
        pdf.set_y(initial_y_for_grid + current_y_offset + metric_cell_height + 1.5); # Move below the last row of the grid
            
        cm_text = (f"(Ref Label: {metrics.get('majority_label_used_as_reference', '?')})  "
                   f"TP: {metrics.get('true_positives','?')} | TN: {metrics.get('true_negatives','?')} | "
                   f"FP: {metrics.get('false_positives','?')} | FN: {metrics.get('false_negatives','?')}")
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.multi_cell(0, 3.5, cm_text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.write_paragraph(f"({consistency_metrics.get('message', 'Metrics not calculated or N/A.') if consistency_metrics else 'Metrics N/A.'})", font_style='I', font_size=8)
    pdf.ln(3)
    
    est_height_similarity_section = 25 + 60 # text + plot
    if pdf.get_y() + est_height_similarity_section > pdf.h - pdf.b_margin: pdf.add_page()
    pdf.section_title("Signal Shape Similarity Analysis (DTW)")
    if similarity_data and not similarity_data.get('error'):
        pdf.set_font('Helvetica', '', 8.5); pdf.set_text_color(*pdf.text_color_dark)
        pdf.write_paragraph("Similarity Analysis (DTW):", height=4)
        overall_assessment = similarity_data.get('overall_similarity', 'N/A')
        pdf.write_paragraph(f"- Overall Assessment: {overall_assessment}", indent=2, height=3.5, bullet_char_override="-")
        n_ch_total = similarity_data.get('total_channels', 0)
        norm_c = similarity_data.get('normal_closer_count', 0); alz_c = similarity_data.get('alz_closer_count', 0)
        norm_p = (norm_c / n_ch_total * 100) if n_ch_total > 0 else 0
        alz_p = (alz_c / n_ch_total * 100) if n_ch_total > 0 else 0
        pdf.write_paragraph(f"- Channels More Similar to Normal Ref: {norm_c} ({norm_p:.1f}%)", indent=2, height=3.5, bullet_char_override="-")
        pdf.write_paragraph(f"- Channels More Similar to Alzheimer's Ref: {alz_c} ({alz_p:.1f}%)", indent=2, height=3.5, bullet_char_override="-")
        pdf.ln(1)
        pdf.set_font('Helvetica', 'I', 7.5); pdf.set_text_color(*pdf.text_color_light)
        pdf.write_paragraph("Disclaimer: This analysis compares overall signal shapes using Dynamic Time Warping (DTW) against reference patterns and does not constitute a medical diagnosis. Results indicate pattern similarity, not disease presence. Consult a healthcare professional for diagnosis.", height=3)
        pdf.ln(2.5)
        
        plotted_ch_idx = similarity_data.get('plotted_channel_index')
        sim_plot_title_text = f"Channel {plotted_ch_idx + 1 if plotted_ch_idx is not None else 'Default'} Comparison Plot"
        pdf.add_image_section(sim_plot_title_text, similarity_data.get('plot_base64'))
    else: pdf.write_paragraph(f"(Similarity Analysis Error: {similarity_data.get('error', 'Data N/A') if similarity_data else 'Data N/A'})", font_style='I', indent=3, font_size=8)
    pdf.ln(3)

    est_height_stats_section = 40
    if pdf.get_y() + est_height_stats_section > pdf.h - pdf.b_margin : pdf.add_page()
    pdf.section_title("Descriptive Statistics")
    if stats_data and not stats_data.get('error'):
        pdf.set_font("Helvetica",'B',9.5); pdf.set_text_color(*pdf.text_color_dark)
        pdf.write(5,"Avg Relative Band Power (%):"); pdf.ln(4); pdf.set_font("Helvetica",'',8.5)
        avg_power = stats_data.get('avg_band_power',{});
        for band, powers_dict in avg_power.items():
            rel_power = powers_dict.get('relative');
            rel_power_str = f"{rel_power * 100:.2f}%" if isinstance(rel_power, (float, int)) else "N/A"
            pdf.write_paragraph(f"- {band.capitalize()}: {rel_power_str}", indent=2, height=3.5, bullet_char_override="-")
        
        std_devs = stats_data.get('std_dev_per_channel')
        if std_devs and isinstance(std_devs, list):
            pdf.ln(1.5); pdf.set_font("Helvetica",'B',9.5); pdf.set_text_color(*pdf.text_color_dark)
            pdf.write(5,"Std. Deviation per Channel (uV):"); pdf.ln(4); pdf.set_font("Helvetica",'',7.5)
            pdf.multi_cell(0, 3.5, ", ".join([f"{s:.2f}" if isinstance(s, (float,int)) else "N/A" for s in std_devs]), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else: pdf.write_paragraph(f"(Statistics Error: {stats_data.get('error', 'Data N/A') if stats_data else 'Data N/A'})", font_style='I', indent=3, font_size=8)
    pdf.ln(3)
    
    img_section_h_estimate = 80 # Title + image + spacing
    if pdf.get_y() + img_section_h_estimate > pdf.h - pdf.b_margin: pdf.add_page()
    pdf.section_title("Standard Visualizations")
    pdf.add_image_section("Stacked Time Series", ts_img_data)
    pdf.ln(1)

    if pdf.get_y() + img_section_h_estimate > pdf.h - pdf.b_margin: # Check again for the second plot
        pdf.add_page()
        pdf.section_title("Standard Visualizations") # Re-add title if it's a new page
    pdf.add_image_section("Average Power Spectral Density (PSD)", psd_img_data)
    pdf.ln(3)
    
    if pdf.get_y() + 30 > pdf.h - pdf.b_margin: pdf.add_page()
    pdf.section_title("Report Disclaimer & Methodology Note")
    pdf.write_paragraph("This report is generated by an AI model (ADFormer) analyzing EEG data for pattern recognition potentially associated with Alzheimer's disease. It is intended for technical review and to supplement clinical assessment by qualified professionals. This report IS NOT A MEDICAL DIAGNOSIS. The 'Internal Consistency' metrics reflect the model's stability on this specific sample by comparing segment-wise predictions to the overall file prediction. DTW analysis compares signal morphology against reference patterns. Descriptive statistics summarize spectral power distribution. Visualizations provide qualitative signal overview. All interpretations should be made by qualified personnel considering the full clinical context.", height=3.5, font_size=8)


# --- Patient PDF Content Builder ---
def _build_patient_pdf_report_content(pdf: PatientPDFReport, prediction_data, similarity_data, consistency_metrics, similarity_plot_data):
    pdf.add_page() # Start fresh
    pdf.section_title("Analysis Summary") # Use the simplified section_title
    created_at_str = pd.to_datetime(prediction_data.get('created_at')).strftime('%B %d, %Y') if prediction_data.get('created_at') else 'N/A'
    pdf.key_value_pair("File Analyzed", prediction_data.get('filename', 'N/A'))
    pdf.key_value_pair("Date of Analysis", created_at_str); pdf.ln(6)
    
    pdf.add_explanation_box( # This method needs to be in BasePDFReport
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
        pdf.add_image_section(plot_title, similarity_plot_data)
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


# --- Predict Endpoint (Logic for handling request and orchestrating calls) ---
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
        except Exception as e: print(f"ERR PatientPDFUpload: {e}"); report_generation_errors.append("PatientPDFURL")

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