# backend/app.py
import os
import uuid
import json
import subprocess
import io # Ensure BytesIO is imported
import base64 # Ensure base64 is imported
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
        self.report_title = "EEG Analysis Report"
        self.logo_path = os.path.join(BACKEND_DIR, "logo.png") 
        # General colors - can be overridden or extended by subclasses or specific functions
        self.base_colors = {
            'primary': (52, 73, 94), 
            'secondary': (74, 144, 226), 
            'accent': (80, 227, 194),   
            'text_dark': (40, 40, 40), 
            'text_light': (100, 100, 100), 
            'text_normal': (0, 0, 0),
            'line': (200, 200, 200), 
            'card_bg': (245, 249, 252),
            'highlight_alz': (200, 50, 50), 
            'highlight_norm': (30, 150, 80),
            'warning_bg': (255, 243, 205), 
            'warning_text': (133, 100, 4) 
        }
        self.page_margin = 10 # Define a consistent page margin

    def header(self): # Retained from previous version, assuming logo and base title are fine
        try:
            if os.path.exists(self.logo_path): 
                self.image(self.logo_path, x=self.page_margin, y=8, h=12)
                self.set_x(self.page_margin + 30 + 5) 
            else: 
                self.set_x(self.page_margin) 
            self.set_font('Helvetica', 'B', 18)
            self.set_text_color(*self.base_colors['primary'])
            remaining_width = self.w - self.get_x() - self.r_margin 
            self.multi_cell(remaining_width, 10, self.report_title, border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*self.base_colors['text_normal'])
            self.ln(3) 
            self.set_draw_color(*self.base_colors['accent'])
            self.set_line_width(0.6)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(7) 
        except Exception as e: print(f"PDF Header Error: {e}")

    def footer(self): # Retained from previous version
        try:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(*self.base_colors['text_light'])
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', border=0, align='C')
            # Position for timestamp on the right
            self.set_xy(self.w - self.r_margin - 70, -15) # Adjust 70 as needed for width of timestamp
            self.cell(70, 10, f'Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}', border=0, align='R')
            self.set_text_color(*self.base_colors['text_normal'])
        except Exception as e: print(f"PDF Footer Error: {e}")
        
    # Default section_title, can be overridden by subclasses
    def section_title(self, title, color=None, icon_char=""):
        if color is None: color = self.base_colors['secondary']
        try:
            self.ln(2)
            self.set_font('Helvetica', 'B', 13)
            self.set_fill_color(*color) 
            self.set_text_color(255,255,255)
            icon_prefix = f"{icon_char} " if icon_char else " " 
            self.cell(0, 9, f"{icon_prefix}{title}", border=0, align='L', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*self.base_colors['text_normal'])
            self.ln(6) 
        except Exception as e: print(f"PDF Section Title Error: {e}")

    def key_value_pair(self, key, value_str, key_width=60, is_bold_value=False, value_color=None, text_color_key=None):
        if text_color_key is None: text_color_key = self.base_colors['text_dark']
        if value_color is None: value_color = self.base_colors['text_normal']
        
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(*text_color_key)
        key_start_y = self.get_y()
        self.multi_cell(key_width, 6, str(key)+":", align='L', new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=self.font_size)
        
        self.set_y(key_start_y)
        self.set_x(self.l_margin + key_width + 2) 
        value_width = self.w - self.l_margin - self.r_margin - key_width - 2 
        font_style_val = 'B' if is_bold_value else ''
        self.set_font('Helvetica', font_style_val, 9.5) 
        self.set_text_color(*value_color)
        self.multi_cell(value_width, 6, str(value_str), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*self.base_colors['text_normal'])
        self.ln(1.5) 

    def write_paragraph(self, text, height=5, indent=0, font_style='', font_size=9.5, text_color=None, bullet_char=None, align='L'):
        try:
             self.set_font('Helvetica', font_style, font_size)
             if text_color: self.set_text_color(*text_color)
             else: self.set_text_color(*self.base_colors['text_dark']) 
             current_x_start = self.l_margin + indent
             self.set_x(current_x_start)
             if bullet_char: 
                 self.set_font('Helvetica', 'B', font_size + 1) 
                 self.cell(self.get_string_width(bullet_char) + 1, height, bullet_char) 
                 self.set_x(current_x_start + self.get_string_width(bullet_char) + 2) 
                 self.set_font('Helvetica', font_style, font_size) 
                 self.multi_cell(self.w - self.get_x() - self.r_margin, height, text, align=align, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
             else: 
                 self.multi_cell(0, height, text, align=align, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
             self.ln(height / 3)
             self.set_text_color(*self.base_colors['text_normal']) 
        except Exception as e: print(f"PDF write_paragraph Error: {e}")

    def add_image_section(self, title, image_data_base64, caption="", icon_char="🖼️", title_color=None): # Base version
        if title_color is None: title_color = self.base_colors['secondary']
        self.section_title(title, color=title_color, icon_char=icon_char)
        if image_data_base64 and isinstance(image_data_base64, str) and image_data_base64.startswith('data:image/png;base64,'):
            try:
                img_bytes = base64.b64decode(image_data_base64.split(',')[1])
                img_file = io.BytesIO(img_bytes)
                page_content_width = self.w - self.l_margin - self.r_margin
                img_display_width = page_content_width * 0.85
                x_pos = self.l_margin + (page_content_width - img_display_width) / 2 
                self.image(img_file, x=x_pos, w=img_display_width)
                img_file.close()
                self.ln(4)
                if caption: 
                    self.set_font('Helvetica', 'I', 8.5)
                    self.set_text_color(*self.base_colors['text_light'])
                    self.set_x(self.l_margin + (page_content_width - (page_content_width * 0.9)) / 2) 
                    self.multi_cell(page_content_width * 0.9, 5, caption, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_text_color(*self.base_colors['text_normal'])
            except Exception as e: 
                self.set_font("Helvetica",'I',9); self.set_text_color(200,0,0) 
                self.cell(0,8,f"(Error embedding image '{title}': {str(e)[:100]})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_text_color(*self.base_colors['text_normal']); print(f"PDF Image Embed Error for {title}: {e}")
        else: 
            self.write_paragraph("(Image data not available or in invalid format)", font_style='I', indent=5)
        self.ln(6)

    def add_explanation_box(self, title, text_lines, icon_char="💡", bg_color=None, title_color=None, text_color_override=None, font_size_text=9, border_color=None):
        if bg_color is None: bg_color = (235, 245, 255)
        if title_color is None: title_color = self.base_colors['primary']
        if text_color_override is None: text_color_override = self.base_colors['text_dark']
        if border_color is None: border_color = self.base_colors['line']
        
        self.ln(2)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*title_color)
        self.cell(0, 8, f"{icon_char} {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.set_fill_color(*bg_color)
        self.set_draw_color(*border_color)
        y_before = self.get_y()
        
        # Estimate height
        est_height = 3 # top padding
        line_h = 5 # Base line height for font_size_text
        temp_pdf = FPDF() # Use a temporary PDF for width calculation without affecting main PDF state
        temp_pdf.set_font('Helvetica', '', font_size_text)
        
        for line_item in text_lines:
            is_list_item = isinstance(line_item, tuple) and len(line_item) == 2 and line_item[0] == "bullet"
            text_to_measure = line_item[1] if is_list_item else line_item
            is_sub_bullet = isinstance(text_to_measure, tuple) and len(text_to_measure) == 2 and text_to_measure[0] == "sub_bullet"
            final_text_to_measure = text_to_measure[1] if is_sub_bullet else text_to_measure
            
            temp_pdf.set_font('Helvetica', 'B' if "**" in final_text_to_measure else '', font_size_text)
            text_width_available = self.w - self.l_margin - self.r_margin - 8 # base padding for box
            if is_list_item: text_width_available -= (5 if not is_sub_bullet else 10)

            # Simplified height estimation - this is tricky with multi_cell, direct calculation might be better
            # For fpdf2, multi_cell with split_only=True is the way
            num_lines_for_this_text = len(self.multi_cell(text_width_available, line_h, final_text_to_measure.replace("**",""), split_only=True, dry_run=True))
            est_height += num_lines_for_this_text * line_h + (1 if is_list_item else 0.5) # Add some inter-line/bullet spacing
        est_height += 3 # bottom padding
        
        self.rect(self.l_margin, y_before, self.w - self.l_margin - self.r_margin, est_height, 'DF') 
        self.set_y(y_before + 3) 
        
        for line_item in text_lines: 
            is_list_item = isinstance(line_item, tuple) and len(line_item) == 2 and line_item[0] == "bullet"
            text_content = line_item[1] if is_list_item else line_item
            is_sub_bullet = isinstance(text_content, tuple) and len(text_content) == 2 and text_content[0] == "sub_bullet"
            final_text_content = text_content[1] if is_sub_bullet else text_content
            
            parts = final_text_content.split("**")
            base_indent = 3
            current_x = self.l_margin + base_indent
            
            self.set_x(current_x)

            if is_list_item and not is_sub_bullet:
                self.set_font('Helvetica', 'B', font_size_text + 2)
                self.set_text_color(*title_color) 
                self.cell(5, line_h, "•") 
                current_x += 5 + 2 # bullet width + space
            elif is_sub_bullet:
                self.set_x(current_x + 5) # Indent for sub-bullet
                self.set_font('Helvetica', '', font_size_text + 1)
                self.set_text_color(*text_color_override)
                self.cell(5, line_h, "◦")
                current_x += 5 + 5 + 2 # parent bullet + sub bullet indent + space

            self.set_x(current_x)
            available_width_for_text = self.w - current_x - self.r_margin
            
            for i, part in enumerate(parts): 
                is_bold_part = i % 2 == 1
                self.set_font('Helvetica', 'B' if is_bold_part else '', font_size_text)
                self.set_text_color(*(self.base_colors['primary'] if is_bold_part else text_color_override))
                # Use multi_cell for better wrapping within the explanation box for each part
                # This requires managing y position more carefully if parts wrap.
                # For simplicity, using write here, assuming parts are not excessively long.
                # For robust wrapping of bolded parts, you might need to calculate width of each part.
                self.write(line_h, part)
            
            self.ln(line_h + (0.5 if is_list_item else 0.2)) # Ensure YPos.NEXT semantics
            if is_list_item: self.set_y(self.get_y() - (line_h * 0.3)) 
            
        self.set_y(y_before + est_height)
        self.ln(5)


# --- Technical PDF Report Class (with new helper methods) ---
class TechnicalPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Comprehensive EEG Analysis Report"
        # Define the specific color palette for this report type if it differs significantly
        # or inherit from BasePDFReport and use specific colors in _build_technical_pdf_report_content

    # Specific section_title for Technical Report (as per user's code)
    def section_title(self, title, color=(0, 51, 102)): # Default color matches user's 'heading'
        """Enhanced section title with consistent styling"""
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(*color)
        self.cell(0, 8, title, 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT) # Ensure new_x, new_y
        
        y = self.get_y() - 2 # Position underline slightly above the baseline of the cell
        self.set_draw_color(*color) # Underline with the same color as title
        self.set_line_width(0.3) # Thinner underline
        self.line(self.l_margin, y, self.w - self.r_margin, y) # Use r_margin
        self.ln(4)

    def rounded_rect(self, x, y, w, h, r, style=''):
        """Draw a rounded rectangle (from user's code)"""
        k = self.k # FPDF scale factor
        hp = self.h # Page height
        if style == 'F':
            op = 'f'
        elif style == 'FD' or style == 'DF':
            op = 'B'
        else:
            op = 'S'
        
        my_arcs = []
        # Each arc is: x, y, rx, ry, start_angle, end_angle, clockwise (1) or counter-clockwise (0)
        # Top-left
        my_arcs.append(((x + r), (y + r), r, r, 180, 270))
        # Top-right
        my_arcs.append(((x + w - r), (y + r), r, r, 270, 360))
        # Bottom-right
        my_arcs.append(((x + w - r), (y + h - r), r, r, 0, 90))
        # Bottom-left
        my_arcs.append(((x + r), (y + h - r), r, r, 90, 180))

        self._out('q') # Save state
        self._set_color_op(op)

        self._out(f'{x + r:.2f} {(hp - y):.2f} m') # Move to start of top line (after arc)
        self._out(f'{x + w - r:.2f} {(hp - y):.2f} l') # Top line
        self._draw_arc(my_arcs[1], k, hp) # Top-right arc

        self._out(f'{x + w:.2f} {(hp - (y + h - r)):.2f} l') # Right line
        self._draw_arc(my_arcs[2], k, hp) # Bottom-right arc

        self._out(f'{x + r:.2f} {(hp - (y + h)):.2f} l') # Bottom line
        self._draw_arc(my_arcs[3], k, hp) # Bottom-left arc
        
        self._out(f'{x:.2f} {(hp - (y + r)):.2f} l') # Left line
        self._draw_arc(my_arcs[0], k, hp) # Top-left arc
        
        self._out(op) # Perform drawing operation
        self._out('Q') # Restore state


    def _draw_arc(self, arc, k, hp):
        """Helper to draw an arc for rounded_rect, compatible with FPDF2"""
        xc, yc, rx, ry, start_angle, end_angle = arc
        # FPDF uses counter-clockwise angles from East (3 o'clock)
        # We need to adjust the angles and potentially the logic if they were for a different system
        # For FPDF's ellipse: x, y (center), rx, ry, angle_start, angle_end
        # This FPDF ellipse is not directly used for rounded corners in the rect method above.
        # The `rounded_rect` provided by user uses path drawing commands (m, l, c).
        # The 'c' command (Bezier curve) needs careful calculation for arcs.
        # The user's `_out(f'{x+r*k} {y+h} {x+r} {y+h} {x+r} {y+h} c')` type of commands are for Bezier curves.
        # For simplicity, if the user's rounded_rect path commands work, we keep them.
        # If an actual arc drawing primitive is needed, `self.ellipse` is available but works differently.
        # The provided `rounded_rect` appears to be constructing Bezier curves for corners.
        pass # The logic is embedded in the user's `rounded_rect` path commands.


    def add_enhanced_image_section(self, title, img_data, colors):
        """Add an image with enhanced styling (from user's code)"""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*colors.get('subheading', self.base_colors['text_dark']))
        self.cell(0, 6, title, 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT) # Ensure new_x, new_y
        
        if img_data and img_data.startswith('data:image/png;base64,'):
            img_y_start = self.get_y()
            img_height_approx = 50 # Approximate image height
            container_height = img_height_approx + 4 # Padding

            # Check for page break before drawing container and image
            if self.get_y() + container_height > self.h - self.b_margin:
                self.add_page()
                img_y_start = self.get_y() # Update y after page break

            self.set_fill_color(248, 248, 252) # Light gray-blue background
            self.rect(self.l_margin, img_y_start, self.w - 2*self.l_margin, container_height, 'F')
            
            try:
                img_bytes = base64.b64decode(img_data.split(',')[1])
                img_file = io.BytesIO(img_bytes)
                
                # Calculate dimensions to fit image, maintaining aspect ratio
                img_orig_w, img_orig_h = FPDF.IMAGE_TYPE_BY_EXTENSION[".png"](img_file) # Needs fpdf2 Pillow support or manual image parsing
                img_file.seek(0) # Reset pointer after getting dimensions

                max_w = self.w - 2*self.l_margin - 10 # Max width with padding
                max_h = container_height - 4          # Max height with padding

                ratio = min(max_w / img_orig_w, max_h / img_orig_h)
                img_disp_w = img_orig_w * ratio
                img_disp_h = img_orig_h * ratio
                
                # Center image within the padded area
                img_x_pos = self.l_margin + 5 + (max_w - img_disp_w) / 2
                img_y_pos = img_y_start + 2 + (max_h - img_disp_h) / 2

                self.image(img_file, x=img_x_pos, y=img_y_pos, w=img_disp_w, h=img_disp_h)
                img_file.close()
            except Exception as e:
                self.set_xy(self.l_margin + 5, img_y_start + (container_height / 2) - 5)
                self.set_font('Helvetica', 'I', 9)
                self.set_text_color(*colors.get('light_text', (128,128,128)))
                self.cell(0, 10, f"Error displaying image: {str(e)[:50]}...", 0, 1, 'L')
            self.set_y(img_y_start + container_height) # Move Y below the image container
        else:
            self.set_font('Helvetica', 'I', 9)
            self.set_text_color(*colors.get('light_text', (128,128,128)))
            self.cell(0, 10, "Image data not available", 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.ln(2)

    def metric_card_item(self, title, value, description, width, height, colors_dict=None):
        """Create an enhanced metric card (from user's code, adapted)"""
        # Using colors from the provided `colors` dict in _build_technical_pdf_report_content
        # This method assumes it's called within a context where `self.l_margin` and `self.get_x/y` are valid
        if colors_dict is None: # Fallback if not passed
            colors_dict = {
                'subheading': self.base_colors['primary'],
                'highlight': self.base_colors['accent'],
                'light_text': self.base_colors['text_light']
            }

        card_x, card_y = self.get_x(), self.get_y()
        
        self.set_fill_color(245, 245, 250) # Light blue from user's metric grid
        # Use the `rounded_rect` method if you want rounded corners for these cards too.
        # self.rounded_rect(card_x, card_y, width, height, 2, 'F')
        self.rect(card_x, card_y, width, height, 'F') # Simpler rectangle for now

        # Title
        self.set_xy(card_x + 2, card_y + 2)
        self.set_font('Helvetica', 'B', 9) # Slightly smaller for card title
        self.set_text_color(*colors_dict.get('subheading'))
        self.multi_cell(width - 4, 5, title, 0, 'L') # Use multi_cell for potential wrapping
        
        # Value - position after title
        current_y_val = self.get_y() 
        self.set_xy(card_x + 2, current_y_val)
        self.set_font('Helvetica', 'B', 11) # Value font size
        self.set_text_color(*colors_dict.get('highlight'))
        self.multi_cell(width - 4, 7, value, 0, 'L') # Use multi_cell

        # Description - position at bottom of card
        desc_y_pos = card_y + height - 7 # Position description near bottom
        self.set_xy(card_x + 2, desc_y_pos)
        self.set_font('Helvetica', 'I', 7) # Description font size
        self.set_text_color(*colors_dict.get('light_text'))
        self.multi_cell(width - 4, 3, description, 0, 'L') # Use multi_cell

        self.set_text_color(*self.base_colors['text_normal']) # Reset text color


# --- Patient PDF Report Class ---
class PatientPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Your EEG Pattern Analysis Report"

# --- Helper Functions (Retained from previous version) ---
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


# --- New Technical PDF Content Builder ---
def _build_technical_pdf_report_content(pdf: TechnicalPDFReport, prediction_data, stats_data, similarity_data, consistency_metrics, ts_img_data, psd_img_data):
    pdf.set_auto_page_break(auto=True, margin=15) # Consistent margin
    pdf.add_page()
    
    # Color definitions for better visual appeal (as provided by user)
    colors = {
        'normal': (0, 128, 0),     # Green
        'alzheimers': (220, 50, 50), # Red
        'heading': (0, 51, 102),    # Dark blue
        'subheading': (70, 70, 130), # Medium blue
        'text': (60, 60, 60),       # Dark gray
        'light_text': (100, 100, 100), # Light gray
        'highlight': (70, 130, 180)  # Steel blue
    }
    
    # HEADER SECTION WITH BRANDING
    pdf.set_fill_color(240, 240, 250)  # Light blue background
    pdf.rect(pdf.page_margin, pdf.page_margin, pdf.w - 2*pdf.page_margin, 15, 'F')
    pdf.set_xy(pdf.page_margin, pdf.page_margin)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(*colors['heading'])
    pdf.cell(pdf.w - 2*pdf.page_margin, 15, "EEG Analysis Report", 0, 1, 'C')
    pdf.ln(5)
    
    # ANALYSIS DETAILS SECTION
    pdf.section_title("Analysis Details", color=colors['heading']) # Use the class method
    pdf.set_fill_color(245, 245, 245)  # Light gray background for info box
    details_y = pdf.get_y()
    # Check for page break before drawing rect
    if details_y + 16 > pdf.h - pdf.b_margin: pdf.add_page(); details_y = pdf.get_y()
    pdf.rect(pdf.l_margin, details_y, pdf.w - 2*pdf.l_margin, 16, 'F')
    pdf.set_xy(pdf.l_margin + 2, details_y + 2)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*colors['subheading'])
    pdf.cell(30, 6, "Filename:", 0, 0)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*colors['text'])
    pdf.cell(pdf.w - 2*pdf.l_margin - 32, 6, prediction_data.get('filename', 'N/A'), 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT) # ensure new_x, new_y
    
    pdf.set_x(pdf.l_margin + 2)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*colors['subheading'])
    pdf.cell(30, 6, "Analyzed On:", 0, 0)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*colors['text'])
    created_at_str = pd.to_datetime(prediction_data.get('created_at')).strftime('%Y-%m-%d %H:%M:%S UTC') if prediction_data.get('created_at') else 'N/A'
    pdf.cell(pdf.w - 2*pdf.l_margin - 32, 6, created_at_str, 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(details_y + 16) # Ensure Y is below the rect
    pdf.ln(4) # Reduced from 8 for tighter spacing
    
    # PREDICTION SECTION
    # Check for page break before section
    if pdf.get_y() + 40 > pdf.h - pdf.b_margin: pdf.add_page()
    pdf.section_title("Prediction Results", color=colors['heading'])
    
    prediction_label = prediction_data.get('prediction', 'N/A')
    is_alz = prediction_label == "Alzheimer's"
    pred_color_rgb = colors['alzheimers'] if is_alz else colors['normal']
    
    # Prediction box
    box_width = pdf.w - 2*pdf.l_margin
    box_height = 20 # Reduced height
    box_y_start = pdf.get_y()
    # Check for page break before drawing rect for prediction
    if box_y_start + box_height > pdf.h - pdf.b_margin: pdf.add_page(); box_y_start = pdf.get_y()
    
    pdf.set_fill_color(pred_color_rgb[0], pred_color_rgb[1], pred_color_rgb[2]) 
    pdf.set_alpha(0.15) # Set alpha after color
    pdf.rect(pdf.l_margin, box_y_start, box_width, box_height, 'F')
    pdf.set_alpha(1.0) # Reset alpha
    
    # Prediction text
    pdf.set_xy(pdf.l_margin + 5, box_y_start + 2) # Adjusted y for less padding
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(*pred_color_rgb)
    pdf.cell(box_width - 10, 8, f"Overall Assessment: {prediction_label}", 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Confidence text
    probabilities = prediction_data.get('probabilities')
    prob_str = 'N/A'
    if isinstance(probabilities, list) and len(probabilities) == 2:
        try: 
            prob_str = f"Confidence: Normal: {probabilities[0]*100:.1f}%, Alzheimer's: {probabilities[1]*100:.1f}%"
        except: 
            prob_str = str(probabilities) # Fallback
    
    pdf.set_x(pdf.l_margin + 5)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*colors['text']) # Use standard text color for confidence
    pdf.cell(box_width - 10, 7, prob_str, 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(box_y_start + box_height) # Move below the box
    pdf.ln(4) # Reduced spacing
    
    # CONSISTENCY METRICS SECTION
    est_height_consistency = 70 # Estimate height needed for section title + text + grid
    if pdf.get_y() + est_height_consistency > pdf.h - pdf.b_margin: pdf.add_page()
    
    pdf.section_title("Analysis Consistency", color=colors['heading'])
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*colors['text'])
    pdf.write_paragraph("These metrics show how consistent the model's findings were across different segments of the recording, using the overall assessment as a reference.", height=4, align='L')
    pdf.ln(2) # Reduced spacing
    
    if consistency_metrics and not consistency_metrics.get('error') and isinstance(consistency_metrics.get('num_trials'), int) and consistency_metrics.get('num_trials', 0) > 0:
        metrics_data = consistency_metrics
        
        metric_items = [
            ("Overall Agreement", f"{metrics_data.get('accuracy', 0)*100:.1f}%", "Segments matching overall finding"),
            ("Precision (Alz.)", f"{metrics_data.get('precision', 0):.3f}", "Correct Alz. prediction rate"),
            ("Sensitivity (Alz.)", f"{metrics_data.get('recall_sensitivity', 0):.3f}", "Alz. pattern detection rate"),
            ("F1-Score (Alz.)", f"{metrics_data.get('f1_score', 0):.3f}", "Balanced Alz. measure"),
            ("Specificity (Normal)", f"{metrics_data.get('specificity', 0):.3f}", "Correct Normal prediction rate"),
            ("Segments Checked", str(metrics_data.get('num_trials', 'N/A')), "Total segments analyzed")
        ]
        
        initial_y_metrics = pdf.get_y()
        
        # Define grid parameters
        num_metric_cols = 3
        metric_col_width = (pdf.w - 2*pdf.l_margin - (num_metric_cols - 1) * 2) / num_metric_cols # 2 for spacing
        metric_cell_h = 18 # Height for each metric card

        for i, (title, value, desc) in enumerate(metric_items):
            col = i % num_metric_cols
            row = i // num_metric_cols
            
            current_x_metric = pdf.l_margin + col * (metric_col_width + 2) # 2 for spacing
            current_y_metric = initial_y_metrics + row * (metric_cell_h + 2) # 2 for spacing

            # Check for page break before drawing each metric card
            if current_y_metric + metric_cell_h > pdf.h - pdf.b_margin:
                pdf.add_page()
                pdf.section_title("Analysis Consistency (Continued)", color=colors['heading'])
                initial_y_metrics = pdf.get_y() # Reset initial_y for the new page
                current_y_metric = initial_y_metrics # Current card starts at top of new section
                # Note: This might orphan the section intro text if all cards move. Handle this by checking total estimated height before starting the grid.
            
            pdf.set_xy(current_x_metric, current_y_metric)
            pdf.metric_card_item(title, value, desc, metric_col_width, metric_cell_h, colors) # Using the new method

        # Set Y position after the last row of metric cards
        num_rows = (len(metric_items) + num_metric_cols - 1) // num_metric_cols
        pdf.set_y(initial_y_metrics + num_rows * (metric_cell_h + 2))
        pdf.ln(1) # Small space after grid
        
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(*colors['light_text'])
        cm_text = (f"Technical details (Alz. as positive class): TP: {metrics_data.get('true_positives','?')} | "
                   f"TN: {metrics_data.get('true_negatives','?')} | FP: {metrics_data.get('false_positives','?')} | "
                   f"FN: {metrics_data.get('false_negatives','?')}")
        pdf.multi_cell(0, 4, cm_text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.write_paragraph("Consistency metrics not available for this analysis.", font_style='I', font_size=9)
    pdf.ln(3)
    
    # SIGNAL SIMILARITY SECTION
    est_height_similarity = 80 # Approximate height for this section (text + plot)
    if pdf.get_y() + est_height_similarity > pdf.h - pdf.b_margin: pdf.add_page()
    pdf.section_title("Signal Pattern Analysis", color=colors['heading'])
    
    if similarity_data and not similarity_data.get('error'):
        sim_box_y_start = pdf.get_y()
        # Estimate height for similarity text content (approx 25-30)
        if sim_box_y_start + 30 > pdf.h - pdf.b_margin: pdf.add_page(); sim_box_y_start = pdf.get_y()

        pdf.set_fill_color(240, 248, 255)  # Light blue background
        pdf.rect(pdf.l_margin, sim_box_y_start, pdf.w - 2*pdf.l_margin, 28, 'F') # Adjusted height
        
        pdf.set_xy(pdf.l_margin + 5, sim_box_y_start + 2)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*colors['subheading'])
        overall_assessment_sim = similarity_data.get('overall_similarity', 'N/A')
        pdf.multi_cell(pdf.w - 2*pdf.l_margin - 10, 5, f"Overall Pattern Assessment: {overall_assessment_sim}", 0, 'L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        n_ch_total = similarity_data.get('total_channels', 0)
        norm_c = similarity_data.get('normal_closer_count', 0)
        alz_c = similarity_data.get('alz_closer_count', 0)
        norm_p = (norm_c / n_ch_total * 100) if n_ch_total > 0 else 0
        alz_p = (alz_c / n_ch_total * 100) if n_ch_total > 0 else 0
        
        pdf.set_x(pdf.l_margin + 5)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(*colors['normal'])
        pdf.multi_cell(pdf.w - 2*pdf.l_margin - 10, 4, f"Channels similar to Normal pattern: {norm_c} ({norm_p:.1f}%)", 0, 'L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_x(pdf.l_margin + 5)
        pdf.set_text_color(*colors['alzheimers'])
        pdf.multi_cell(pdf.w - 2*pdf.l_margin - 10, 4, f"Channels similar to Alzheimer's pattern: {alz_c} ({alz_p:.1f}%)", 0, 'L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_x(pdf.l_margin + 5)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(*colors['light_text'])
        pdf.multi_cell(pdf.w - 2*pdf.l_margin - 10, 3.5, "Higher similarity implies closer match to reference patterns (Dynamic Time Warping).", 0, 'L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_y(sim_box_y_start + 28) # Move below the box
        pdf.ln(3)
        
        plotted_ch_idx_sim = similarity_data.get('plotted_channel_index')
        sim_plot_title_disp = f"Channel {plotted_ch_idx_sim + 1 if plotted_ch_idx_sim is not None else 'Default'} Comparison"
        # Check for page break before image
        if pdf.get_y() + 55 > pdf.h - pdf.b_margin: pdf.add_page(); pdf.section_title("Signal Pattern Analysis (Continued)", color=colors['heading'])
        pdf.add_enhanced_image_section(sim_plot_title_disp, similarity_data.get('plot_base64'), colors)
        
        pdf.ln(1) # Reduced spacing
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.set_text_color(*colors['light_text'])
        pdf.multi_cell(0, 3.5, "Note: DTW analysis compares signal shapes. This is a technical assessment, not a clinical diagnosis.", align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else: 
        pdf.write_paragraph("Signal pattern analysis data not available.", font_style='I', font_size=9)
    pdf.ln(3)
    
    # BRAIN WAVE STATISTICS SECTION
    est_height_stats = 85 # Approximate height
    if pdf.get_y() + est_height_stats > pdf.h - pdf.b_margin: pdf.add_page()
    pdf.section_title("Brain Wave Statistics", color=colors['heading'])
    
    if stats_data and not stats_data.get('error'):
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(*colors['text'])
        pdf.multi_cell(0, 4, "Brain waves are categorized by frequency. Different states and conditions are associated with varying power in these bands.", align='L')
        pdf.ln(2)
        
        avg_power_stats = stats_data.get('avg_band_power', {})
        if avg_power_stats:
            band_colors_stats = {
                'delta': (70, 90, 150), 'theta': (100, 145, 190),
                'alpha': (70, 150, 70), 'beta': (220, 160, 50),
                'gamma': (200, 80, 80)
            }
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(*colors['subheading'])
            pdf.cell(0, 6, "Relative Brain Wave Activity:", 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            
            max_bar_width_stats = 80 # Adjusted max bar width
            bar_h_stats = 6 # Adjusted bar height
            
            band_desc_stats = {
                'delta': "Deep sleep, rest", 'theta': "Drowsiness, light meditation",
                'alpha': "Relaxed wakefulness", 'beta': "Active thinking, focus",
                'gamma': "Cognitive processing"
            }
            
            for band, powers in avg_power_stats.items():
                # Page break check before drawing each band item
                if pdf.get_y() + bar_h_stats + 2 > pdf.h - pdf.b_margin: # +2 for ln
                     pdf.add_page()
                     pdf.section_title("Brain Wave Statistics (Continued)", color=colors['heading'])

                rel_p = powers.get('relative', 0)
                rel_p_pct = rel_p * 100 if isinstance(rel_p, (float, int)) else 0
                
                pdf.set_font('Helvetica', 'B', 9)
                pdf.set_text_color(*band_colors_stats.get(band.lower(), colors['text']))
                pdf.cell(25, bar_h_stats, band.capitalize(), 0, 0)
                
                pdf.set_font('Helvetica', '', 9)
                pdf.set_text_color(*colors['text'])
                pdf.cell(20, bar_h_stats, f"{rel_p_pct:.1f}%", 0, 0)
                
                bar_x_pos = pdf.get_x()
                bar_y_pos = pdf.get_y() + 1.5 # Center bar vertically a bit
                current_bar_width = (rel_p_pct / 100) * max_bar_width_stats
                pdf.set_fill_color(*band_colors_stats.get(band.lower(), colors['text']))
                pdf.rect(bar_x_pos, bar_y_pos, current_bar_width, bar_h_stats - 3, 'F')
                
                pdf.set_x(bar_x_pos + max_bar_width_stats + 5)
                pdf.set_font('Helvetica', 'I', 8)
                pdf.set_text_color(*colors['light_text'])
                pdf.cell(0, bar_h_stats, band_desc_stats.get(band.lower(), ""), 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            
        std_devs_data = stats_data.get('std_dev_per_channel')
        if std_devs_data and isinstance(std_devs_data, list):
            if pdf.get_y() + 20 > pdf.h - pdf.b_margin: pdf.add_page(); pdf.section_title("Brain Wave Statistics (Continued)", color=colors['heading'])
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(*colors['subheading'])
            pdf.cell(0, 5, "Signal Variability (Standard Deviation per Channel, µV):", 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(*colors['text'])
            std_text_data = ", ".join([f"{s:.2f}" if isinstance(s, (float,int)) else "N/A" for s in std_devs_data])
            pdf.multi_cell(0, 4, std_text_data, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            pdf.set_font('Helvetica', 'I', 8)
            pdf.set_text_color(*colors['light_text'])
            pdf.multi_cell(0, 4, "Indicates signal amplitude variability. Higher values suggest more fluctuation.", align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.write_paragraph("Brain wave statistics not available for this analysis.", font_style='I', font_size=9)
    pdf.ln(3)
    
    # VISUALIZATIONS SECTION
    est_height_viz = 80 # For one plot
    if pdf.get_y() + est_height_viz > pdf.h - pdf.b_margin: pdf.add_page()
    pdf.section_title("Signal Visualizations", color=colors['heading'])
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*colors['text'])
    pdf.multi_cell(0, 4, "Visual representations of the EEG recording and its frequency components.", align='L')
    pdf.ln(2)
    
    if pdf.get_y() + 60 > pdf.h - pdf.b_margin: pdf.add_page(); pdf.section_title("Signal Visualizations (Continued)", color=colors['heading'])
    pdf.add_enhanced_image_section("Time Series Recording", ts_img_data, colors)
    pdf.ln(3)
    
    if pdf.get_y() + 60 > pdf.h - pdf.b_margin: pdf.add_page(); pdf.section_title("Signal Visualizations (Continued)", color=colors['heading'])
    pdf.add_enhanced_image_section("Frequency Analysis (PSD)", psd_img_data, colors)
    pdf.ln(3)
    
    # DISCLAIMER SECTION
    if pdf.get_y() + 25 > pdf.h - pdf.b_margin: pdf.add_page()
    
    pdf.set_draw_color(*colors.get('light_text', (180,180,180))) # Lighter line
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(*colors['text']) # Darker for heading of disclaimer
    pdf.cell(0, 5, "Important Information", 0, 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(*colors['light_text'])
    pdf.multi_cell(0, 3.5, "This report details an analysis of EEG data using computational methods for pattern recognition, potentially associated with Alzheimer's disease. It is intended for technical review and to supplement clinical assessment by qualified professionals. THIS REPORT IS NOT A MEDICAL DIAGNOSIS. Interpretations should be made by healthcare professionals considering the full clinical context.", align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# --- Patient PDF Content Builder (Refined for layman's terms and less "AI" focus) ---
def _build_patient_pdf_report_content(pdf: PatientPDFReport, prediction_data, similarity_data, consistency_metrics, similarity_plot_data):
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    colors = pdf.base_colors # Use colors from BasePDFReport for consistency or define patient-specific ones

    # --- Report Header ---
    pdf.set_font('Helvetica', 'B', 16) # Slightly larger title for patient
    pdf.set_text_color(*colors['primary'])
    pdf.cell(0, 10, "Your EEG Pattern Analysis Report", 0, 1, 'C')
    pdf.ln(5)

    # --- Introduction Box ---
    pdf.add_explanation_box(
        "About This Report",
        [
            "This report shows the results of an analysis of your brainwave (EEG) activity.",
            "The analysis looks for specific patterns by comparing your EEG to many examples.",
            ("bullet", "This is an informational tool. **It is NOT a medical diagnosis.**"),
            ("bullet", "Please discuss these results with your doctor. They can explain what these findings mean for you in the context of your overall health.")
        ],
        icon_char="ℹ️",
        bg_color=(230, 240, 255), # Light, friendly blue
        title_color=colors['primary'],
        font_size_text=9.5,
        border_color=(200,220,250)
    )
    pdf.ln(3)

    # --- Analysis Details ---
    pdf.section_title("Analysis Details", color=colors['secondary'], icon_char="📋")
    created_at_str = pd.to_datetime(prediction_data.get('created_at')).strftime('%B %d, %Y') if prediction_data.get('created_at') else 'N/A'
    pdf.key_value_pair("File Analyzed", prediction_data.get('filename', 'N/A'), key_width=40, text_color_key=colors['text_dark'])
    pdf.key_value_pair("Date of Analysis", created_at_str, key_width=40, text_color_key=colors['text_dark'])
    pdf.ln(5)

    # --- Main Finding Section ---
    if pdf.get_y() + 45 > pdf.h - pdf.b_margin: pdf.add_page() # Check space
    pdf.section_title("Main Finding: Brainwave Pattern Assessment", color=colors['secondary'], icon_char="💡")
    
    prediction_label = prediction_data.get('prediction', 'Not Determined')
    pred_display_text = "Pattern assessment inconclusive"
    pred_color_patient = colors['text_dark']
    explanation_finding = "The analysis could not clearly determine a specific pattern type."

    if prediction_label == "Alzheimer's":
        pred_display_text = "Patterns Suggestive of Alzheimer's Characteristics Observed"
        pred_color_patient = colors['highlight_alz']
        explanation_finding = "The analysis found patterns in your EEG that are sometimes seen in individuals with characteristics associated with Alzheimer's disease. "
    elif prediction_label == "Normal":
        pred_display_text = "Normal Brainwave Patterns Observed"
        pred_color_patient = colors['highlight_norm']
        explanation_finding = "The analysis found that your EEG patterns are similar to those typically considered normal for this type of assessment. "

    pdf.write_paragraph("The analysis of your EEG recording suggests:", font_size=10, height=5)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(*pred_color_patient)
    pdf.multi_cell(0, 7, pred_display_text, border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*colors['text_normal'])
    pdf.ln(2)

    probabilities = prediction_data.get('probabilities')
    confidence_text_patient = "The certainty of this finding was not specifically calculated for this report."
    if isinstance(probabilities, list) and len(probabilities) == 2:
        try:
            conf_val = probabilities[1]*100 if prediction_label == "Alzheimer's" else probabilities[0]*100
            confidence_text_patient = f"The system is **{conf_val:.0f}%** confident that the patterns found align with the main finding above (based on an initial part of your EEG data)."
        except: pass
    
    explanation_finding += confidence_text_patient
    pdf.add_explanation_box("Understanding this Finding", [explanation_finding], icon_char="🧠", font_size_text=9.5, bg_color=(245,249,252), title_color=colors['subheading'])
    pdf.ln(3)

    # --- Consistency Check Section (Simplified) ---
    if pdf.get_y() + 60 > pdf.h - pdf.b_margin: pdf.add_page() # Check space
    pdf.section_title("Consistency of Findings", color=colors['secondary'], icon_char="🤔")
    
    if consistency_metrics and not consistency_metrics.get('error') and isinstance(consistency_metrics.get('num_trials'), int) and consistency_metrics.get('num_trials', 0) > 0:
        num_segments = consistency_metrics.get('num_trials', 'multiple')
        accuracy_val = f"{consistency_metrics.get('accuracy', 0) * 100:.0f}%"

        consistency_explanation = [
            f"To ensure reliability, the analysis looked at your EEG data in **{num_segments} smaller pieces** (segments).",
            ("bullet", f"**Overall Agreement:** The findings from these smaller pieces matched the main result about **{accuracy_val}** of the time."),
            "Higher agreement suggests that similar patterns were seen throughout the recording.",
            "This is an internal quality check for this specific analysis."
        ]
        pdf.add_explanation_box("How Consistent Were the Findings?", consistency_explanation, icon_char="🧐", bg_color=(230,250,230), title_color=colors['highlight_norm'], font_size_text=9)
    else:
        pdf.write_paragraph("Detailed consistency checks were not applicable or did not yield specific summaries for this sample.", font_style='I', font_size=9, text_color=colors['text_light'])
    pdf.ln(5)

    # --- Signal Shape Comparison (Simplified) ---
    if similarity_plot_data or (not similarity_plot_data and pdf.get_y() + 70 > pdf.h - pdf.b_margin): # Approx height for plot + text
        if pdf.get_y() + 70 > pdf.h - pdf.b_margin: pdf.add_page() # Check space
        pdf.section_title("Comparing Your Brainwave Shape", color=colors['secondary'], icon_char="📈")

        if similarity_data and not similarity_data.get('error') and similarity_plot_data:
            plotted_ch_idx_patient = similarity_data.get('plotted_channel_index')
            plot_title_patient = f"Brainwave Shape from Channel {plotted_ch_idx_patient + 1 if plotted_ch_idx_patient is not None else 'Selected'}"
            
            # Check space for image before drawing
            if pdf.get_y() + 60 > pdf.h - pdf.b_margin: pdf.add_page(); pdf.section_title("Comparing Your Brainwave Shape (Continued)", color=colors['secondary'], icon_char="📈")
            
            pdf.add_image_section(plot_title_patient, similarity_plot_data, title_color=colors['subheading']) # Use base add_image_section

            similarity_explanation = [
                "This graph shows a sample of your brainwave activity (white line).",
                "It's compared to typical 'Normal' brainwave shapes (blue dashed line) and patterns sometimes seen with Alzheimer's (red dotted line).",
                "The analysis looks at how closely the *shape* of your brainwave matches these references."
            ]
            overall_sim_patient = similarity_data.get('overall_similarity', '')
            if "Higher Similarity to Alzheimer's Pattern" in overall_sim_patient:
                similarity_explanation.append(("bullet", "Overall, your brainwave shapes showed **more resemblance to the Alzheimer's-like reference patterns** in this comparison."))
            elif "Higher Similarity to Normal Pattern" in overall_sim_patient:
                similarity_explanation.append(("bullet", "Overall, your brainwave shapes showed **more resemblance to the Normal reference patterns** in this comparison."))
            else:
                similarity_explanation.append(("bullet", "The comparison of your brainwave shapes to the references was mixed or inconclusive."))

            pdf.add_explanation_box("Understanding This Graph", similarity_explanation, icon_char="📊", font_size_text=9, bg_color=(240,248,255), title_color=colors['subheading'])
        else:
            pdf.write_paragraph("The brainwave shape comparison graph is not available for this report.", font_style='I', text_color=colors['text_light'])
    pdf.ln(5)
    
    # --- Final Important Information Section ---
    if pdf.get_y() + 50 > pdf.h - pdf.b_margin: pdf.add_page() # Check space
    pdf.section_title("Important Information & Your Next Steps", color=colors['primary'], icon_char="❗")
    
    next_steps_explanation = [
        "This report is based on an analysis of patterns in your EEG data. **It is NOT a medical diagnosis.**",
        "Only a qualified healthcare professional can diagnose medical conditions. They will consider this report along with your full medical history and other tests.",
        ("bullet", "The main finding from this analysis was: **" + pred_display_text + "**."),
        ("bullet", "**Please share this entire report with your doctor or a neurologist.**"),
        ("bullet", "Discuss these findings and any health concerns you have with them."),
        ("bullet", "Follow your doctor's medical advice for any further steps or evaluation.")
    ]
    pdf.add_explanation_box(
        "Please Discuss This Report With Your Doctor",
        next_steps_explanation,
        icon_char="👨‍⚕️", # Consider a doctor/health icon
        bg_color=colors['warning_bg'], 
        title_color=(100, 70, 20), # Darker orange/brown for title
        text_color_override=colors['warning_text'], 
        font_size_text=9.5,
        border_color=(230,160,90) # Softer orange border
    )

# --- Predict Endpoint (Main logic mostly retained, with calls to new PDF builders) ---
@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    user_id = request.form.get('user_id')
    try: 
        channel_index_str = request.form.get('channel_index', '0')
        channel_index_for_plot = int(channel_index_str)
        assert 0 <= channel_index_for_plot <= 18 # Assuming 19 channels (0-18)
    except (ValueError, TypeError, AssertionError): 
        channel_index_for_plot = 0 # Default to first channel if invalid
        print(f"Warning: Invalid or missing channel index '{channel_index_str}'. Defaulting to 0.")

    if not file or not user_id: return jsonify({'error': "Missing 'file' or 'user_id'"}), 400
    if not file.filename or not file.filename.lower().endswith('.npy'): return jsonify({'error': 'Invalid/Missing filename or type (.npy required).'}), 400

    filename_base, file_extension = os.path.splitext(file.filename)
    unique_id = str(uuid.uuid4())
    save_filename = f"{filename_base}_{unique_id}{file_extension}"
    absolute_temp_filepath = os.path.abspath(os.path.join(UPLOAD_FOLDER, save_filename))
    raw_eeg_storage_path = f'raw_eeg/{user_id}/{save_filename}'
    prediction_id = None
    report_generation_errors = []
    similarity_analysis_results = None
    consistency_metrics_results = None # This will be populated by ML output
    
    # For plot data to pass to PDFs
    ts_img_data, psd_img_data, similarity_plot_base64_data = None, None, None
    # For URLs to store in DB
    ts_url, psd_url, similarity_plot_url, technical_pdf_url, patient_pdf_url = None, None, None, None, None
    asset_prefix = "" 

    try:
        print(f"Step 1/2: Processing '{file.filename}' for user '{user_id}'...")
        os.makedirs(os.path.dirname(absolute_temp_filepath), exist_ok=True) 
        file.save(absolute_temp_filepath)
        print(f"File saved temporarily to: {absolute_temp_filepath}")

        # Upload raw EEG to Supabase Storage
        with open(absolute_temp_filepath, 'rb') as f_upload: 
            supabase.storage.from_(RAW_EEG_BUCKET).upload(
                path=raw_eeg_storage_path, 
                file=f_upload, 
                file_options={"content-type": "application/octet-stream", "upsert": "false"}
            )
        print(f"Raw EEG uploaded to Supabase: {raw_eeg_storage_path}")

        # Step 3: Run ML model
        print(f"Step 3: Running ML model on {absolute_temp_filepath}...")
        run_model(absolute_temp_filepath) # This will create output.json in SIDDHI folder
        
        ml_output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_JSON_PATH)
        if not os.path.exists(ml_output_file_path):
            raise FileNotFoundError(f"ML output file '{ml_output_file_path}' not found after script execution.")
        
        with open(ml_output_file_path, 'r') as f:
            ml_output_data = json.load(f)

        prediction_label = "Alzheimer's" if ml_output_data.get('majority_prediction') == 1 else "Normal"
        probabilities = ml_output_data.get('probabilities') # Should be from the first trial
        consistency_metrics_results = ml_output_data.get('consistency_metrics')
        trial_predictions = ml_output_data.get('trial_predictions')


        # Step 4: Insert initial prediction record
        insert_data = {
            "user_id": user_id, "filename": file.filename, 
            "prediction": prediction_label, "eeg_data_url": raw_eeg_storage_path,
            "probabilities": probabilities, "status": "Processing Report Assets",
            "trial_predictions": trial_predictions, # Store individual trial predictions
            "consistency_metrics": consistency_metrics_results # Store full consistency dict
        }
        print(f"Step 4: Inserting prediction record...")
        insert_payload = json.loads(json.dumps(insert_data, cls=NpEncoder, allow_nan=False))
        insert_res = supabase.table('predictions').insert(insert_payload).execute()
        
        if insert_res.data and len(insert_res.data) > 0:
            prediction_id = insert_res.data[0].get('id')
            print(f"DB Insert successful. Prediction ID: {prediction_id}")
        else:
            _cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path)
            raise Exception(f"DB insert failed for prediction record: {getattr(insert_res, 'error', insert_res)}")

        asset_prefix = f"report_assets/{prediction_id}" 

        # --- Step 5: Generate Report Assets ---
        print(f"--- Step 5: Generating Report Assets for ID: {prediction_id} ---")
        prediction_data_for_report, eeg_data, error_msg = get_prediction_and_eeg(prediction_id)
        if error_msg or eeg_data is None:
            raise Exception(f"Cannot load data for report generation: {error_msg or 'No EEG data retrieved'}")

        # Run Similarity Analysis
        similarity_analysis_results = run_similarity_analysis(absolute_temp_filepath, ALZ_REF_PATH, NORM_REF_PATH, channel_index_for_plot)
        similarity_plot_base64_data = similarity_analysis_results.get('plot_base64') if isinstance(similarity_analysis_results, dict) else None
        
        # Generate other visualizations
        stats_json = generate_descriptive_stats(eeg_data, DEFAULT_FS)
        ts_img_data = generate_stacked_timeseries_image(eeg_data, DEFAULT_FS)
        psd_img_data = generate_average_psd_image(eeg_data, DEFAULT_FS)

        # Upload visual assets
        if similarity_plot_base64_data:
            try:
                sim_plot_filename = f"{asset_prefix}/similarity_plot_ch{channel_index_for_plot + 1}.png"
                sim_plot_bytes = base64.b64decode(similarity_plot_base64_data.split(',')[1])
                supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=sim_plot_filename, file=sim_plot_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                similarity_plot_url_res = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(sim_plot_filename)
                similarity_plot_url = similarity_plot_url_res if isinstance(similarity_plot_url_res, str) and similarity_plot_url_res.startswith('http') else None
                if not similarity_plot_url: report_generation_errors.append("SimPlotURLGenFail")
            except Exception as e: print(f"ERR SimPlotUpload: {e}"); report_generation_errors.append("SimPlotUploadErr")
        
        if ts_img_data:
            try:
                ts_filename = f"{asset_prefix}/timeseries.png"; ts_bytes = base64.b64decode(ts_img_data.split(',')[1])
                supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=ts_filename, file=ts_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                ts_url_res = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(ts_filename)
                ts_url = ts_url_res if isinstance(ts_url_res, str) and ts_url_res.startswith('http') else None
                if not ts_url: report_generation_errors.append("TSPlotURLGenFail")
            except Exception as e: print(f"ERR TSPlotUpload: {e}"); report_generation_errors.append("TSPlotUploadErr")
        
        if psd_img_data:
            try:
                psd_filename = f"{asset_prefix}/psd.png"; psd_bytes = base64.b64decode(psd_img_data.split(',')[1])
                supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=psd_filename, file=psd_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                psd_url_res = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(psd_filename)
                psd_url = psd_url_res if isinstance(psd_url_res, str) and psd_url_res.startswith('http') else None
                if not psd_url: report_generation_errors.append("PSDPlotURLGenFail")
            except Exception as e: print(f"ERR PSDPlotUpload: {e}"); report_generation_errors.append("PSDPlotUploadErr")

        # Generate and Upload Technical PDF
        print("Generating Technical PDF report...")
        tech_pdf = TechnicalPDFReport()
        _build_technical_pdf_report_content(tech_pdf, prediction_data_for_report, stats_json, similarity_analysis_results, consistency_metrics_results, ts_img_data, psd_img_data) # Use the new builder
        tech_pdf_bytes = tech_pdf.output(dest='S').encode('latin-1') # Get as bytes
        technical_pdf_filename = f"{asset_prefix}/technical_report.pdf"
        try:
            supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=technical_pdf_filename, file=tech_pdf_bytes, file_options={"content-type": "application/pdf", "upsert": "true"})
            technical_pdf_url_res = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(technical_pdf_filename)
            technical_pdf_url = technical_pdf_url_res if isinstance(technical_pdf_url_res, str) and technical_pdf_url_res.startswith('http') else None
            if not technical_pdf_url: report_generation_errors.append("TechPDFURLGenFail")
        except Exception as e: print(f"ERR TechPDFUpload: {e}"); report_generation_errors.append("TechPDFUploadErr")

        # Generate and Upload Patient PDF
        print("Generating Patient PDF report...")
        patient_pdf = PatientPDFReport()
        _build_patient_pdf_report_content(patient_pdf, prediction_data_for_report, similarity_analysis_results, consistency_metrics_results, similarity_plot_base64_data) # Pass base64 data for plot
        patient_pdf_bytes = patient_pdf.output(dest='S').encode('latin-1') # Get as bytes
        patient_pdf_filename = f"{asset_prefix}/patient_report.pdf"
        try:
            supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=patient_pdf_filename, file=patient_pdf_bytes, file_options={"content-type": "application/pdf", "upsert": "true"})
            patient_pdf_url_res = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(patient_pdf_filename)
            patient_pdf_url = patient_pdf_url_res if isinstance(patient_pdf_url_res, str) and patient_pdf_url_res.startswith('http') else None
            if not patient_pdf_url: report_generation_errors.append("PatientPDFURLGenFail")
        except Exception as e: print(f"ERR PatientPDFUpload: {e}"); report_generation_errors.append("PatientPDFUploadErr")
        
        # Final DB Update
        report_generation_status = "Completed" if not report_generation_errors else f"Completed with errors ({', '.join(report_generation_errors)})"
        db_similarity_data_to_store = {k: v for k, v in similarity_analysis_results.items() if k != 'plot_base64'} if isinstance(similarity_analysis_results, dict) and not similarity_analysis_results.get('error') else similarity_analysis_results
        
        update_data_payload = {
            "stats_data": stats_json, 
            "timeseries_plot_url": ts_url, "psd_plot_url": psd_url, 
            "pdf_report_url": technical_pdf_url, # Retain for backward compatibility if needed, or remove
            "technical_pdf_url": technical_pdf_url, 
            "patient_pdf_url": patient_pdf_url, 
            "report_generated_at": datetime.now(timezone.utc).isoformat(), 
            "status": report_generation_status, 
            "similarity_results": db_similarity_data_to_store, 
            "similarity_plot_url": similarity_plot_url,
        }
        print(f"Step 6: Updating DB for {prediction_id} with status '{report_generation_status}'...")
        update_payload_final_str = json.dumps(update_data_payload, cls=NpEncoder, allow_nan=False)
        update_payload_final = json.loads(update_payload_final_str)
        
        supabase.table('predictions').update(update_payload_final).eq('id', prediction_id).execute()
        
        return jsonify({"filename": file.filename, "prediction": prediction_label, "prediction_id": prediction_id})

    except Exception as e: 
        print(f"ERROR in /api/predict: {e}"); traceback.print_exc()
        error_status_message = f"Failed: {type(e).__name__} - {str(e)[:100]}"
        if prediction_id:
            try: supabase.table('predictions').update({"status": error_status_message }).eq('id', prediction_id).execute()
            except Exception as final_update_e: print(f"Failed to update DB on error: {final_update_e}")
        
        # Attempt to clean up report assets even if prediction_id was created
        if asset_prefix: 
            paths_to_clean = [
                f"{asset_prefix}/similarity_plot_ch{channel_index_for_plot + 1}.png",
                f"{asset_prefix}/timeseries.png",
                f"{asset_prefix}/psd.png",
                f"{asset_prefix}/technical_report.pdf",
                f"{asset_prefix}/patient_report.pdf"
            ]
            for p_clean in paths_to_clean: _cleanup_storage_on_error(REPORT_ASSET_BUCKET, p_clean)
        _cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path) 
        
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500
    finally: 
        # Cleanup temp file
        if 'absolute_temp_filepath' in locals() and os.path.exists(absolute_temp_filepath): # type: ignore
            try: os.remove(absolute_temp_filepath); print(f"Removed temp file: {absolute_temp_filepath}") # type: ignore
            except Exception as e_rem_temp: print(f"Error removing temp file {absolute_temp_filepath}: {e_rem_temp}") # type: ignore
        # Cleanup ML output JSON
        if 'ml_output_file_path' in locals() and os.path.exists(ml_output_file_path): # type: ignore
            try: os.remove(ml_output_file_path); print(f"Removed ML output file: {ml_output_file_path}") # type: ignore
            except Exception as e_rem_ml: print(f"Error removing ML output file {ml_output_file_path}: {e_rem_ml}")


# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.exists(ALZ_REF_PATH): print(f"WARNING: Alzheimer's reference file missing: {ALZ_REF_PATH}")
    if not os.path.exists(NORM_REF_PATH): print(f"WARNING: Normal reference file missing: {NORM_REF_PATH}")
    print("--- Starting Flask Server ---")
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"--- Debug Mode: {debug_mode} ---")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)