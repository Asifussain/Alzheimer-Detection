from fpdf import FPDF, XPos, YPos
from utils import sanitize_for_helvetica # Relative import

class BasePDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "EEG Analysis Report" # Default, can be overridden
        # Color definitions (can be moved to config or kept here if specific to PDF styling)
        self.primary_color = (52, 73, 94) 
        self.secondary_color = (74, 144, 226)
        self.text_color_dark = (30, 30, 30)
        self.text_color_light = (100, 100, 100)
        self.text_color_normal = (0,0,0) # Black for normal text
        self.line_color = (220, 220, 220)
        self.card_bg_color = (248, 249, 250)
        self.highlight_color_alz = (220, 60, 60) 
        self.highlight_color_norm = (60, 179, 113)
        self.warning_bg_color = (255, 243, 205)
        self.warning_text_color = (133, 100, 4)
        
        self.page_margin = 15
        self.set_auto_page_break(auto=True, margin=self.page_margin)
        self.set_line_width(0.2) # Default line width

    def _is_bold_font(self):
        return 'B' in self.font_style

    def cell(self, w, h=0, txt="", border=0, ln=0, align="", fill=False, link=""):
        txt_to_render = sanitize_for_helvetica(txt)
        super().cell(w, h, txt_to_render, border, ln, align, fill, link)

    def multi_cell(self, w, h, txt="", border=0, align="J", fill=False, max_line_height=0, new_x=XPos.START, new_y=YPos.TOP):
        txt_to_render = sanitize_for_helvetica(txt)
        if max_line_height == 0: max_line_height = h # Ensure max_line_height is reasonable
        super().multi_cell(w, h, txt_to_render, border, align, fill, max_line_height=max_line_height, new_x=new_x, new_y=new_y)

    def write(self, h, txt="", link=""):
        txt_to_render = sanitize_for_helvetica(txt)
        super().write(h, txt_to_render, link)

    def header(self):
        try:
            self.set_font('Helvetica', 'B', 15)
            title = sanitize_for_helvetica(self.report_title)
            title_w = self.get_string_width(title) + 6
            doc_w = self.w
            self.set_x((doc_w - title_w) / 2)
            self.set_text_color(*self.secondary_color) # Primary Blue for header text
            self.cell(title_w, 10, title, border=0, align='C', ln=1)
            self.set_text_color(*self.text_color_normal) # Reset color
            self.ln(5)
            self.set_draw_color(*self.line_color)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(8)
        except Exception as e:
            print(f"PDF Header Error: {e}")

    def footer(self):
        try:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128) # Gray
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
            self.set_text_color(*self.text_color_normal) # Reset color
        except Exception as e:
            print(f"PDF Footer Error: {e}")

    def section_title(self, title_text: str):
        try:
            self.set_font('Helvetica', 'B', 13)
            self.set_fill_color(80, 227, 194)  # Accent Teal
            self.set_text_color(10, 15, 26)    # Dark text for contrast
            self.cell(0, 8, " " + sanitize_for_helvetica(title_text), border='B', align='L', fill=True, ln=1)
            self.set_text_color(*self.text_color_normal) # Reset text color
            self.ln(6)
        except Exception as e:
            print(f"PDF Section Title Error for '{title_text}': {e}")

    def key_value_pair(self, key: str, value, key_width=45):
        try:
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(*self.text_color_dark)
            
            key_start_y = self.get_y()
            # Use multi_cell for key to handle potential wrapping, though less likely for keys
            self.multi_cell(key_width, 6, sanitize_for_helvetica(str(key))+":", align='L', new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=self.font_size)
            
            self.set_y(key_start_y) # Reset Y to align value with key
            self.set_x(self.l_margin + key_width + 2) # Position for value
            
            self.set_font('Helvetica', '', 10)
            self.set_text_color(*self.text_color_normal)
            self.multi_cell(0, 6, sanitize_for_helvetica(str(value)), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size)
            self.ln(1) # Minimal gap
        except Exception as e:
            print(f"PDF Key/Value Error for key '{key}': {e}")

    def write_multiline(self, text: str, height=5, indent=5):
        try:
            self.set_font('Helvetica', '', 10)
            self.set_text_color(80, 80, 80) # Slightly lighter text
            self.set_left_margin(self.l_margin + indent)
            self.multi_cell(0, height, sanitize_for_helvetica(text), align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size)
            self.set_left_margin(self.l_margin) # Reset margin
            self.ln(height / 2)
            self.set_text_color(*self.text_color_normal)
        except Exception as e:
            print(f"PDF Multiline Error: {e}")

    def metric_card(self, title: str, value, unit: str = "", description: str = ""):
        # This method was complex and might need more adjustment if used heavily.
        # For simplicity, this is a direct port. Consider if it's still needed or can be simplified.
        try:
            start_x = self.get_x()
            start_y = self.get_y()
            card_width = (self.w - self.l_margin - self.r_margin - 5) / 2 # For two cards side-by-side
            card_height = 25 # Fixed height

            self.set_fill_color(240, 245, 250) # Light grey background
            self.set_draw_color(80, 227, 194)  # Accent teal border
            self.set_line_width(0.3)
            self.rect(start_x, start_y, card_width, card_height, 'DF') # Draw and fill

            # Title
            self.set_xy(start_x + 3, start_y + 3)
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(80, 80, 80)
            self.cell(card_width - 6, 5, sanitize_for_helvetica(title.upper()), align='L')

            # Value
            self.set_xy(start_x + 3, start_y + 9)
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(*self.secondary_color) # Blue
            value_str = f"{sanitize_for_helvetica(str(value))}{sanitize_for_helvetica(str(unit))}"
            self.cell(card_width - 6, 8, value_str, align='R')

            # Description
            if description:
                self.set_xy(start_x + 3, start_y + 18)
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(*self.text_color_light)
                self.cell(card_width - 6, 5, sanitize_for_helvetica(description), align='L')

            # Reset position for next element (if placing cards side-by-side manually)
            self.set_y(start_y)
            self.set_x(start_x + card_width + 5) 
            self.set_text_color(*self.text_color_normal)
            self.set_line_width(0.2) # Reset line width
        except Exception as e:
            print(f"PDF Metric Card Error for title '{title}': {e}")

    def write_paragraph(self, text, height=4, indent=0, font_style='', font_size=8.5, text_color=None, bullet_char_override=None):
        try:
             self.set_font('Helvetica', font_style, font_size)
             current_text_color = text_color if text_color else self.text_color_dark
             self.set_text_color(*current_text_color)
             
             current_x_start = self.l_margin + indent
             self.set_x(current_x_start)
             
             sanitized_text = sanitize_for_helvetica(text)
             
             if bullet_char_override:
                 safe_bullet = sanitize_for_helvetica(bullet_char_override)
                 # Store original font to restore after printing bullet
                 original_font_family, original_font_size, original_font_style = self.font_family, self.font_size_pt, self.font_style
                 
                 self.set_font('Helvetica', 'B', font_size) # Bullet bold
                 self.cell(self.get_string_width(safe_bullet) + 0.5, height, safe_bullet, ln=0) # ln=0 to continue on same line
                 
                 # Restore original font for the text part
                 self.set_font(original_font_family, original_font_style, original_font_size)
                 self.set_x(current_x_start + self.get_string_width(safe_bullet) + 1.5) # Position after bullet
                 
                 # Multi_cell for the actual text content
                 self.multi_cell(self.w - self.get_x() - self.r_margin, height, sanitized_text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size)
             else:
                 self.multi_cell(0, height, sanitized_text, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size)
                 
             self.ln(height / 4) # Small space after paragraph
             self.set_text_color(*self.text_color_normal) # Reset to default
        except Exception as e:
            print(f"PDF write_paragraph Error: {e}")

    def add_image_section(self, title: str, image_data_base64: str):
        import base64 # Local import
        import io     # Local import

        image_height_estimate = 70  # mm, rough estimate for pre-page break check
        title_height_estimate = 10 if title else 0

        # Check if we need a new page
        if self.get_y() + title_height_estimate + image_height_estimate > self.h - self.b_margin:
            self.add_page()

        if title:
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(*self.text_color_dark)
            self.cell(0, 8, sanitize_for_helvetica(title), ln=1, align='L')
            self.ln(1)
        
        if image_data_base64 and isinstance(image_data_base64, str) and image_data_base64.startswith('data:image/png;base64,'):
            try:
                img_bytes = base64.b64decode(image_data_base64.split(',', 1)[1])
                img_file = io.BytesIO(img_bytes)
                
                page_content_width = self.w - 2 * self.page_margin
                img_display_width = page_content_width * 0.95 # Use 95% of content width
                
                # Center the image
                x_pos = self.l_margin + (page_content_width - img_display_width) / 2
                
                self.image(img_file, x=x_pos, w=img_display_width) # Height will be auto-calculated
                img_file.close()
                self.ln(2) # Space after image
            except Exception as e:
                error_text = f"(Error embedding image '{sanitize_for_helvetica(title)}': {sanitize_for_helvetica(str(e)[:50])})"
                self.write_paragraph(error_text, font_style='I')
                print(f"PDF Image Embed Error for '{title}': {e}")
        else:
            if title: # Only print if title was provided, to avoid lonely "(Image data not available)"
                 self.write_paragraph(sanitize_for_helvetica("(Image data not available)"), font_style='I', indent=5)
        self.ln(4) # Overall space after section

    def add_explanation_box(self, title: str, text_lines: list, icon_char: str = "[i]", 
                            bg_color=None, title_color=None, text_color_override=None, 
                            font_size_text=9, line_h=5):
        self.ln(1) # Space before box
        current_bg_color = bg_color if bg_color else self.card_bg_color
        current_title_color = title_color if title_color else self.primary_color
        current_text_color = text_color_override if text_color_override else self.text_color_dark
        
        safe_icon = sanitize_for_helvetica(icon_char)
        
        # Title part
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*current_title_color)
        title_to_render = f"{safe_icon} {sanitize_for_helvetica(title)}" if safe_icon else sanitize_for_helvetica(title)
        self.multi_cell(0, 7, title_to_render, new_x=XPos.LMARGIN, new_y=YPos.NEXT, max_line_height=self.font_size) # Title on its own line
        
        y_before_text_content = self.get_y()
        
        # Estimate height for the box background
        estimated_box_height = 3 # Padding top/bottom
        for item in text_lines:
            estimated_box_height += line_h + 1 # Line height + small gap

        # Draw background rectangle
        self.set_fill_color(*current_bg_color)
        self.set_draw_color(*self.line_color) # Border color
        self.rect(self.l_margin, y_before_text_content, self.w - self.l_margin - self.r_margin, estimated_box_height, 'DF')
        
        self.set_y(y_before_text_content + 1.5) # Start text slightly inside the box

        for item in text_lines:
            self.set_x(self.l_margin + 2) # Indent text inside box
            
            is_list_item = isinstance(item, tuple) and item[0] == "bullet"
            actual_text = item[1] if is_list_item else item
            
            is_sub_list_item = isinstance(actual_text, tuple) and actual_text[0] == "sub_bullet"
            final_text_content = actual_text[1] if is_sub_list_item else actual_text

            bullet_char_to_use = "-" # Default for sub-bullet or if not specified
            
            if is_list_item:
                item_x_start = self.get_x()
                self.set_font('Helvetica', 'B', font_size_text) # Bullet font
                self.set_text_color(*current_text_color) # Bullet color

                if is_sub_list_item:
                    self.set_x(item_x_start + 5) # Indent sub-bullet
                
                self.cell(5, line_h, bullet_char_to_use, ln=0)
                self.set_x(item_x_start + 5 + (5 if is_sub_list_item else 0)) # Position after bullet
            
            # Text content with bold handling
            # Assuming bold is indicated by **text**
            parts = sanitize_for_helvetica(final_text_content).split("**")
            for i, part in enumerate(parts):
                is_bold_part = (i % 2 == 1)
                self.set_font('Helvetica', 'B' if is_bold_part else '', font_size_text)
                self.set_text_color(*(self.primary_color if is_bold_part else current_text_color))
                self.write(line_h, part) # write handles continuation on the line
            
            self.ln(line_h + 0.5) # Move to next line for next item
            
        self.set_y(y_before_text_content + estimated_box_height) # Position after the box
        self.ln(3) # Space after box
        self.set_text_color(*self.text_color_normal) # Reset default text color