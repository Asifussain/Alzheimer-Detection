import pandas as pd # Added for date formatting
import traceback # Added to handle exception printing
from fpdf import XPos, YPos # Import for multi_cell positioning
from .base_report import BasePDFReport # Relative import
from utils import sanitize_for_helvetica # Relative import
from .technical_report import format_metric_for_pdf # Reusing formatter

class PatientPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Your AI EEG Pattern Report"
        # Patient-specific color overrides if any, or use BasePDFReport's
        self.primary_color = (74, 144, 226) # Example: Softer blue for patient report titles
        self.highlight_color_alz = (231, 76, 60) # More prominent red
        self.highlight_color_norm = (46, 204, 113) # Clear green

def build_patient_pdf_report_content(pdf: PatientPDFReport, prediction_data, 
                                     similarity_data, consistency_metrics, 
                                     similarity_plot_data):
    try:
        pdf.add_page()
        
        # --- Section 1: Analysis Summary ---
        pdf.section_title("Analysis Summary")
        created_at_str = 'N/A'
        if prediction_data.get('created_at'):
            try: created_at_str = pd.to_datetime(prediction_data['created_at']).strftime('%B %d, %Y')
            except: created_at_str = str(prediction_data['created_at'])
        pdf.key_value_pair("File Analyzed", prediction_data.get('filename', 'N/A'))
        pdf.key_value_pair("Date of Analysis", created_at_str)
        pdf.ln(6)

        pdf.add_explanation_box(
            "About This Report",
            [
                "This report uses Artificial Intelligence (AI) to look for specific patterns in your brainwave (EEG) activity.",
                "The AI compares your EEG patterns to those it has learned from many examples.",
                ("bullet", "Important: This is an informational tool to help your doctor. **It is not a medical diagnosis.** Please discuss these results with them.")
            ],
            icon_char="[i]", # Using text icon for simplicity
            font_size_text=9.5, line_h=5.5
        )

        # --- Section 2: AI's Main Finding ---
        pdf.section_title("AI's Main Finding: Pattern Assessment")
        prediction_label = prediction_data.get('prediction', 'Not Determined')
        pred_display_text = "Pattern assessment inconclusive"
        pred_color = pdf.text_color_dark # Default color

        if prediction_label == "Alzheimer's":
            pred_display_text = "Patterns Suggestive of Alzheimer's Characteristics"
            pred_color = pdf.highlight_color_alz
        elif prediction_label == "Normal":
            pred_display_text = "Normal Brainwave Patterns Observed"
            pred_color = pdf.highlight_color_norm
        
        pdf.write_paragraph("The AI analyzed your EEG and found that the patterns are most similar to:", font_size=10, height=5.5)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(*pred_color)
        pdf.multi_cell(0, 8, pred_display_text, border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*pdf.text_color_normal) # Reset color
        pdf.ln(3)

        probabilities = prediction_data.get('probabilities')
        confidence_text = "AI confidence score for this finding is not available."
        if isinstance(probabilities, list) and len(probabilities) == 2:
            try:
                conf_val_idx = 1 if prediction_label == "Alzheimer's" else 0
                conf_val = probabilities[conf_val_idx] * 100
                confidence_text = f"The AI is **{conf_val:.0f}%** confident that the patterns it found align with the finding above (based on the first segment of your EEG data)."
            except Exception as e:
                print(f"Error formatting confidence: {e}")
        
        pdf.add_explanation_box(
            "AI's Confidence Level", 
            [confidence_text], 
            icon_char="[T]", 
            font_size_text=9.5, line_h=5.5
        )

        # --- Section 3: AI's Internal Consistency Check ---
        pdf.section_title("AI's Internal Consistency Check")
        if consistency_metrics and not consistency_metrics.get('error') and isinstance(consistency_metrics.get('num_trials'), int) and consistency_metrics.get('num_trials', 0) > 0:
            num_segments = consistency_metrics.get('num_trials', 'multiple')
            intro_text = [f"To double-check its findings, the AI looked at your EEG data in **{num_segments} smaller pieces** (segments). This helps assess how stable the AI's finding was across your entire recording. Here's a simple breakdown:"]
            
            metric_items_for_patient = []
            accuracy_val = format_metric_for_pdf(consistency_metrics.get('accuracy'), 'percent', 0)
            metric_items_for_patient.append(("bullet", f"**Overall Consistency (Accuracy):** {accuracy_val}. This shows how often the AI's checks on the small pieces matched its main finding for your whole EEG sample."))

            if prediction_label == "Alzheimer's":
                sensitivity_val = format_metric_for_pdf(consistency_metrics.get('recall_sensitivity'), 'percent', 0)
                precision_val = format_metric_for_pdf(consistency_metrics.get('precision'), 'percent', 0)
                f1_val = format_metric_for_pdf(consistency_metrics.get('f1_score'), 'float', 2)
                metric_items_for_patient.extend([
                    ("bullet", f"**Finding Alzheimer's-like Patterns (Sensitivity):** {sensitivity_val}. If segments showed Alzheimer's-like patterns (based on the main finding), the AI found them this often."),
                    ("bullet", f"**Confirming Alzheimer's-like Patterns (Precision):** {precision_val}. When the AI said a segment was Alzheimer's-like, it was consistent with the main finding this often."),
                    ("bullet", f"**Balanced Score for Alzheimer's Patterns (F1-Score):** {f1_val}. A combined score (0 to 1, higher is better) reflecting how well the AI balanced finding and confirming these patterns.")
                ])
            else: # Normal
                specificity_val = format_metric_for_pdf(consistency_metrics.get('specificity'), 'percent', 0)
                metric_items_for_patient.append(("bullet", f"**Finding Normal Patterns (Specificity):** {specificity_val}. If Normal patterns were present in segments (based on the main finding), the AI correctly identified them this often."))
            
            metric_items_for_patient.extend([
                ("bullet", f"**Number of Segments Checked:** {num_segments}."),
                "Higher percentages and scores in these checks generally suggest the AI was consistent in what it observed throughout your EEG sample."
            ])
            
            pdf.add_explanation_box(
                "Understanding AI's Consistency", 
                intro_text + metric_items_for_patient, 
                icon_char="[M]", 
                bg_color=(230,250,230), title_color=pdf.highlight_color_norm, 
                font_size_text=9, line_h=5
            )
        elif consistency_metrics and consistency_metrics.get('message'):
             pdf.write_paragraph(f"(Consistency check: {consistency_metrics['message']})", font_style='I', indent=3, font_size=9)
        else:
            pdf.write_paragraph("(Detailed internal consistency checks were not applicable or did not yield specific metrics for this sample.)", font_style='I', indent=3, font_size=9)
        pdf.ln(5)

        # --- Section 4: Brainwave Shape Comparison ---
        if pdf.get_y() > pdf.h - 120 : pdf.add_page() # Check for page break before image
        
        if similarity_data and not similarity_data.get('error') and similarity_plot_data:
            plotted_ch_idx = similarity_data.get('plotted_channel_index')
            plot_title_sim = f"Comparing Your Brainwave Shape (from Channel {plotted_ch_idx + 1 if plotted_ch_idx is not None else 'Selected'})"
            pdf.add_image_section(plot_title_sim, similarity_plot_data) # Uses BasePDFReport method

            sim_interp_text_main = "The AI found that your sample's brainwave shapes showed "
            overall_sim = similarity_data.get('overall_similarity', '')
            if "Higher Similarity to Alzheimer's Pattern" in overall_sim:
                sim_interp_text_main += "**more resemblance to the Alzheimer's reference patterns**."
            elif "Higher Similarity to Normal Pattern" in overall_sim:
                sim_interp_text_main += "**more resemblance to the Normal reference patterns**."
            else:
                sim_interp_text_main += "a mixed or inconclusive resemblance when compared to the reference patterns."
            
            sim_interpretation_from_data = similarity_data.get('interpretation', "").split("Disclaimer:")[0].replace("Similarity Analysis (DTW):", "").replace("Overall Assessment:", "").strip()
            
            pdf.add_explanation_box(
                "What This Graph Shows",
                [
                    sim_interp_text_main,
                    f"Further Details: \"{sim_interpretation_from_data}\""
                ],
                icon_char="[D]", 
                font_size_text=9.5, line_h=5.5
            )
        else:
            pdf.section_title("Comparing Your Brainwave Shape") # Still add title if no plot
            pdf.write_paragraph("(The brainwave shape comparison graph is not available for this report.)", font_style='I')
        pdf.ln(6)

        # --- Section 5: Important Information & Next Steps ---
        pdf.section_title("Important Information & Your Next Steps")
        pdf.add_explanation_box(
            "Please Discuss This Report With Your Doctor",
            [
                "This AI report is an informational tool based on EEG patterns. **It is NOT a medical diagnosis.**",
                "Only a qualified healthcare professional can diagnose medical conditions. They will consider this report along with your full medical history and other tests.",
                ("bullet", f"Key Takeaway: The AI analysis suggests your EEG patterns are most similar to **{sanitize_for_helvetica(pred_display_text)}**."),
                ("bullet", "**Recommended Next Steps:**"),
                ("bullet", ("sub_bullet", "Share this entire report with your doctor or a neurologist.")),
                ("bullet", ("sub_bullet", "Discuss any health concerns and follow their medical advice."))
            ],
            icon_char="[!]", 
            bg_color=pdf.warning_bg_color, title_color=(106, 63, 20), text_color_override=(85,60,10), 
            font_size_text=9.5, line_h=5.5
        )
    except Exception as e:
        print(f"Error building Patient PDF content: {e}")
        traceback.print_exc()
        try:
            if pdf.page_no() == 0: pdf.add_page()
            elif pdf.get_y() > pdf.h - 30 : pdf.add_page()
            pdf.set_font("Helvetica",'B',12); pdf.set_text_color(255,0,0)
            pdf.multi_cell(0,10,f"Critical Error Building PDF Content:\n{sanitize_for_helvetica(str(e))}",align='C')
            pdf.set_text_color(*pdf.text_color_normal)
        except Exception as pdf_err_fallback:
            print(f"Fallback error writing to Patient PDF failed: {pdf_err_fallback}")