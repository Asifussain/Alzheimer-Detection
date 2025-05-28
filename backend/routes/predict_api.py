import os
import uuid
import json
import traceback
from datetime import datetime, timezone
import base64 # For handling image data

from flask import request, jsonify, Blueprint
from ..supabase_client_setup import get_supabase_client
from ..config import (
    UPLOAD_FOLDER, OUTPUT_JSON_PATH, RAW_EEG_BUCKET, REPORT_ASSET_BUCKET,
    DEFAULT_FS, ALZ_REF_PATH, NORM_REF_PATH, BACKEND_DIR
)
from ..utils import NpEncoder
from ..database import get_prediction_and_eeg, cleanup_storage_on_error
from ..ml_runner import run_model

# Import visualization and similarity analysis (assuming they are in the backend root)
# Adjust paths if they are located elsewhere.
import sys
# Add backend directory to sys.path to allow direct import of visualization and similarity_analyzer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from visualization import (
        generate_stacked_timeseries_image,
        generate_average_psd_image,
        generate_descriptive_stats
    )
    from similarity_analyzer import run_similarity_analysis
except ImportError as e:
    print(f"Error importing visualization/similarity modules in predict_api.py: {e}")
    # Define dummy functions if import fails to prevent app crash during init
    def generate_stacked_timeseries_image(*args, **kwargs): return None
    def generate_average_psd_image(*args, **kwargs): return None
    def generate_descriptive_stats(*args, **kwargs): return {"error": "Visualization module not loaded"}
    def run_similarity_analysis(*args, **kwargs):
        return {"error": "Similarity analyzer module not loaded", 'interpretation': 'N/A', 'plot_base64': None}


# Import PDF generation modules
from ..pdf_generation import (
    TechnicalPDFReport, build_technical_pdf_report_content,
    PatientPDFReport, build_patient_pdf_report_content,
    ClinicianPDFReport, build_clinician_pdf_report_content
)

# Use the Blueprint from routes/__init__.py
from . import api_bp


@api_bp.route('/predict', methods=['POST'])
def predict_route():
    supabase = get_supabase_client()
    
    file = request.files.get('file')
    user_id = request.form.get('user_id')
    
    try:
        channel_index_str = request.form.get('channel_index', '0')
        channel_index_for_plot = int(channel_index_str)
        if not (0 <= channel_index_for_plot <= 18): # Assuming 19 channels, 0-indexed
            raise ValueError("Channel index out of range 0-18")
    except (ValueError, TypeError):
        channel_index_for_plot = 0 # Default to first channel (index 0)
        print(f"Warning: Invalid or missing channel index '{channel_index_str}'. Defaulting to 0.")

    if not file or not user_id:
        return jsonify({'error': "Missing 'file' or 'user_id'"}), 400
    if not file.filename or not file.filename.lower().endswith('.npy'):
        return jsonify({'error': 'Invalid/Missing filename or type (.npy required).'}), 400

    filename_base, file_extension = os.path.splitext(file.filename)
    unique_id = str(uuid.uuid4()) # For unique filenames
    # Save to a temporary location within UPLOAD_FOLDER
    save_filename = f"{filename_base}_{unique_id}{file_extension}"
    absolute_temp_filepath = os.path.join(BACKEND_DIR, UPLOAD_FOLDER, save_filename)
    
    raw_eeg_storage_path = f'raw_eeg/{user_id}/{save_filename}'
    prediction_id = None
    report_generation_errors = []
    similarity_analysis_results = None
    consistency_metrics_results = None # From ML output
    
    ts_img_data, psd_img_data, similarity_plot_base64_data = None, None, None
    ts_url, psd_url, similarity_plot_url = None, None, None
    technical_pdf_url, patient_pdf_url, clinician_pdf_url = None, None, None
    
    asset_prefix = "" # Will be set after prediction_id is known

    try:
        print(f"API: Step 1/2: Processing uploaded file '{file.filename}'...")
        os.makedirs(os.path.dirname(absolute_temp_filepath), exist_ok=True)
        file.save(absolute_temp_filepath)
        print(f"API: File saved temporarily to: {absolute_temp_filepath}")

        # Upload raw EEG to Supabase Storage
        with open(absolute_temp_filepath, 'rb') as f_upload:
            supabase.storage.from_(RAW_EEG_BUCKET).upload(
                path=raw_eeg_storage_path, 
                file=f_upload, 
                file_options={"content-type": "application/octet-stream", "upsert": "false"}
            )
        print(f"API: Raw EEG uploaded to Supabase: {raw_eeg_storage_path}")

        # Run ML model
        print(f"API: Step 3: Running ML model on {absolute_temp_filepath}...")
        # run_model now returns the path to output.json
        ml_output_file_path = run_model(absolute_temp_filepath)
        
        if not os.path.exists(ml_output_file_path):
            raise FileNotFoundError(f"API Error: ML output file missing at {ml_output_file_path}")
        
        with open(ml_output_file_path, 'r') as f:
            ml_output_data = json.load(f)
        
        prediction_label = "Alzheimer's" if ml_output_data.get('majority_prediction') == 1 else "Normal"
        probabilities = ml_output_data.get('probabilities') # List of probabilities
        consistency_metrics_results = ml_output_data.get('consistency_metrics') # Dict
        trial_predictions = ml_output_data.get('trial_predictions') # List

        # Insert initial prediction record
        insert_data = {
            "user_id": user_id,
            "filename": file.filename,
            "prediction": prediction_label,
            "eeg_data_url": raw_eeg_storage_path,
            "probabilities": probabilities,
            "status": "Processing Report Assets",
            "trial_predictions": trial_predictions, # Store individual trial predictions
            "consistency_metrics": consistency_metrics_results # Store consistency metrics
        }
        print(f"API: Step 4: Inserting prediction record into database...")
        insert_payload_str = json.dumps(insert_data, cls=NpEncoder, allow_nan=False)
        insert_payload = json.loads(insert_payload_str)
        
        insert_res = supabase.table('predictions').insert(insert_payload).execute()

        if insert_res.data and len(insert_res.data) > 0:
            prediction_id = insert_res.data[0].get('id')
            print(f"API: DB Insert successful. Prediction ID: {prediction_id}")
        else:
            cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path) # Cleanup raw EEG if DB fails
            raise Exception(f"API Error: DB insert for prediction failed: {getattr(insert_res, 'error', insert_res)}")
        
        asset_prefix = f"report_assets/{prediction_id}" # Define asset path prefix

        # --- Generate and Upload Report Assets ---
        print(f"API: --- Step 5: Generating Report Assets for ID: {prediction_id} ---")
        # Fetch the record and EEG data again (or pass eeg_data if loaded and standardized from temp file)
        prediction_data_for_report, eeg_data, error_msg_get_eeg = get_prediction_and_eeg(prediction_id)
        if error_msg_get_eeg or eeg_data is None:
            raise Exception(f"API Error: Cannot load EEG data for report generation: {error_msg_get_eeg or 'No EEG data found'}")

        # Run similarity analysis
        print(f"API: Running similarity analysis for channel index {channel_index_for_plot}...")
        similarity_analysis_results = run_similarity_analysis(
            absolute_temp_filepath, # Use the saved temp file for analysis
            ALZ_REF_PATH, 
            NORM_REF_PATH, 
            channel_index_for_plot
        )
        similarity_plot_base64_data = similarity_analysis_results.get('plot_base64') if isinstance(similarity_analysis_results, dict) else None
        
        # Generate visualizations
        print("API: Generating statistical and plot assets...")
        stats_json = generate_descriptive_stats(eeg_data, DEFAULT_FS)
        ts_img_data = generate_stacked_timeseries_image(eeg_data, DEFAULT_FS)
        psd_img_data = generate_average_psd_image(eeg_data, DEFAULT_FS)
        
        # Upload assets
        asset_upload_details = [
            (similarity_plot_base64_data, f"{asset_prefix}/similarity_plot_ch{channel_index_for_plot + 1}.png", "SimPlotURL"),
            (ts_img_data, f"{asset_prefix}/timeseries.png", "TSPlotURL"),
            (psd_img_data, f"{asset_prefix}/psd.png", "PSDPlotURL")
        ]
        uploaded_asset_urls = {}

        for img_data, filename_s3, url_key_suffix in asset_upload_details:
            if img_data:
                try:
                    img_bytes = base64.b64decode(img_data.split(',',1)[1])
                    supabase.storage.from_(REPORT_ASSET_BUCKET).upload(
                        path=filename_s3, file=img_bytes, 
                        file_options={"content-type": "image/png", "upsert": "true"}
                    )
                    public_url_res = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(filename_s3)
                    if isinstance(public_url_res, str) and public_url_res.startswith('http'):
                        uploaded_asset_urls[url_key_suffix.lower().replace("url","")] = public_url_res
                    else:
                        report_generation_errors.append(url_key_suffix)
                        print(f"API Warning: Could not get public URL for {filename_s3}. Response: {public_url_res}")
                except Exception as e:
                    print(f"API Error: Uploading asset {filename_s3} failed: {e}")
                    report_generation_errors.append(url_key_suffix + "Upload")
        
        similarity_plot_url = uploaded_asset_urls.get("simplot")
        ts_url = uploaded_asset_urls.get("tsplot")
        psd_url = uploaded_asset_urls.get("psdplot")

        # --- Generate and Upload PDFs ---
        pdf_types = [
            ("technical", TechnicalPDFReport, build_technical_pdf_report_content, "TechPDF"),
            ("patient", PatientPDFReport, build_patient_pdf_report_content, "PatientPDF"),
            ("clinician", ClinicianPDFReport, build_clinician_pdf_report_content, "ClinicianPDF")
        ]
        pdf_urls = {}

        for pdf_type_name, PdfClass, builder_func, url_key_suffix in pdf_types:
            print(f"API: Generating {pdf_type_name.capitalize()} PDF report...")
            pdf_doc = PdfClass()
            pdf_doc.alias_nb_pages()
            builder_func_args = [pdf_doc, prediction_data_for_report, stats_json, 
                                 similarity_analysis_results, consistency_metrics_results, 
                                 ts_img_data, psd_img_data, similarity_plot_base64_data]
            if pdf_type_name == "patient": # Patient PDF has a simpler builder
                builder_func_args = [pdf_doc, prediction_data_for_report, 
                                     similarity_analysis_results, consistency_metrics_results, 
                                     similarity_plot_base64_data]
            
            builder_func(*builder_func_args)
            
            pdf_bytes = bytes(pdf_doc.output())
            pdf_filename_s3 = f"{asset_prefix}/{pdf_type_name}_report.pdf"
            try:
                supabase.storage.from_(REPORT_ASSET_BUCKET).upload(
                    path=pdf_filename_s3, file=pdf_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
                public_url_res_pdf = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(pdf_filename_s3)
                if isinstance(public_url_res_pdf, str) and public_url_res_pdf.startswith('http'):
                    pdf_urls[f"{url_key_suffix.lower()}_url"] = public_url_res_pdf
                else:
                    report_generation_errors.append(url_key_suffix + "URL")
                    print(f"API Warning: Could not get public URL for {pdf_filename_s3}. Response: {public_url_res_pdf}")
            except Exception as e_pdf:
                print(f"API Error: Uploading {pdf_type_name} PDF failed: {e_pdf}")
                report_generation_errors.append(url_key_suffix + "Upload")

        technical_pdf_url = pdf_urls.get("techpdf_url")
        patient_pdf_url = pdf_urls.get("patientpdf_url")
        clinician_pdf_url = pdf_urls.get("clinicianpdf_url")
        
        # --- Final DB Update ---
        report_generation_status = "Completed"
        if report_generation_errors:
            report_generation_status = f"Completed with errors ({', '.join(report_generation_errors)})"
            print(f"API: Report generation completed with errors: {report_generation_errors}")

        db_similarity_data_to_store = None
        if isinstance(similarity_analysis_results, dict) and not similarity_analysis_results.get('error'):
            db_similarity_data_to_store = {k: v for k, v in similarity_analysis_results.items() if k != 'plot_base64'}
        elif isinstance(similarity_analysis_results, dict): # Store error if present
            db_similarity_data_to_store = {"error": similarity_analysis_results.get('error', 'Unknown similarity error')}
        
        update_data = {
            "stats_data": stats_json,
            "timeseries_plot_url": ts_url,
            "psd_plot_url": psd_url,
            "technical_pdf_url": technical_pdf_url,
            "patient_pdf_url": patient_pdf_url,
            "clinician_pdf_url": clinician_pdf_url, # Save clinician PDF URL
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "status": report_generation_status,
            "similarity_results": db_similarity_data_to_store,
            "similarity_plot_url": similarity_plot_url,
            # Consistency metrics are already in the initial insert from ml_output_data
        }
        
        print(f"API: Step 6: Updating DB for {prediction_id} with status '{report_generation_status}'...")
        update_payload_final_str = json.dumps(update_data, cls=NpEncoder, allow_nan=False)
        update_payload_final = json.loads(update_payload_final_str)
        
        supabase.table('predictions').update(update_payload_final).eq('id', prediction_id).execute()
        
        return jsonify({
            "filename": file.filename, 
            "prediction": prediction_label, 
            "prediction_id": prediction_id
        })

    except Exception as e:
        print(f"API ERROR in /api/predict: {e}")
        traceback.print_exc()
        if prediction_id: # If prediction record was created, update its status to Failed
            try:
                fail_status = f"Failed: {type(e).__name__} - {str(e)[:100]}"
                supabase.table('predictions').update({"status": fail_status}).eq('id', prediction_id).execute()
            except Exception as final_update_e:
                print(f"API: Failed to update DB on error: {final_update_e}")
        
        # Cleanup storage if error occurred after prediction_id was set
        if asset_prefix: # Implies prediction_id was set
            assets_to_clean = [
                f"{asset_prefix}/similarity_plot_ch{channel_index_for_plot + 1}.png",
                f"{asset_prefix}/timeseries.png", f"{asset_prefix}/psd.png",
                f"{asset_prefix}/technical_report.pdf", f"{asset_prefix}/patient_report.pdf",
                f"{asset_prefix}/clinician_report.pdf" # Added clinician PDF
            ]
            for asset_path in assets_to_clean:
                cleanup_storage_on_error(REPORT_ASSET_BUCKET, asset_path)
        
        cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path) # Attempt to cleanup raw EEG

        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500
    finally:
        # Clean up the temporary saved file from UPLOAD_FOLDER
        if 'absolute_temp_filepath' in locals() and os.path.exists(absolute_temp_filepath):
            try:
                os.remove(absolute_temp_filepath)
                print(f"API: Removed temp file: {absolute_temp_filepath}")
            except Exception as e_rem_temp:
                print(f"API Error: Error removing temp file {absolute_temp_filepath}: {e_rem_temp}")
        
        # Clean up ML output JSON from SIDDHI folder (now handled by run_model or this main route)
        # The run_model function returns the path, so we can use that.
        if 'ml_output_file_path' in locals() and os.path.exists(ml_output_file_path):
             try:
                 os.remove(ml_output_file_path)
                 print(f"API: Removed ML output file: {ml_output_file_path}")
             except Exception as e_rem_ml:
                 print(f"API Error: Error removing ML output file {ml_output_file_path}: {e_rem_ml}")