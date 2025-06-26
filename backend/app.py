from flask import Flask
from flask_cors import CORS
from config import FRONTEND_URL, BACKEND_DIR # Ensure BACKEND_DIR is used if SIDDHI path is relative
from routes import api_bp
import os # For path checks

# --- Check for critical external files/folders needed by the backend ---
# This is mostly for robust startup diagnostics
alz_ref_path_check = os.path.join(BACKEND_DIR, 'feature_07.npy')
norm_ref_path_check = os.path.join(BACKEND_DIR, 'feature_35.npy')
siddhi_dir_check = os.path.join(BACKEND_DIR, 'SIDDHI')

if not os.path.exists(alz_ref_path_check):
    print(f"STARTUP WARNING: Alzheimer's reference file missing: {alz_ref_path_check}")
if not os.path.exists(norm_ref_path_check):
    print(f"STARTUP WARNING: Normal reference file missing: {norm_ref_path_check}")
if not os.path.isdir(siddhi_dir_check):
    print(f"STARTUP CRITICAL ERROR: SIDDHI directory missing: {siddhi_dir_check}. ML model will not function.")
# --- End Startup Checks ---

app = Flask(__name__)

# Apply CORS settings from config
CORS(app, resources={r"/api/*": {"origins": FRONTEND_URL}})

# Initialize Supabase client (this happens when supabase_client_setup is imported by other modules)
# from . import supabase_client_setup # Ensures client is initialized early if needed by other setup
# No, this should be fine as modules import it.

# Register Blueprints
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    print("--- Starting Flask Server (Refactored) ---")
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"--- Debug Mode: {debug_mode} ---")
    # Host 0.0.0.0 makes it accessible externally if needed (e.g., in Docker)
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)