# backend/app.py
from flask import Flask
from flask_cors import CORS
from routes import api_bp # Your routes blueprint
import os # Import os to access environment variables

app = Flask(__name__)

# --- Correct CORS Configuration ---
# 1. Get the frontend URL from the environment variable set in Render.
# 2. Provide a default fallback value for when you run it locally.
live_frontend_url = os.environ.get('FRONTEND_URL')
local_fallback_url = "http://localhost:3000"

# Use the live URL if it exists, otherwise use the local fallback.
# This makes your app work correctly in BOTH deployment and local testing.
origins = [live_frontend_url] if live_frontend_url else [local_fallback_url]

# Add the Supabase callback URL to the origins list for Google Auth redirects
# This can sometimes help with post-login issues, though it's not strictly for API calls
# supabase_callback = f"https://{os.environ.get('SUPABASE_PROJECT_REF')}.supabase.co"
# if "SUPABASE_PROJECT_REF" in os.environ:
#    origins.append(supabase_callback)

print(f"--- CORS is configured to allow origins: {origins} ---")
CORS(app, resources={r"/api/*": {"origins": origins}})


# Register Blueprints
app.register_blueprint(api_bp, url_prefix='/api')

@app.route('/')
def index():
    return "Welcome to the Alzheimer Detection API! The server is running."

if __name__ == '__main__':
    print("--- Starting Flask Server ---")
    # Host 0.0.0.0 makes it accessible externally (required by Render)
    app.run(host='0.0.0.0', port=5000)