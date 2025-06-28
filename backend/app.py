# backend/app.py
import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import the blueprint from your routes
from routes import api_bp
# Import the celery_app from our new utility file
from celery_utils import celery_app

# Load .env file for local development
load_dotenv()

# --- Main Application Factory ---
def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)

    # --- CORS Configuration ---
    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
    origins = [frontend_url]
    if "localhost" not in frontend_url:
        origins.append("http://localhost:3000")
    print(f"--- CORS is configured to allow origins: {origins} ---")
    CORS(app, resources={r"/api/*": {"origins": origins}})

    # --- Register Routes ---
    app.register_blueprint(api_bp, url_prefix='/api')

    # --- Configure Celery ---
    # Update Celery config with the Flask app context
    celery_app.conf.update(app.config)
    class ContextTask(celery_app.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery_app.Task = ContextTask
    print("--- Celery App configured with Flask context ---")

    # A simple root route to check if the server is up
    @app.route('/')
    def index():
        return jsonify({"message": "Backend is alive and running!"})

    return app

# Create the app instance
app = create_app()

# This part is for running the app directly with `python app.py`
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
