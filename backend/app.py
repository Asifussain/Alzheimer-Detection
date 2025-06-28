# backend/app.py
import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from routes import api_bp # Your routes blueprint

load_dotenv()

app = Flask(__name__)
# 1. Get the frontend URL from environment variables.
#    Default to a common local frontend port if the variable isn't set.
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')

# 2. Create the list of allowed origins.
origins = [frontend_url]

# 3. For convenience, if your deployed URL is not localhost,
#    add the localhost URL to the list anyway to make local testing easier.
if "localhost" not in frontend_url:
    origins.append("http://localhost:3000")

# 4. **CRUCIAL DEBUGGING STEP**: Print the origins list to the logs.
#    This will show up in your Render logs.
print(f"--- CORS is configured to allow origins: {origins} ---")

# 5. Initialize the CORS extension.
CORS(app, resources={r"/api/*": {"origins": origins}})

# --- End CORS Configuration ---

# Register your API routes under the /api prefix
app.register_blueprint(api_bp, url_prefix='/api')

# A simple route for the root URL to confirm the server is up
@app.route('/')
def index():
    return jsonify({"message": "Backend is alive and running!"})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)