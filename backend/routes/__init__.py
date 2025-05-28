from flask import Blueprint

# Create a Blueprint for API routes
# This can be expanded later if you have more route files
api_bp = Blueprint('api_bp', __name__, url_prefix='/api')

# Import routes from this package to register them with the blueprint
from . import predict_api # noqa E402 F401 : ignore unused import and import not at top warnings

def register_blueprints(app):
    app.register_blueprint(api_bp)