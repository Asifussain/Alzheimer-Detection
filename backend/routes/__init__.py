from flask import Blueprint

# Define the blueprint that other files in this 'routes' package will use
api_bp = Blueprint('api_bp', __name__)

# Import your route files here.
# This line is crucial because it runs the code in predict_api.py,
# which in turn registers its routes with the 'api_bp' blueprint.
from . import predict_api