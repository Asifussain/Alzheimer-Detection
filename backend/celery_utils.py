# backend/celery_utils.py
import os
from celery import Celery

# This line gets the Redis URL from your environment variables.
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# This creates the Celery application instance.
# The 'imports' line is the new, crucial part.
# It tells Celery to look inside 'routes.predict_api' to find any defined tasks.
celery_app = Celery(
    __name__,
    backend=redis_url,
    broker=redis_url,
    imports=('routes.predict_api',) # <-- THIS IS THE FIX
)

print("--- Celery instance created and tasks imported ---")
