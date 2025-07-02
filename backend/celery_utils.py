import os
from celery import Celery
from config import REDIS_URL

# Check if the Redis URL is for a secure connection (rediss://)
is_secure_redis = REDIS_URL.startswith("rediss://")

# Define SSL options only if it's a secure connection
ssl_options = {'ssl_cert_reqs': 'none'} if is_secure_redis else {}

celery_app = Celery(
    __name__,
    broker=REDIS_URL,
    backend=REDIS_URL,
    broker_use_ssl=ssl_options,
    redis_backend_use_ssl=ssl_options,
    # --- THIS LINE IS THE FIX ---
    imports=('routes.predict_api',)
)

print("--- Celery instance created and tasks imported ---")