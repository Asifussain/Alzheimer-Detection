# backend/celery_utils.py
import os
from celery import Celery
from config import REDIS_URL # MODIFICATION: Import REDIS_URL from your config

# MODIFICATION: Use REDIS_URL for backend and broker
celery_app = Celery(
    __name__,
    backend=REDIS_URL,
    broker=REDIS_URL,
    imports=('routes.predict_api',)
)

print("--- Celery instance created and tasks imported ---")