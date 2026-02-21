# Gunicorn config: 300s worker timeout for file uploads (avoids WORKER TIMEOUT during multipart parse)
# Loaded via: gunicorn -c gunicorn.conf.py app:app
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 2
timeout = 300
