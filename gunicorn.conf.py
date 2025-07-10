"""
Production configuration for the Food Recognition API.
"""

import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 1  # Use 1 worker to avoid model loading issues
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 300
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = "food_recognition_api"

# Server mechanics
daemon = False
pidfile = "/tmp/food_recognition_api.pid"
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (configure if needed)
# keyfile = None
# certfile = None

# Application specific
preload_app = True
sendfile = False

# Worker tmp directory
worker_tmp_dir = "/dev/shm"

# Maximum allowed header size
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
