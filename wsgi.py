# wsgi.py

from app import app
import os

server = app.server  # 暴露 WSGI callable
