# src/api/routers/__init__.py
"""
Routers for Spam Detection API
"""

from .predict import router as predict_router
from .health import router as health_router
from .admin import router as admin_router

__all__ = ["predict_router", "health_router", "admin_router"] 
