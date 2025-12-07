# src/utils/__init__.py
"""
Utils package for MLOps Spam Detection
"""

from .config_loader import ConfigLoader, get_config, get_path, get_model_config

__all__ = [
    'ConfigLoader',
    'get_config',
    'get_path', 
    'get_model_config'
]

__version__ = '1.0.0' 
