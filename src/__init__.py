"""
Steatosis Classification package initialization.
"""

# Expose main components
from .model import SteatosisModel, get_loss_fn
from .data import create_dataloaders
from .evaluation import MetricsCalculator

# For convenient imports
__all__ = [
    'SteatosisModel',
    'get_loss_fn', 
    'create_dataloaders',
    'MetricsCalculator'
]