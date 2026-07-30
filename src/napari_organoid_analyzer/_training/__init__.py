"""
Training module for napari-organoid-analyzer.

This module provides functionality for training custom models on organoid detection data.
"""

from .training_thread import TrainingThread
from .architecture_manager import ArchitectureManager
from .training_widget import TrainingWidget

__all__ = ['TrainingThread', 'ArchitectureManager', 'TrainingWidget']
