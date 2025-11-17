"""
IRB PD Model Segmentation Framework

A production-ready segmentation framework for Internal Ratings-Based (IRB)
Probability of Default models.
"""

from .params import IRBSegmentationParams
from .engine import IRBSegmentationEngine
from .validators import SegmentValidator
from .config import (
    SegmentationConfig,
    DataConfig,
    OutputConfig,
    create_default_config
)
from .pipeline import SegmentationPipeline
from .config_builder import ConfigBuilder, quick_config

__version__ = "0.2.0"
__all__ = [
    "IRBSegmentationParams",
    "IRBSegmentationEngine",
    "SegmentValidator",
    "SegmentationConfig",
    "DataConfig",
    "OutputConfig",
    "create_default_config",
    "SegmentationPipeline",
    "ConfigBuilder",
    "quick_config"
]
