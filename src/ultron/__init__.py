"""
Ultron Character Bot Package
"""

__version__ = "2.0.0"
__author__ = "Your Name"
__description__ = "Professional Ultron character chatbot with voice capabilities"

from .core.model import UltronModel
from .audio.manager import AudioManager
from .personality.core import UltronPersonality

__all__ = ["UltronModel", "AudioManager", "UltronPersonality"]