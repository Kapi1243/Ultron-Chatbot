"""
Ultron Character Bot - Configuration Management
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field
import os
import json


class ModelConfig(BaseSettings):
    """Model-specific configuration"""
    model_path: str = Field(default="ultron_model", description="Path to the model directory")
    max_tokens: int = Field(default=150, ge=10, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Generation temperature")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p sampling parameter")
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0, description="Repetition penalty")
    use_flash_attention: bool = Field(default=True, description="Use flash attention if available")
    enable_compilation: bool = Field(default=True, description="Enable torch.compile for optimization")


class QuantizationConfig(BaseSettings):
    """Quantization configuration for memory optimization"""
    load_in_4bit: bool = Field(default=True, description="Use 4-bit quantization")
    use_double_quant: bool = Field(default=True, description="Use double quantization")
    quant_type: str = Field(default="nf4", description="Quantization type")
    compute_dtype: str = Field(default="float16", description="Compute data type")


class AudioConfig(BaseSettings):
    """Audio input/output configuration"""
    tts_rate: int = Field(default=110, ge=50, le=200, description="TTS speech rate")
    stt_timeout: float = Field(default=5.0, ge=1.0, le=30.0, description="STT timeout in seconds")
    energy_threshold: int = Field(default=4000, ge=1000, le=10000, description="Audio energy threshold")
    enable_voice_commands: bool = Field(default=True, description="Enable voice recognition")
    fallback_to_text: bool = Field(default=True, description="Fallback to text input if voice fails")


class CacheConfig(BaseSettings):
    """Caching configuration"""
    max_cache_size: int = Field(default=20, ge=1, le=100, description="Maximum number of cached responses")
    enable_response_cache: bool = Field(default=True, description="Enable response caching")
    cache_timeout: int = Field(default=3600, ge=60, description="Cache timeout in seconds")


class PerformanceConfig(BaseSettings):
    """Performance monitoring and optimization"""
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    log_response_times: bool = Field(default=True, description="Log response generation times")
    auto_memory_cleanup: bool = Field(default=True, description="Automatic memory cleanup")
    cleanup_interval: int = Field(default=3, ge=1, le=10, description="Memory cleanup interval")


class UltronConfig(BaseSettings):
    """Main configuration class"""
    # Model settings
    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    
    # Audio settings
    audio: AudioConfig = Field(default_factory=AudioConfig)
    
    # Performance settings
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Development
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    
    class Config:
        env_prefix = "ULTRON_"
        case_sensitive = False


def load_config(config_file: Optional[str] = None) -> UltronConfig:
    """Load configuration from file or environment variables"""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        return UltronConfig(**config_data)
    
    return UltronConfig()


def save_config(config: UltronConfig, config_file: str) -> None:
    """Save configuration to file"""
    config_dict = config.dict()
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)


# Global configuration instance
_config: Optional[UltronConfig] = None


def get_config() -> UltronConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: UltronConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config