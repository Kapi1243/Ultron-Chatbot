# Ultron Chatbot# 🤖 Ultron Character Bot



AI-powered Ultron character chatbot with fine-tuned Mistral-7B.



## FeaturesA state-of-the-art AI character implementation featuring **Ultron** inpired by Marvel's Avengers, built with professional software architecture, advanced optimization, and voice capabilities. Perfect for showcasing AI engineering skills to leading tech companies.



- **Personality-Enhanced AI**: Mistral-7B-Instruct-v0.2 with LoRA fine-tuning for authentic Ultron responses[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)

- **4-bit Quantization**: Optimized for 8GB VRAM GPUs (RTX 4060 Ti tested)[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)

- **Conversation Memory**: Maintains context across interactions[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

- **Audio Caching**: Pre-generated responses cached for instant playback

## ✨ Key Features

## Requirements

### 🧠 **Advanced AI Capabilities**

### Hardware- **QLoRA Fine-tuning**: Memory-efficient training with 4-bit quantization

- **GPU**: NVIDIA with 8GB+ VRAM (NVIDIA 4060 TI Used) - **Response Caching**: LRU cache for faster repeated queries  

- **RAM**: 16GB+ recommended- **Mixed Precision**: Automatic FP16 inference optimization

- **Storage**: ~15GB for model files- **Smart Memory Management**: Auto-cleanup and CUDA optimization

- **Conversation Context**: Multi-turn dialogue with personality consistency

### Software

- **Python**: 3.12.x### 🎤 **Voice Interface**

- **CUDA**: 11.8+ (for GPU acceleration)- **Push-to-Talk**: Space bar activation for voice input

- **ffmpeg**: Required for audio effects ([Install Guide](https://ffmpeg.org/download.html))- **Text-to-Speech**: Optimized voice selection for character immersion

  - Windows: `winget install ffmpeg`- **Fallback Support**: Automatic text input when audio unavailable

- **Noise Handling**: Dynamic energy threshold adjustment

## Installation

### 🏗️ **Professional Architecture**

1. **Clone repository**- **Modular Design**: Clean separation of concerns (Model, Audio, Personality)

```bash- **Configuration Management**: JSON/environment variable configuration

git clone <your-repo-url>- **Error Handling**: Comprehensive exception management with graceful degradation

cd gpt2-character-bot- **Performance Monitoring**: Real-time metrics and memory tracking

```- **Logging**: Structured logging with configurable levels



2. **Create virtual environment**### ⚡ **Performance Optimized**

```bash- **4-bit Quantization**: Runs efficiently on 8GB VRAM

python -m venv venv- **Model Compilation**: PyTorch 2.0 optimization when available

venv\Scripts\activate  # Windows- **Batch Processing Ready**: Designed for scalable deployment

```- **Resource Monitoring**: GPU memory tracking and auto-cleanup



3. **Install dependencies**## 🚀 Quick Start

```bash

pip install -r requirements.txt### Prerequisites

```- Python 3.8+

- CUDA-capable GPU (8GB+ VRAM recommended)

4. **Download base model**- Windows 10/11, Linux, or macOS

   - Download [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

   - Place in `models/Mistral-7B-Instruct-v0.2/`### Installation



5. **Add Ultron LoRA adapter**1. **Clone the repository**

   - Place fine-tuned adapter in `ultron_model/````bash

   - Ensure `adapter_model.safetensors` and `adapter_config.json` existgit clone https://github.com/yourusername/ultron-character-bot.git

cd ultron-character-bot

## Usage```



Run the chatbot:2. **Install dependencies**

```bash```bash

python ultron_app.pypip install -r requirements.txt

``````



Voice-only mode:3. **Run the application**

```bash```bash

python ultron_wav_player.pypython ultron_app.py

``````



## Configuration## 📖 Usage Guide



Edit `config/default_config.json`:### Basic Interaction

```bash

```json# Start the bot

{python ultron_app.py

  "model": {

    "base_model_path": "models/Mistral-7B-Instruct-v0.2",# Hold SPACE to speak or type your message

    "adapter_model_path": "ultron_model",# Say "exit" or "quit" to terminate

    "load_in_4bit": true,# Type "stats" for performance metrics

    "device_map": {"": 0}# Type "clear" to reset conversation history

  },```

  "generation": {

    "max_new_tokens": 150,### Configuration

    "temperature": 0.8,Ultron now layers configuration in the following order:

    "top_p": 0.9

  }1. Built-in sane defaults

}2. `config/default_config.json` (ships with the repo)

```3. File referenced by `ULTRON_CONFIG`/`ULTRON_CONFIG_PATH`

4. CLI `--config` argument

**Key Settings:**

- `load_in_4bit`: Enable 4-bit quantization (~4GB VRAM)Create or override settings with a `config.json` file when needed:

- `device_map`: GPU assignment (`{"": 0}` = single GPU)

- `max_new_tokens`: Response length limit```json

{

## Technical Details  "model": {

    "max_tokens": 150,

### Model Architecture    "temperature": 0.7,

- **Base**: Mistral-7B-Instruct-v0.2 (7B parameters)    "top_p": 0.9

- **Adapter**: LoRA fine-tuned on Ultron dialogue  },

- **Quantization**: 4-bit NF4 with double quantization  "audio": {

- **VRAM**: ~3.5-4GB (4-bit mode)    "tts_rate": 110,

- **Load Time**: ~12 seconds (GPU-only)    "enable_voice_commands": true,

    "fallback_to_text": true,

### Voice System    "test_tts_on_startup": false

- **Engine**: pyttsx3 (Windows SAPI5)  },

- **Effects**:   "performance": {

  - Pitch shift: -5% (deeper tone)    "enable_monitoring": true,

  - High-pass: 150Hz (remove rumble)    "auto_cleanup_interval": 3

  - Low-pass: 3500Hz (metallic ceiling)  }

  - Echo: 50ms delay, 30% decay}

- **Playback**: pygame```



## Troubleshooting### Command Line Options

```bash

**CUDA Out of Memory**python ultron_app.py --help

- Ensure `load_in_4bit: true` in configpython ultron_app.py --config config.json

- Close other GPU appspython ultron_app.py --model-path /path/to/model

- Reduce `max_new_tokens`python ultron_app.py --debug

```

**Voice Issues**

- Verify ffmpeg: `ffmpeg -version`## 🏛️ Architecture Overview

- Windows: Check SAPI5 voices installed

- Verify `output/` permissions```

ultron-character-bot/

**Model Not Loading**├── src/ultron/

- Check model files in `models/Mistral-7B-Instruct-v0.2/`│   ├── core/           # Model loading and inference

- Verify adapter in `ultron_model/`│   │   └── model.py

- Update CUDA drivers│   ├── audio/          # TTS/STT management  

│   │   └── manager.py

**Slow Generation**│   ├── personality/    # Character behavior

- Confirm GPU usage: `nvidia-smi`│   │   └── core.py

- Verify `device_map: {"": 0}` in config│   └── config.py       # Configuration management

- Disable CPU offload if enabled├── tests/              # Unit and integration tests

├── config/             # Configuration files

## License├── data/               # Training data and models

├── ultron_app.py       # Main application

Uses Mistral-7B under Apache 2.0 license.└── requirements.txt    # Dependencies

```

## Credits

### Core Components

- [Mistral AI](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) - Base model

- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - Quantization#### 🤖 **UltronModel** (`src/ultron/core/model.py`)

- [PEFT](https://github.com/huggingface/peft) - LoRA fine-tuning- Optimized model loading with quantization

- Cached inference with memory management
- Performance monitoring and error handling
- Mixed precision and model compilation support

#### 🎵 **AudioManager** (`src/ultron/audio/manager.py`)
- Modular TTS/STT engines with fallback support
- Push-to-talk functionality with customizable keys
- Audio system testing and error recovery
- Voice selection and rate optimization

#### 🧠 **UltronPersonality** (`src/ultron/personality/core.py`)
- Character-consistent prompt engineering
- Response post-processing and cleanup
- Personality metrics and quality assessment
- Few-shot learning examples

## 🔧 Performance Benchmarks

### Hardware Requirements
| Component | Minimum | Recommended | 
|-----------|---------|-------------|
| GPU VRAM  | 6GB     | 8GB+        |
| RAM       | 8GB     | 16GB+       |
| Storage   | 10GB    | 20GB+       |
| CPU       | 4 cores | 8+ cores    |

### Performance Metrics
| Metric | Value |
|--------|-------|
| Model Load Time | ~15-30 seconds |
| Average Response Time | 2-5 seconds |
| Memory Usage (4-bit) | ~4-6GB VRAM |
| Throughput | ~5-10 tokens/second |

### Optimization Features
- ✅ 4-bit quantization (75% memory reduction)
- ✅ Response caching (50% faster for repeated queries)
- ✅ Mixed precision inference (20% speedup)
- ✅ Model compilation (10-15% additional speedup)
- ✅ Smart memory cleanup (prevents OOM errors)

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/unit/ -v
```

### Integration Tests  
```bash
python -m pytest tests/integration/ -v
```

### Performance Tests
```bash
python -m pytest tests/performance/ -v
```

### Audio System Test
```bash
python -c "from src.ultron.audio.manager import AudioManager; AudioManager({}).test_audio_system()"
```

## 🚢 Deployment

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "ultron_app.py"]
```

### Cloud Deployment
- **AWS**: EC2 g4dn.xlarge or larger with GPU support
- **Google Cloud**: n1-standard-4 with Tesla T4 
- **Azure**: Standard_NC6s_v3 or equivalent

### API Server Mode
```python
# Future feature: REST API endpoint
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 🛠️ Development

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
flake8 src/ tests/
```

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement with tests: `tests/test_new_feature.py`
3. Update documentation: `README.md`, `CHANGELOG.md`
4. Submit pull request with benchmarks

### Code Quality Standards
- **Test Coverage**: >90% for core modules
- **Documentation**: Comprehensive docstrings and type hints
- **Performance**: Response time <5s, memory usage <8GB
- **Error Handling**: Graceful degradation for all failure modes

## 📊 Technical Highlights

### AI Engineering Excellence
- **Memory-Efficient Training**: QLoRA with 4-bit quantization
- **Production-Ready Inference**: Caching, batching, and optimization
- **Character Consistency**: Advanced prompt engineering and post-processing
- **Scalable Architecture**: Modular design ready for microservices

### Software Engineering Best Practices
- **Clean Architecture**: SOLID principles and dependency injection
- **Configuration Management**: Environment-based configuration
- **Error Resilience**: Comprehensive exception handling
- **Performance Monitoring**: Real-time metrics and alerting
- **Testing Strategy**: Unit, integration, and performance tests

### DevOps & Deployment
- **Containerization**: Docker support for consistent deployment
- **CI/CD Ready**: GitHub Actions workflows (planned)
- **Cloud Integration**: Multi-cloud deployment guides
- **Monitoring**: Performance dashboards and logging

### Technical Demonstrations
- **Model Optimization**: 4-bit quantization, caching, compilation
- **Voice AI**: Real-time STT/TTS integration
- **Character AI**: Consistent personality through prompt engineering
- **Production Systems**: Error handling, monitoring, scalability

### Current Features (v2.0)
- ✅ QLoRA fine-tuned Ultron personality
- ✅ Voice interface with push-to-talk
- ✅ Performance optimization and monitoring  
- ✅ Modular architecture
- ✅ Comprehensive error handling

### Planned Features (v2.1+)
- 🔄 Web UI with real-time chat
- 🔄 REST API with FastAPI
- 🔄 RAG integration for dynamic knowledge
- 🔄 Multi-modal input (vision + text)
- 🔄 Conversation memory persistence
- 🔄 A/B testing framework
- 🔄 Model ensemble support

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `black`, `flake8`, `pytest`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library and model ecosystem
- **Microsoft**: For QLoRA implementation and optimization techniques

## 📞 Contact & Support

- **GitHub Issues**: [Create an issue](https://github.com/Kapi1243/Ultron-Chatbot/issues)
