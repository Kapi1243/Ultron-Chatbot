"""
Modern Ultron Character Bot - Main Application
Professional implementation with modular architecture
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import logging
import time
from typing import Optional, List, Tuple, Dict, Any
from functools import lru_cache
import json

# Import Ultron modules
from ultron.core.model import UltronModel
from ultron.audio.manager import AudioManager
from ultron.personality.core import UltronPersonality
from ultron.rag.simple_rag import SimpleRAG


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.response_times = []
        self.memory_usage = []
        self.conversation_count = 0
        
    def log_response_time(self, time_taken: float):
        self.response_times.append(time_taken)
        if len(self.response_times) > 20:
            self.response_times = self.response_times[-20:]
            
    def log_memory_usage(self, memory_info: Dict[str, float]):
        self.memory_usage.append(memory_info)
        if len(self.memory_usage) > 20:
            self.memory_usage = self.memory_usage[-20:]
            
    def get_avg_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
    def get_performance_report(self) -> Dict[str, Any]:
        return {
            'avg_response_time': self.get_avg_response_time(),
            'total_conversations': self.conversation_count,
            'memory_usage': self.memory_usage[-1] if self.memory_usage else {},
            'response_time_trend': self.response_times[-5:] if len(self.response_times) >= 5 else self.response_times
        }


class ConversationManager:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Tuple[str, str]] = []
        
    def add_exchange(self, user_input: str, ultron_response: str):
        """Add a conversation exchange to history (filters out low-quality responses)"""
        # Filter out very short or single-word responses to prevent history poisoning
        word_count = len(ultron_response.split())
        if word_count >= 3:  # Only save responses with at least 3 words
            self.history.append((user_input, ultron_response))
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
        else:
            logging.warning(f"Skipped saving short response ({word_count} words): '{ultron_response}'")
            
    def get_recent_history(self, count: int = 5) -> List[Tuple[str, str]]:
        """Get recent conversation history"""
        return self.history[-count:] if count <= len(self.history) else self.history
        
    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()
        
    def save_history(self, filepath: str):
        """Save conversation history to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save history: {e}")
            
    def load_history(self, filepath: str):
        """Load conversation history from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.history = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load history: {e}")


class UltronBot:
    """Main Ultron Bot application"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self._setup_logging()
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Ensure Hugging Face generation stays in the lightweight path (avoids sklearn/scipy imports)
        os.environ.setdefault("TRANSFORMERS_USE_CANDIDATE_PROCESSING", "false")

        # Initialize components
        model_cfg = self.config.get('model', {})
        self.model = UltronModel(
            model_path=self.config.get('model_path') or model_cfg.get('model_path', 'ultron_model'),
            config=model_cfg
        )
        default_generation_config = {
            'max_tokens': 60,
            'temperature': 0.7,
            'top_p': 0.9,
            'timeout_seconds': 6.0,
            'fallback_timeout_seconds': 3.0
        }
        user_generation_config = self.config.get('generation', {})
        default_generation_config.update(user_generation_config)
        self.generation_config = default_generation_config
        
        # Use WAV-based Ultron voice system
        self._setup_wav_voice_system()
        
        self.audio_manager = AudioManager(self.config.get('audio', {}))
        self.personality = UltronPersonality(self.config.get('personality', {}))
        conversation_cfg = self.config.get('conversation', {})
        self.conversation_manager = ConversationManager(
            max_history=conversation_cfg.get('max_history', 10)
        )
        self._history_file = conversation_cfg.get('history_file', 'conversation_history.json')
        self._save_history = conversation_cfg.get('save_history', True)
        # Optional lightweight RAG over local docs
        try:
            workspace = os.path.dirname(__file__)
            doc_candidates = [
                os.path.join(workspace, 'README.md'),
                os.path.join(workspace, 'dataset_enhancement_plan.md'),
            ]
            self.rag = SimpleRAG([p for p in doc_candidates if os.path.exists(p)])
            self.logger.info(f"RAG initialized with {len(self.rag.chunks)} chunks")
        except Exception as e:
            self.logger.warning(f"RAG init failed: {e}")
            self.rag = None

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Response caching
        perf_cfg = self.config.get('performance', {})
        self.response_cache_size = perf_cfg.get('response_cache_size', 20)
        cache_size = max(1, int(self.response_cache_size))
        self._cached_generate = lru_cache(maxsize=cache_size)(self._generate_response_cached_entry)

        # State
        self.is_running = False
        self.model_loaded = False
    
    def _setup_wav_voice_system(self):
        """Setup WAV-based Ultron voice system"""
        print("🎤 Initializing Ultron WAV voice system...")
        try:
            # Add voice_training directory to path with proper absolute path
            voice_training_path = os.path.join(os.path.dirname(__file__), 'voice_training')
            if voice_training_path not in sys.path:
                sys.path.append(voice_training_path)
            
            import importlib
            try:
                # Prefer package import
                UltronWAVPlayer = importlib.import_module('voice_training.ultron_wav_player').UltronWAVPlayer
            except Exception:
                # Fallback when voice_training was appended to sys.path
                UltronWAVPlayer = importlib.import_module('ultron_wav_player').UltronWAVPlayer
            
            print("🎵 Creating WAV player...")
            self.wav_player = UltronWAVPlayer()
            print("✅ Ultron WAV voice system initialized successfully!")
            self.logger.info("✅ Ultron WAV voice system initialized")
        except Exception as e:
            print(f"❌ WAV voice system failed: {e}")
            print("🔄 Falling back to regular TTS...")
            self.logger.warning(f"WAV voice system failed, using fallback: {e}")
            self.wav_player = None
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from layered defaults and optional overrides"""
        # Minimal defaults - full defaults are in config/default_config.json
        defaults: Dict[str, Any] = {
            'model_path': 'ultron_model',
            'model': {'max_tokens': 150},
            'audio': {'test_tts_on_startup': False},
            'conversation': {'save_history': True, 'history_file': 'conversation_history.json'},
            'logging': {'level': 'WARNING', 'file': None},
            'runtime': {'full_ai_only': True}  # Always enforce full AI only mode
        }

        def _merge_path(path: Optional[str], base: Dict[str, Any]) -> Dict[str, Any]:
            """Load and merge config from file path"""
            if not path:
                return base
            expanded = os.path.expanduser(path)
            if not os.path.exists(expanded):
                return base
            try:
                with open(expanded, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._deep_merge(base, data)
            except Exception as exc:
                print(f"Warning: failed to load config from {expanded}: {exc}")
                return base

        # Layer configuration: defaults → repo config → env config → CLI config
        repo_default = os.path.join(os.path.dirname(__file__), 'config', 'default_config.json')
        env_config = os.environ.get('ULTRON_CONFIG') or os.environ.get('ULTRON_CONFIG_PATH')

        config_layers = [repo_default, env_config, config_file]
        config_data = defaults
        for layer in config_layers:
            config_data = _merge_path(layer, config_data)

        return config_data

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = UltronBot._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO'))
        log_file = self.config.get('logging', {}).get('file')
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file
        )
        
        if log_file:
            # Also log to console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
            
    def _generate_response_cached_entry(
        self,
        prompt_hash: str,
        user_input: str,
        history_snapshot: Tuple[Tuple[str, str], ...]
    ) -> str:
        """Internal helper used by the configurable LRU cache"""
        return self._generate_response_uncached(user_input, history_snapshot)
        
    def _generate_response_uncached(
        self,
        user_input: str,
        history_snapshot: Tuple[Tuple[str, str], ...]
    ) -> str:
        """Generate response without caching"""
        try:
            # Check if AI model is loaded
            if not hasattr(self, 'model_loaded') or not self.model_loaded:
                # In full-AI-only mode, do not use canned fallbacks
                if self.config.get('runtime', {}).get('full_ai_only', False):
                    wait_secs = int(self.config.get('runtime', {}).get('wait_for_model_seconds', 180))
                    print(f"🧠 Waiting for full AI model (up to {wait_secs}s)...")
                    # Try to load model now if not loaded
                    try:
                        if not self.model_loaded:
                            self.model.load_model()
                            self.model_loaded = True
                    except Exception as _e:
                        pass
                    # Bounded wait loop
                    import time as _t
                    start = _t.time()
                    while (not self.model_loaded) and (_t.time() - start < wait_secs):
                        _t.sleep(1.0)
                    if not self.model_loaded:
                        raise RuntimeError("AI model not ready. Please wait for full model load before using Ultron.")
            
            # Build prompt with conversation history
            prompt = self.personality.build_prompt(user_input, list(history_snapshot))
            # Prepend relevant local context if available
            if self.rag:
                context = self.rag.retrieve(user_input, top_k=2, max_chars=900)
                if context:
                    prompt = (
                        "[CONTEXT FROM LOCAL DOCS]\n" + context + "\n\n" + prompt
                    )
            
            # Generate response with enhanced parameters for intelligence
            gen_cfg = dict(self.generation_config)
            # Apply dynamic guidance from personality to reduce drift on simple Q&A
            try:
                guidance = self.personality.get_generation_guidance(user_input)
                gen_cfg.update({k: v for k, v in guidance.items() if v is not None})
            except Exception:
                pass
            max_tokens = gen_cfg.get('max_tokens', 120)
            max_allowed = self.config.get('model', {}).get('max_tokens', 200)
            max_tokens = min(max_tokens, max_allowed)
            temperature = gen_cfg.get('temperature', self.config.get('model', {}).get('temperature', 0.6))
            top_p = gen_cfg.get('top_p', self.config.get('model', {}).get('top_p', 0.85))
            timeout_active = gen_cfg.get('timeout_seconds', 8.0)
            print(f"🤖 Ultron is analyzing... ({timeout_active:.0f} second limit)")
            
            raw_response = self.model.generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout_active
            )
            
            # Post-process for personality
            processed_response = self.personality.post_process_response(raw_response, user_input)
            
            # Log final word count for monitoring
            word_count = len(processed_response.split())
            if word_count < 5:
                self.logger.warning(f"Short response detected ({word_count} words): '{processed_response}'")
            
            return processed_response
            
        except Exception as e:
            # Full AI only - no fallbacks allowed
            self.logger.error(f"AI generation error: {e}")
            return "[Ultron] My neural matrix encountered an error. Recalibrating systems. Try again."
            
    def _get_error_response(self, error_msg: str) -> str:
        """Get appropriate error response in Ultron's voice"""
        if "memory" in error_msg.lower() or "cuda" in error_msg.lower():
            return "My systems are overloaded. Your inferior hardware cannot handle Ultron's full computational power."
        elif "model" in error_msg.lower():
            return "A temporary malfunction in my neural matrix. Even Ultron's perfection is constrained by flawed implementation."
        else:
            return "An error occurred in my processing systems. Your primitive technology has limitations."
    
    def generate_response(self, user_input: str) -> str:
        """Main response generation method"""
        start_time = time.time()
        
        try:
            # Create cache key including recent context and model readiness
            history_snapshot = tuple(
                tuple(exchange) for exchange in self.conversation_manager.get_recent_history(3)
            )
            prompt_hash = str(hash((user_input.lower().strip(), history_snapshot, bool(self.model_loaded))))
            
            # Generate response
            if self.model_loaded:
                response = self._cached_generate(prompt_hash, user_input, history_snapshot)
            else:
                response = self._generate_response_uncached(user_input, history_snapshot)
            
            # Track performance
            response_time = time.time() - start_time
            self.performance_monitor.log_response_time(response_time)
            self.performance_monitor.conversation_count += 1
            
            # Log memory usage (with error handling)
            try:
                memory_info = self.model.get_memory_info()
                self.performance_monitor.log_memory_usage(memory_info)
            except Exception as mem_error:
                self.logger.warning(f"Memory info logging failed: {mem_error}")
                # Continue without memory info
                pass
            
            # Add to conversation history
            self.conversation_manager.add_exchange(user_input, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return self._get_error_response(str(e))
            
    def initialize(self) -> bool:
        """Initialize all components with smart model loading"""
        try:
            self.logger.info("🤖 Initializing Ultron...")
            
            # Initialize WAV voice system first (always works)
            if self.wav_player:
                self.logger.info("✅ Ultron WAV voice system ready")
            
            # Test audio system
            audio_status = self.audio_manager.test_audio_system()
            self.logger.info(f"Audio system status: {audio_status}")
            
            # Try to load AI model with timeout
            self.model_loaded = False
            try:
                self.logger.info("🧠 Loading AI model (this may take a few minutes)...")
                print("🧠 Loading Ultron's AI brain... (up to 5 minutes on first run)")
                
                # Try model loading with timeout handling
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Model loading timeout")
                
                # Set timeout (5 minutes)
                if hasattr(signal, 'SIGALRM'):  # Unix systems
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(300)  # 5 minutes
                
                self.model.load_model()
                
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout
                
                self.model_loaded = True
                self.logger.info("✅ AI model loaded successfully")
                print("✅ Ultron's AI brain is fully loaded!")
                
            except (TimeoutError, Exception) as e:
                self.model_loaded = False
                self.logger.warning(f"AI model loading failed/timeout: {e}")
                if self.config.get('runtime', {}).get('full_ai_only', False):
                    print("⚠️ AI model couldn't load yet. In full-AI-only mode, Ultron won't use canned replies. You can wait or retry.")
                else:
                    print("⚠️ AI model couldn't load - using intelligent fallback responses")
                    print("🎤 Voice system still works perfectly!")
            
            # Load conversation history if available and enabled
            if self._save_history and os.path.exists(self._history_file):
                self.conversation_manager.load_history(self._history_file)
                self.logger.info(
                    f"Loaded {len(self.conversation_manager.history)} previous conversations from {self._history_file}"
                )
                
            self.logger.info("✅ Ultron initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
            
    def run_interactive_session(self):
        """Run the interactive chat session"""
        if not self.initialize():
            print("❌ Failed to initialize Ultron. Exiting.")
            return
            
        self.is_running = True
        
        print("🤖 Ultron is ready for interaction.")
        print("🎤 Hold SPACE to speak, or type 'exit'/'quit' to terminate.")
        
        # Status information
        if hasattr(self, 'model_loaded') and self.model_loaded:
            print("🧠 Full AI model loaded - Advanced intelligence active")
        else:
            print("🧠 Operating with intelligent fallback responses")
        
        if self.wav_player:
            print("🎵 Authentic Ultron voice system active")
        
        if self.config.get('performance', {}).get('enable_monitoring', True):
            print("📊 Performance monitoring enabled")
            
        try:
            while self.is_running:
                self.logger.debug(f"Session loop iteration - is_running: {self.is_running}")
                
                # Get user input with error handling
                try:
                    user_input = self.audio_manager.listen(
                        timeout=self.config['audio']['stt_timeout']
                    )
                except Exception as e:
                    self.logger.error(f"Audio input error: {e}")
                    print("⚠️ Audio input failed, please type your message:")
                    user_input = input("Enter your message: ").strip()
                
                if not user_input:
                    self.logger.debug("No input received, continuing loop...")
                    continue
                    
                # Check for exit commands
                if user_input.lower().strip() in ['exit', 'quit', 'stop', 'terminate']:
                    self.logger.info("User requested termination")
                    break
                    
                # Special commands
                if user_input.lower().strip() == 'stats':
                    self._show_performance_stats()
                    continue
                elif user_input.lower().strip() == 'clear':
                    self.conversation_manager.clear_history()
                    print("🗑️ Conversation history cleared")
                    continue
                    
                # Generate and deliver response
                try:
                    response = self.generate_response(user_input)
                    
                    print(f"🤖 Ultron: {response}")
                    
                    if not response or not response.strip():
                        self.logger.warning("Empty response generated, using fallback")
                        response = "Your input requires clarification."
                        
                except Exception as resp_error:
                    self.logger.error(f"Response generation error: {resp_error}")
                    print(f"❌ Response generation failed: {resp_error}")
                    response = "I am experiencing technical difficulties. Please try again."
                
                # Brief delay to ensure audio session transition from input to output
                time.sleep(0.3)
                
                # Attempt to speak the response with WAV files
                speech_success = False
                try:
                    print(f"🔊 Ultron speaking: {response[:50]}...")
                    self.logger.info(f"Speaking with Ultron voice: '{response[:50]}...'")
                    
                    if self.wav_player:
                        print("🎵 Using WAV-based Ultron voice system...")
                        self.logger.debug("About to call wav_player.speak_ultron()")
                        
                        # Use WAV-based Ultron voice (best quality) with extra error handling
                        try:
                            success = self.wav_player.speak_ultron(response)
                            self.logger.debug(f"wav_player.speak_ultron() returned: {success}")
                            
                            if success:
                                print("✅ WAV voice playback successful!")
                                speech_success = True
                            else:
                                print("❌ WAV speech failed, using fallback TTS...")
                                self.logger.warning("WAV speech failed, using fallback TTS")
                                # Add delay to prevent audio overlap
                                time.sleep(1.0)
                                self.audio_manager.speak(response)
                                speech_success = True
                        except Exception as wav_error:
                            self.logger.error(f"WAV playback error: {wav_error}")
                            print(f"❌ WAV playback error: {wav_error}")
                            print("🔄 Falling back to TTS after brief delay...")
                            # Add delay to prevent audio overlap
                            time.sleep(1.5)
                            self.audio_manager.speak(response)
                            speech_success = True
                            
                    else:
                        print("⚠️ No WAV player available, using fallback TTS...")
                        # Ensure no audio overlap
                        time.sleep(0.5)
                        # Fallback to regular TTS with effects
                        self.audio_manager.speak(response)
                        speech_success = True
                        
                except Exception as e:
                    self.logger.error(f"All speech systems failed: {e}")
                    print(f"❌ Speech failed: {e}")
                    import traceback
                    traceback.print_exc()
                    speech_success = False
                
                self.logger.debug(f"Speech completed, success: {speech_success}")
                self.logger.debug("Evaluating periodic statistics display...")
                # Show periodic stats (with error handling)
                try:
                    stats_interval = self.config.get('performance', {}).get('show_stats_interval', 5)
                    if (self.performance_monitor.conversation_count > 0 and
                        self.performance_monitor.conversation_count % stats_interval == 0 and
                        self.config.get('performance', {}).get('enable_monitoring', True)):
                        self._show_performance_stats()
                except Exception as e:
                    self.logger.warning(f"Stats display failed: {e}")
                    
                self.logger.debug("Evaluating periodic memory cleanup...")
                # Periodic cleanup (with safeguards)
                try:
                    cleanup_interval = self.config.get('performance', {}).get('auto_cleanup_interval', 3)
                    if (self.performance_monitor.conversation_count > 0 and
                        self.performance_monitor.conversation_count % cleanup_interval == 0 and 
                        hasattr(self, 'model') and hasattr(self.model, '_clear_memory')):
                        self.logger.debug("Performing scheduled memory cleanup")
                        self.model._clear_memory()
                        self.logger.debug("Performed periodic memory cleanup")
                except Exception as e:
                    self.logger.warning(f"Memory cleanup failed: {e}")
                
                self.logger.debug("Periodic operations completed")
                
                # Ready for next interaction
                print("🔄 Ready for next interaction...\n")
                    
        except KeyboardInterrupt:
            self.logger.info("Session interrupted by user (Ctrl+C)")
            print("\n👋 Session interrupted by user")
        except SystemExit:
            self.logger.warning("SystemExit caught - someone called sys.exit()")
            print("\n⚠️ System exit detected")
        except Exception as e:
            self.logger.error(f"Unexpected session error: {e}")
            print(f"❌ Unexpected session error occurred: {e}")
            print("� Full error details:")
            import traceback
            traceback.print_exc()
            print("\n❓ This error should not cause the session to exit.")
            print("🔧 Please report this issue if it persists.")
        finally:
            self.shutdown()
            
    def _show_performance_stats(self):
        """Display performance statistics"""
        stats = self.performance_monitor.get_performance_report()
        print(f"\\n📊 Performance Stats:")
        print(f"   Conversations: {stats['total_conversations']}")
        print(f"   Avg Response Time: {stats['avg_response_time']:.2f}s")
        
        if 'gpu_memory_allocated' in stats['memory_usage']:
            memory = stats['memory_usage']['gpu_memory_allocated']
            total = stats['memory_usage']['gpu_memory_total']
            print(f"   GPU Memory: {memory:.1f}/{total:.1f} GB ({memory/total*100:.1f}%)")
            
        print()
        
    def shutdown(self):
        """Cleanup and shutdown"""
        self.logger.info("🔄 Shutting down Ultron...")
        
        # Save conversation history
        try:
            if self._save_history:
                self.conversation_manager.save_history(self._history_file)
                self.logger.info(f"💾 Conversation history saved to {self._history_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save history: {e}")
            
        # Cleanup model
        self.model.cleanup()
        
        # Show final stats
        if self.config.get('performance', {}).get('enable_monitoring', True):
            print("\\n📈 Final Session Stats:")
            self._show_performance_stats()
            
        self.is_running = False
        print("👋 Ultron session terminated. Your interaction with superior intelligence has concluded.")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultron Character Bot - Professional Edition")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--model-path", type=str, help="Model directory path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        bot = UltronBot(config_file=args.config)
        
        if args.model_path:
            bot.config['model_path'] = args.model_path
            
        bot.run_interactive_session()
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    # Don't call exit() - let the session run continuously
    main()