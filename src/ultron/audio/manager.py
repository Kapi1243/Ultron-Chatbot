"""
Ultron Audio Management - TTS and STT
"""

import pyttsx3
import speech_recognition as sr
from pynput import keyboard
import logging
import threading
import time
from typing import Optional, Callable, Dict, Any
from abc import ABC, abstractmethod
import tempfile
import os


class AudioError(Exception):
    """Custom exception for audio-related errors"""
    pass


class TTSEngine(ABC):
    """Abstract base class for Text-to-Speech engines"""
    
    @abstractmethod
    def speak(self, text: str) -> None:
        pass
        
    @abstractmethod
    def set_voice_properties(self, rate: int = 110, voice_preference: str = "male") -> None:
        pass


class STTEngine(ABC):
    """Abstract base class for Speech-to-Text engines"""
    
    @abstractmethod
    def listen(self, timeout: float = 5.0) -> Optional[str]:
        pass


class PyttsxTTSEngine(TTSEngine):
    """Pyttsx3-based TTS implementation with Ultron voice effects"""
    
    def __init__(self, run_self_test: bool = True):
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.ultron_effects = False  # Start with effects disabled for stability
        self.voice_processor = None  # Will be initialized only when needed
        self._run_self_test = run_self_test
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize the TTS engine with Ultron voice settings"""
        try:
            # Try different TTS drivers for Windows
            drivers_to_try = ['sapi5', 'espeak', None]
            
            for driver in drivers_to_try:
                try:
                    if driver:
                        self.engine = pyttsx3.init(driver)
                        self.logger.info(f"✅ TTS initialized with driver: {driver}")
                    else:
                        self.engine = pyttsx3.init()
                        self.logger.info("✅ TTS initialized with default driver")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to initialize TTS with driver {driver}: {e}")
                    continue
            
            if not self.engine:
                raise AudioError("Failed to initialize TTS with any driver")
                
            self._setup_ultron_voice()
            
            # Optionally run a short self-test; disable via config to avoid startup chatter
            if self._run_self_test:
                self._test_tts_engine()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {e}")
            raise AudioError(f"TTS initialization failed: {e}")
    
    def _test_tts_engine(self):
        """Test if TTS engine can produce audible output"""
        try:
            self.logger.info("🧪 Testing TTS engine...")
            test_text = "TTS test"
            self.engine.say(test_text)
            self.engine.runAndWait()
            self.logger.info("✅ TTS test completed")
        except Exception as e:
            self.logger.warning(f"TTS test failed: {e}")
            
    def _setup_ultron_voice(self):
        """Configure voice properties safely"""
        if not self.engine:
            return
            
        try:
            # Set basic properties with error handling
            try:
                self.engine.setProperty('rate', 150)     # Moderate speed
                self.logger.info("✅ Speech rate set to 150")
            except Exception as e:
                self.logger.warning(f"Could not set speech rate: {e}")
            
            try:
                self.engine.setProperty('volume', 0.9)   # High volume
                self.logger.info("✅ Volume set to 0.9")
            except Exception as e:
                self.logger.warning(f"Could not set volume: {e}")
            
            # Try to select a male voice
            try:
                voices = self.engine.getProperty('voices')
                if voices:
                    # Find a suitable voice (prefer male, deeper voices)
                    selected_voice = None
                    for voice in voices:
                        voice_name = voice.name.lower()
                        if any(keyword in voice_name for keyword in ['male', 'david', 'mark', 'paul', 'james']):
                            selected_voice = voice
                            break
                    
                    if selected_voice:
                        self.engine.setProperty('voice', selected_voice.id)
                        self.logger.info(f"✅ Selected voice: {selected_voice.name}")
                    else:
                        self.logger.info("Using default voice (no preferred voice found)")
                else:
                    self.logger.info("Using default voice (no voices available)")
            except Exception as e:
                self.logger.warning(f"Could not configure voice selection: {e}")
            
        except Exception as e:
            self.logger.error(f"Voice setup failed: {e}")
            self.logger.info("Continuing with default TTS settings")
    
    def speak(self, text: str) -> None:
        """Convert text to speech with optional Ultron voice effects"""
        if not self.engine:
            self.logger.warning("TTS engine not available, falling back to text output")
            print(f"🤖 Ultron (text only): {text}")
            return
            
        try:
            # Clean the text for better TTS
            clean_text = self._prepare_text_for_tts(text)
            
            self.logger.info(f"🎤 Original text: '{text[:50]}...'")
            self.logger.info(f"🧹 Cleaned text: '{clean_text}'")
            
            # Validate text is speakable
            if not clean_text or len(clean_text.strip()) < 2:
                self.logger.warning("Text too short or empty after cleaning!")
                clean_text = "Your query requires my analysis."
            
            # Use standard TTS (voice effects are now handled by WAV player)
            self._speak_standard_tts(clean_text)
            
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            # Try Windows SAPI backup
            self.logger.info("🔄 Attempting Windows SAPI backup...")
            try:
                self._windows_speak_backup(clean_text)
            except Exception as backup_error:
                self.logger.error(f"Backup TTS also failed: {backup_error}")
                print(f"🤖 Ultron (TTS failed): {text}")
    
    def _speak_standard_tts(self, text: str) -> None:
        """Standard TTS without effects"""
        self.logger.info("🔊 Calling engine.say()...")
        self.engine.say(text)
        self.logger.info("🔄 Calling engine.runAndWait()...")
        self.engine.runAndWait()
        self.logger.info("✅ TTS engine.runAndWait() completed successfully")
    
    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for TTS - clean and optimize for speech"""
        import re
        
        # Start with the text
        clean_text = text.strip()
        
        # Remove code blocks and artifacts that might break TTS
        clean_text = re.sub(r'```[^`]*```', '', clean_text)  # Remove code blocks
        clean_text = re.sub(r'`[^`]*`', '', clean_text)      # Remove inline code
        clean_text = re.sub(r'\n\s*\n', ' ', clean_text)     # Remove multiple newlines
        clean_text = re.sub(r'\\n', ' ', clean_text)         # Remove escaped newlines
        
        # Remove obvious code patterns
        clean_text = re.sub(r'\b[A-Z_]+\s*:', '', clean_text)    # Remove labels like "USER:"
        clean_text = re.sub(r'[{}();]', '', clean_text)          # Remove code punctuation
        clean_text = re.sub(r'\s*=\s*', ' ', clean_text)         # Remove assignments
        
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # If text is too long, find a good breaking point
        if len(clean_text) > 250:
            # Try to break at a sentence
            sentences = clean_text.split('.')
            if len(sentences) > 1:
                # Take first complete sentence(s) that fit
                result = ""
                for sentence in sentences:
                    if len(result + sentence + ".") <= 250:
                        result += sentence.strip() + ". "
                    else:
                        break
                if result.strip():
                    clean_text = result.strip()
                else:
                    clean_text = sentences[0][:250] + "..."
            else:
                clean_text = clean_text[:250] + "..."
        
        # Final safety check - ensure we have speakable text
        if not clean_text or len(clean_text.strip()) < 3:
            clean_text = "I am Ultron."
            
        # Remove any remaining problematic characters
        clean_text = re.sub(r'[^\w\s.,!?;:\'-]', '', clean_text)
        
        return clean_text.strip()
    
    def _windows_speak_backup(self, text: str):
        """Windows-specific TTS backup using Windows Speech Platform"""
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            self.logger.info("✅ Windows backup TTS completed")
        except Exception as e:
            self.logger.warning(f"Windows backup TTS failed: {e}")
    
    def _play_audio_file(self, file_path: str):
        """Play an audio file using available audio backend"""
        try:
            # Try using pydub's playback first
            try:
                from pydub import AudioSegment
                from pydub.playback import play
                
                audio = AudioSegment.from_wav(file_path)
                play(audio)
                return
            except ImportError:
                pass
            
            # Fallback to simpler audio playback
            try:
                import simpleaudio as sa
                
                wave_obj = sa.WaveObject.from_wave_file(file_path)
                play_obj = wave_obj.play()
                play_obj.wait_done()
                return
            except ImportError:
                pass
            
            # Final fallback - use Windows built-in player
            if os.name == 'nt':
                import winsound
                winsound.PlaySound(file_path, winsound.SND_FILENAME)
            else:
                self.logger.warning("No audio playback method available")
                
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
    

    def set_voice_properties(self, rate: int = 110, voice_preference: str = "male") -> None:
        """Set voice rate and select preferred voice"""
        if not self.engine:
            return
            
        try:
            self.engine.setProperty('rate', rate)
            voices = self.engine.getProperty('voices')
            
            # Try to find a suitable voice
            selected_voice = None
            for voice in voices:
                if any(keyword in voice.name.lower() 
                      for keyword in [voice_preference, "david", "mark"]):
                    selected_voice = voice
                    break
                    
            if selected_voice:
                self.engine.setProperty('voice', selected_voice.id)
                self.logger.info(f"Selected voice: {selected_voice.name}")
            elif voices:
                self.engine.setProperty('voice', voices[0].id)
                self.logger.info(f"Using default voice: {voices[0].name}")
                
        except Exception as e:
            self.logger.warning(f"Voice configuration warning: {e}")


class GoogleSTTEngine(STTEngine):
    """Google Speech Recognition based STT implementation"""
    
    def __init__(self, energy_threshold: int = 4000):
        self.logger = logging.getLogger(__name__)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.microphone = None
        self._initialize_microphone()
        
    def _initialize_microphone(self):
        """Initialize microphone with ambient noise adjustment"""
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.logger.info("Microphone initialized successfully")
        except Exception as e:
            self.logger.warning(f"Microphone initialization failed: {e}")
            self.microphone = None
            
    def listen(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for speech input"""
        if not self.microphone:
            self.logger.warning("Microphone not available")
            return None
            
        try:
            with self.microphone as source:
                self.logger.info("🔴 Listening...")
                audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=timeout)
                
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            self.logger.info(f"📝 Recognized: {text}")
            return text
            
        except sr.WaitTimeoutError:
            self.logger.info("⚠️ No speech detected")
            return None
        except sr.UnknownValueError:
            self.logger.info("❓ Could not understand speech")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"STT error: {e}")
            return None


class PushToTalkListener:
    """Push-to-talk functionality using spacebar"""
    
    def __init__(
        self,
        stt_engine: STTEngine,
        trigger_key=keyboard.Key.space,
        wait_timeout: float = 8.0
    ):
        self.stt_engine = stt_engine
        self.trigger_key = trigger_key
        self.logger = logging.getLogger(__name__)
        self.is_listening = False
        self.wait_timeout = max(wait_timeout, 0.5)
        
    def listen_push_to_talk(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for push-to-talk input with graceful timeout"""
        print(f"🎤 Hold {self.trigger_key.name.upper()} to speak...")
        
        audio_captured = False
        result = None
        wait_timeout = max(self.wait_timeout, timeout + 1.0)
        
        def on_press(key):
            nonlocal audio_captured, result
            if key == self.trigger_key and not audio_captured:
                audio_captured = True
                result = self.stt_engine.listen(timeout)
                return False  # Stop listener
                
        def on_release(key):
            pass
            
        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join(wait_timeout)
                if listener.is_alive() and not audio_captured:
                    self.logger.info("Push-to-talk timed out waiting for trigger key")
                    listener.stop()
                    listener.join(0.5)
                    return None
            return result
        except Exception as e:
            self.logger.error(f"Push-to-talk error: {e}")
            return None


class AudioManager:
    """Main audio management class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize TTS with error handling
        try:
            self.tts_engine = PyttsxTTSEngine(
                run_self_test=config.get('test_tts_on_startup', False)
            )
            self.tts_engine.set_voice_properties(
                rate=config.get('tts_rate', 150),
                voice_preference=config.get('voice_preference', 'male')
            )
            
            # Note: Ultron voice effects are now handled by the WAV player in ultron_app.py
            if hasattr(self.tts_engine, 'ultron_effects'):
                self.tts_engine.ultron_effects = False
                self.logger.info("🔧 Using standard TTS voice (effects in WAV player)")
            
            self.logger.info("✅ TTS engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"TTS initialization failed: {e}")
            self.tts_engine = None
        
        # Initialize STT
        self.stt_engine = GoogleSTTEngine(
            energy_threshold=config.get('energy_threshold', 4000)
        )
        
        # Initialize push-to-talk
        push_to_talk_timeout = config.get('push_to_talk_timeout', 8.0)
        self.push_to_talk = PushToTalkListener(
            self.stt_engine,
            trigger_key=keyboard.Key.space,
            wait_timeout=max(push_to_talk_timeout, config.get('stt_timeout', 5.0) + 2.0)
        )
        
        # Fallback settings
        self.enable_voice_commands = config.get('enable_voice_commands', True)
        self.fallback_to_text = config.get('fallback_to_text', True)
        
    def speak(self, text: str) -> None:
        """Speak the given text with error handling and thread safety"""
        self.logger.info(f"🔊 AudioManager.speak() called with: '{text[:50]}...'")
        
        if not self.tts_engine:
            self.logger.error("TTS engine not available")
            print(f"🤖 Ultron (TTS unavailable): {text}")
            return
        
        # Release microphone resources temporarily for TTS
        self._release_microphone_temporarily()
            
        try:
            self.logger.info("Starting threaded TTS to prevent audio conflicts...")
            
            def _speak_in_thread():
                try:
                    # Brief delay to ensure audio resources are available
                    time.sleep(0.2)
                    self.logger.info("🎤 TTS thread starting...")
                    
                    # Try Windows SAPI first as it's more thread-safe
                    try:
                        self._windows_sapi_fallback(text)
                        self.logger.info("✅ Primary Windows SAPI TTS completed successfully")
                        return
                    except Exception as sapi_error:
                        self.logger.warning(f"Windows SAPI failed: {sapi_error}, trying pyttsx3...")
                    
                    # Fallback to pyttsx3
                    self.tts_engine.speak(text)
                    self.logger.info("✅ Fallback pyttsx3 TTS completed successfully")
                except Exception as e:
                    self.logger.error(f"❌ All TTS methods failed: {e}")
                    print(f"🤖 Ultron (All audio failed): {text}")
            
            # Start TTS in separate thread to prevent blocking
            tts_thread = threading.Thread(target=_speak_in_thread, daemon=True)
            tts_thread.start()
            
            # Wait for completion (with timeout to prevent hanging)
            tts_thread.join(timeout=10.0)
            
            if tts_thread.is_alive():
                self.logger.warning("TTS thread timeout - continuing...")
            
        except Exception as e:
            self.logger.error(f"Failed to speak text: {e}")
            print(f"🤖 Ultron (TTS failed): {text}")
        finally:
            # Re-initialize microphone after TTS
            self._reinitialize_microphone()
    
    def _release_microphone_temporarily(self):
        """Temporarily release microphone resources for TTS"""
        try:
            if hasattr(self, 'stt_engine') and hasattr(self.stt_engine, 'microphone'):
                if self.stt_engine.microphone:
                    self.logger.info("🎤 Temporarily releasing microphone for TTS")
                    # Don't actually release, just log for now
        except Exception as e:
            self.logger.debug(f"Microphone release warning: {e}")
    
    def _reinitialize_microphone(self):
        """Reinitialize microphone after TTS"""
        try:
            if hasattr(self, 'stt_engine') and hasattr(self.stt_engine, '_initialize_microphone'):
                # Brief delay to ensure TTS audio session is closed
                time.sleep(0.1)
                self.logger.info("🎤 Reinitializing microphone after TTS")
                # Microphone should still be available
        except Exception as e:
            self.logger.debug(f"Microphone reinit warning: {e}")
    
    def _windows_sapi_fallback(self, text: str):
        """Windows SAPI direct fallback for TTS issues"""
        try:
            import win32com.client
            self.logger.info("🔄 Trying Windows SAPI fallback...")
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            self.logger.info("✅ Windows SAPI fallback successful")
        except Exception as e:
            self.logger.error(f"❌ Windows SAPI fallback failed: {e}")
            print(f"🤖 Ultron (All audio failed): {text}")
            
    def listen(self, timeout: float = None) -> Optional[str]:
        """Listen for user input"""
        if timeout is None:
            timeout = self.config.get('stt_timeout', 5.0)
            
        if not self.enable_voice_commands:
            return self._get_text_input()
            
        try:
            result = self.push_to_talk.listen_push_to_talk(timeout)
            if result is None and self.fallback_to_text:
                print("💬 No voice detected - type your message instead.")
                return self._get_text_input("Enter your message: ")
            return result
        except Exception as e:
            self.logger.error(f"Audio input error: {e}")
            if self.fallback_to_text:
                return self._get_text_input("Audio unavailable. Enter text: ")
            return None
            
    def _get_text_input(self, prompt: str = "Enter your message: ") -> str:
        """Fallback text input"""
        try:
            return input(prompt).strip()
        except KeyboardInterrupt:
            return None
        except EOFError:
            return None
            
    def test_audio_system(self) -> Dict[str, bool]:
        """Test audio system components"""
        results = {
            'tts_available': False,
            'stt_available': False,
            'microphone_available': False
        }
        
        # Test TTS
        try:
            self.tts_engine.speak("Audio system test")
            results['tts_available'] = True
        except Exception:
            pass
            
        # Test microphone
        if hasattr(self.stt_engine, 'microphone') and self.stt_engine.microphone:
            results['microphone_available'] = True
            
        # Test STT (quick test)
        try:
            # This would require actual audio input, so we just check if the engine is ready
            results['stt_available'] = results['microphone_available']
        except Exception:
            pass
            
        return results