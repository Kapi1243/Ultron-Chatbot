"""
UltronWAVPlayer: Text-to-speech with movie-accurate voice effects
Uses pyttsx3 for TTS and pydub for audio processing (pitch shift, filters, echo)
"""
import pygame
import os
import hashlib

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.effects import low_pass_filter, high_pass_filter
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

class UltronWAVPlayer:
    def __init__(self, enable_effects=True):
        self.enable_effects = enable_effects and PYDUB_AVAILABLE
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.cached_files = self._count_cached_files()
        print(f" Loaded {self.cached_files} cached files")
        print(" Ultron Voice Player initialized!")
        if self.enable_effects:
            print(" Ultron effects ENABLED (metallic/robotic)")
        else:
            print(" Ultron effects DISABLED")
    
    def _apply_ultron_effects(self, audio_path):
        if not PYDUB_AVAILABLE:
            return audio_path
        try:
            print(" Applying Ultron effects...")
            audio = AudioSegment.from_wav(audio_path)
            
            # Pitch shift down slightly for deeper voice
            audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * 0.95)})
            audio = audio.set_frame_rate(22050)
            
            # Metallic high-pass filter
            audio = high_pass_filter(audio, cutoff=150)
            
            # Remove harsh highs
            audio = low_pass_filter(audio, cutoff=3500)
            
            # Robotic echo
            echo = audio - 12
            audio = audio.overlay(echo, position=50)
            
            # Normalize
            audio = audio.normalize(headroom=0.1) + 2
            
            audio.export(audio_path, format="wav")
            print(" Ultron effects applied!")
            return audio_path
        except Exception as e:
            print(f" Effects failed: {e}")
            return audio_path
    
    def speak_ultron(self, text):
        try:
            cleaned_text = text.strip()
            if not cleaned_text:
                return False
            print(f" Ultron speaking: {cleaned_text}")
            text_hash = hashlib.md5(f"{cleaned_text}_effects_{self.enable_effects}".encode()).hexdigest()[:12]
            filename = f"ultron_{text_hash}.wav"
            filepath = os.path.join(self.output_dir, filename)
            
            if not os.path.exists(filepath):
                print(f" Creating voice file...")
                if not PYTTSX3_AVAILABLE:
                    print(f" pyttsx3 not available")
                    return False
                
                # Generate with pyttsx3 - use simple direct approach
                temp_path = filepath + ".temp"
                try:
                    # Create engine just for this one generation
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 140)
                    engine.setProperty('volume', 0.9)
                    
                    # Set male voice
                    voices = engine.getProperty('voices')
                    for voice in voices:
                        if 'david' in voice.name.lower() or 'male' in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                    
                    # Save directly to file
                    engine.save_to_file(cleaned_text, temp_path)
                    engine.runAndWait()
                    del engine
                    
                    print(f" Voice created")
                    
                    # Apply effects if enabled
                    if self.enable_effects:
                        self._apply_ultron_effects(temp_path)
                    
                    # Move to final location
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    os.rename(temp_path, filepath)
                    
                except Exception as e:
                    print(f" TTS failed: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return False
            else:
                print(f" Using cached: {filename}")
            
            print(f" Playing...")
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            print(" Done")
            return True
        except Exception as e:
            print(f" Playback failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _count_cached_files(self):
        try:
            files = [f for f in os.listdir(self.output_dir) if f.endswith('.wav') and f.startswith('ultron_')]
            return len(files)
        except:
            return 0
