"""
Ultron Voice Effects - Advanced audio processing to match the MCU Ultron voice
"""

import numpy as np
import tempfile
import os
import logging
from typing import Optional
try:
    import soundfile as sf
    import scipy.signal
    from pydub import AudioSegment
    from pydub.effects import normalize, low_pass_filter, high_pass_filter
    from pydub.playback import play
    EFFECTS_AVAILABLE = True
except ImportError:
    EFFECTS_AVAILABLE = False


class UltronVoiceProcessor:
    """Advanced voice processor to create MCU Ultron-like voice effects"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.effects_enabled = EFFECTS_AVAILABLE
        
        if not self.effects_enabled:
            self.logger.warning("Advanced audio effects not available. Install soundfile, scipy, and pydub for full Ultron voice.")
    
    def apply_ultron_effects(self, audio_file_path: str) -> Optional[str]:
        """
        Apply Ultron voice effects to an audio file
        Returns path to the processed audio file
        """
        if not self.effects_enabled:
            return audio_file_path
            
        try:
            # Load audio
            audio = AudioSegment.from_wav(audio_file_path)
            
            # Apply effects in sequence to match Ultron's voice characteristics
            processed_audio = self._apply_metallic_distortion(audio)
            processed_audio = self._add_robotic_harmonics(processed_audio)
            processed_audio = self._adjust_pitch_and_formants(processed_audio)
            processed_audio = self._add_subtle_echo(processed_audio)
            processed_audio = self._apply_frequency_filtering(processed_audio)
            processed_audio = self._normalize_and_enhance(processed_audio)
            
            # Save processed audio
            output_path = tempfile.mktemp(suffix=".wav")
            processed_audio.export(output_path, format="wav")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error applying Ultron effects: {e}")
            return audio_file_path
    
    def _apply_metallic_distortion(self, audio: AudioSegment) -> AudioSegment:
        """Add metallic distortion characteristic of Ultron's voice"""
        try:
            # Convert to numpy array for processing
            samples = np.array(audio.get_array_of_samples())
            
            # Apply subtle bit crushing for digital artifacts
            bit_depth = 14  # Reduce from 16-bit to 14-bit for slight digital distortion
            samples = samples / (2**15)  # Normalize to [-1, 1]
            samples = np.round(samples * (2**bit_depth)) / (2**bit_depth)
            samples = samples * (2**15)  # Back to 16-bit range
            
            # Apply gentle saturation for warmth
            samples = np.tanh(samples * 0.8) * (2**15)
            
            # Convert back to AudioSegment
            samples = np.clip(samples, -32768, 32767).astype(np.int16)
            processed_audio = audio._spawn(samples.tobytes())
            
            return processed_audio
            
        except Exception as e:
            self.logger.warning(f"Metallic distortion failed: {e}")
            return audio
    
    def _add_robotic_harmonics(self, audio: AudioSegment) -> AudioSegment:
        """Add harmonic content to create robotic quality"""
        try:
            # Create a slight chorus effect by mixing with a pitch-shifted version
            higher_pitch = audio._spawn(audio.raw_data, 
                                      overrides={"frame_rate": int(audio.frame_rate * 1.02)})
            lower_pitch = audio._spawn(audio.raw_data,
                                     overrides={"frame_rate": int(audio.frame_rate * 0.98)})
            
            # Mix with original (70% original, 15% higher, 15% lower)
            mixed = audio.overlay(higher_pitch - 12).overlay(lower_pitch - 12)
            
            return mixed
            
        except Exception as e:
            self.logger.warning(f"Robotic harmonics failed: {e}")
            return audio
    
    def _adjust_pitch_and_formants(self, audio: AudioSegment) -> AudioSegment:
        """Adjust pitch to be slightly lower and more menacing"""
        try:
            # Lower the pitch slightly (about 5-8% lower)
            new_sample_rate = int(audio.frame_rate * 0.95)
            pitched_audio = audio._spawn(audio.raw_data, 
                                       overrides={"frame_rate": new_sample_rate})
            
            # Resample back to original rate to maintain playback speed but with lower pitch
            pitched_audio = pitched_audio.set_frame_rate(audio.frame_rate)
            
            return pitched_audio
            
        except Exception as e:
            self.logger.warning(f"Pitch adjustment failed: {e}")
            return audio
    
    def _add_subtle_echo(self, audio: AudioSegment) -> AudioSegment:
        """Add a subtle echo effect for depth"""
        try:
            # Create echo with 150ms delay and 25% volume
            echo_delay = 150  # milliseconds
            echo_volume = -12  # dB reduction
            
            # Create silence for delay
            silence = AudioSegment.silent(duration=echo_delay)
            
            # Create echo
            echo = silence + audio + AudioSegment.silent(duration=echo_delay)
            echo = echo + echo_volume  # Reduce volume
            
            # Mix with original
            mixed = audio.overlay(echo)
            
            return mixed
            
        except Exception as e:
            self.logger.warning(f"Echo effect failed: {e}")
            return audio
    
    def _apply_frequency_filtering(self, audio: AudioSegment) -> AudioSegment:
        """Apply EQ filtering to enhance Ultron's frequency signature"""
        try:
            # Apply high-pass filter to reduce muddy low frequencies
            filtered = high_pass_filter(audio, 80)
            
            # Apply low-pass filter to reduce harsh high frequencies  
            filtered = low_pass_filter(filtered, 8000)
            
            # Boost mid frequencies (300-2000 Hz) for voice clarity
            # This is a simplified version - real EQ would need more sophisticated processing
            
            return filtered
            
        except Exception as e:
            self.logger.warning(f"Frequency filtering failed: {e}")
            return audio
    
    def _normalize_and_enhance(self, audio: AudioSegment) -> AudioSegment:
        """Final normalization and enhancement"""
        try:
            # Normalize to consistent volume
            normalized = normalize(audio)
            
            # Slight compression effect by reducing dynamic range
            # This makes the voice more consistent and robotic
            threshold_db = -20
            ratio = 4
            
            # Simple dynamic range compression
            samples = np.array(normalized.get_array_of_samples()).astype(np.float32)
            samples = samples / (2**15)  # Normalize to [-1, 1]
            
            # Apply compression
            threshold = 10**(threshold_db/20)
            compressed_samples = []
            
            for sample in samples:
                if abs(sample) > threshold:
                    # Apply compression above threshold
                    if sample > 0:
                        compressed = threshold + (sample - threshold) / ratio
                    else:
                        compressed = -threshold + (sample + threshold) / ratio
                    compressed_samples.append(compressed)
                else:
                    compressed_samples.append(sample)
            
            # Convert back
            compressed_samples = np.array(compressed_samples) * (2**15)
            compressed_samples = np.clip(compressed_samples, -32768, 32767).astype(np.int16)
            
            result = normalized._spawn(compressed_samples.tobytes())
            return result
            
        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}")
            return audio
    
    def cleanup_temp_files(self, file_path: str):
        """Clean up temporary audio files"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp file {file_path}: {e}")