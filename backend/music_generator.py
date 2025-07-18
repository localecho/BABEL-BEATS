#!/usr/bin/env python3
"""
Music generation module for BABEL-BEATS
Creates personalized music based on language features
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from dataclasses import dataclass
import logging
from scipy import signal
import torch
import torchaudio

logger = logging.getLogger(__name__)


@dataclass
class MusicParameters:
    """Parameters for music generation"""
    tempo: float
    key: str
    time_signature: str
    style: str
    duration: int
    instruments: List[str]
    complexity: str
    focus_areas: List[str]


class MusicGenerator:
    """
    AI-powered music generation based on language patterns
    """
    
    def __init__(self):
        self.sample_rate = 44100
        self.default_duration = 30
        
        # Musical scales for different moods/languages
        self.scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "pentatonic": [0, 2, 4, 7, 9],  # Good for Asian languages
            "blues": [0, 3, 5, 6, 7, 10],
            "chromatic": list(range(12))
        }
        
        # Instrument synthesis parameters
        self.instruments = {
            "piano": {"attack": 0.01, "decay": 0.5, "sustain": 0.3, "release": 0.5},
            "strings": {"attack": 0.1, "decay": 0.3, "sustain": 0.7, "release": 1.0},
            "flute": {"attack": 0.05, "decay": 0.1, "sustain": 0.8, "release": 0.3},
            "bells": {"attack": 0.001, "decay": 1.0, "sustain": 0.0, "release": 2.0},
            "drums": {"attack": 0.001, "decay": 0.1, "sustain": 0.0, "release": 0.1}
        }
        
        # Style mappings
        self.style_configs = {
            "classical": {
                "instruments": ["piano", "strings"],
                "tempo_range": (60, 120),
                "complexity": "medium"
            },
            "pop": {
                "instruments": ["piano", "drums"],
                "tempo_range": (100, 140),
                "complexity": "simple"
            },
            "ambient": {
                "instruments": ["strings", "bells"],
                "tempo_range": (50, 80),
                "complexity": "simple"
            },
            "traditional": {
                "instruments": ["flute", "strings"],
                "tempo_range": (70, 110),
                "complexity": "medium"
            }
        }
    
    def generate(self, 
                 language_features: Dict[str, Any],
                 style: str = "classical",
                 duration: int = 30,
                 preferences: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate music based on language features
        """
        # Extract parameters from language features
        params = self.extract_music_parameters(language_features, style, duration, preferences)
        
        # Generate base rhythm track
        rhythm_track = self.generate_rhythm_track(params, language_features.get("rhythm", {}))
        
        # Generate melody based on tone/pitch
        melody_track = self.generate_melody_track(params, language_features.get("tone", {}))
        
        # Generate harmony
        harmony_track = self.generate_harmony_track(params)
        
        # Mix tracks
        final_audio = self.mix_tracks([rhythm_track, melody_track, harmony_track])
        
        # Apply effects
        final_audio = self.apply_effects(final_audio, params)
        
        # Generate metadata
        metadata = self.generate_metadata(params, language_features)
        
        return final_audio, metadata
    
    def extract_music_parameters(self, features: Dict, style: str, duration: int, preferences: Optional[Dict]) -> MusicParameters:
        """
        Extract music parameters from language features
        """
        # Get style configuration
        style_config = self.style_configs.get(style, self.style_configs["classical"])
        
        # Determine tempo from speech rhythm
        rhythm_features = features.get("rhythm", {})
        speech_tempo = rhythm_features.get("tempo", 120)
        
        # Map speech tempo to musical tempo
        tempo_range = style_config["tempo_range"]
        tempo = np.clip(speech_tempo, tempo_range[0], tempo_range[1])
        
        # Determine key based on language
        language = features.get("language", "en-US")
        key = self.select_key(language)
        
        # Get instruments
        instruments = preferences.get("instruments", style_config["instruments"]) if preferences else style_config["instruments"]
        
        # Determine focus areas
        focus_areas = preferences.get("focus_areas", ["rhythm", "melody"]) if preferences else ["rhythm", "melody"]
        
        return MusicParameters(
            tempo=tempo,
            key=key,
            time_signature="4/4",
            style=style,
            duration=duration,
            instruments=instruments,
            complexity=style_config["complexity"],
            focus_areas=focus_areas
        )
    
    def generate_rhythm_track(self, params: MusicParameters, rhythm_features: Dict) -> np.ndarray:
        """
        Generate rhythm track based on speech rhythm
        """
        duration_samples = int(params.duration * self.sample_rate)
        rhythm_track = np.zeros(duration_samples)
        
        # Calculate beat positions
        beat_interval = 60.0 / params.tempo  # seconds per beat
        beat_samples = int(beat_interval * self.sample_rate)
        
        # Get stress pattern from language
        stress_pattern = rhythm_features.get("stress_pattern", [1, 0, 1, 0])
        
        # Generate rhythm
        current_sample = 0
        pattern_index = 0
        
        while current_sample < duration_samples:
            # Determine beat strength
            strength = stress_pattern[pattern_index % len(stress_pattern)]
            
            if strength > 0:
                # Generate a drum hit
                hit = self.synthesize_drum_hit(strength)
                
                # Add to track
                end_sample = min(current_sample + len(hit), duration_samples)
                rhythm_track[current_sample:end_sample] += hit[:end_sample - current_sample]
            
            # Move to next beat
            current_sample += beat_samples
            pattern_index += 1
        
        # Add subdivision for complexity
        if params.complexity in ["medium", "advanced"]:
            rhythm_track = self.add_rhythm_subdivisions(rhythm_track, params, rhythm_features)
        
        return rhythm_track * 0.3  # Reduce volume
    
    def generate_melody_track(self, params: MusicParameters, tone_features: Dict) -> np.ndarray:
        """
        Generate melody based on tone patterns
        """
        duration_samples = int(params.duration * self.sample_rate)
        melody_track = np.zeros(duration_samples)
        
        # Get scale
        scale_type = "pentatonic" if params.key in ["zh-CN", "ja-JP"] else "major"
        scale = self.scales[scale_type]
        
        # Base frequency for the key
        base_freq = 261.63  # C4
        
        # Generate melody from pitch contour
        pitch_contour = tone_features.get("pitch_contour", [])
        
        if pitch_contour:
            # Map pitch contour to musical notes
            notes = self.map_pitch_to_notes(pitch_contour, scale, base_freq)
        else:
            # Generate random melody following scale
            notes = self.generate_random_melody(params, scale, base_freq)
        
        # Synthesize melody
        note_duration = int(0.5 * self.sample_rate)  # Half second per note
        current_sample = 0
        
        for i, (freq, duration) in enumerate(notes):
            if current_sample >= duration_samples:
                break
            
            # Synthesize note
            note_audio = self.synthesize_note(freq, duration, params.instruments[0])
            
            # Add to track
            end_sample = min(current_sample + len(note_audio), duration_samples)
            melody_track[current_sample:end_sample] += note_audio[:end_sample - current_sample]
            
            current_sample += int(duration * self.sample_rate)
        
        return melody_track * 0.5
    
    def generate_harmony_track(self, params: MusicParameters) -> np.ndarray:
        """
        Generate harmony/accompaniment track
        """
        duration_samples = int(params.duration * self.sample_rate)
        harmony_track = np.zeros(duration_samples)
        
        # Simple chord progression
        chord_progression = self.get_chord_progression(params.key)
        
        # Duration of each chord
        chord_duration = int(2 * 60 / params.tempo * self.sample_rate)  # 2 beats per chord
        
        current_sample = 0
        chord_index = 0
        
        while current_sample < duration_samples:
            # Get current chord
            chord = chord_progression[chord_index % len(chord_progression)]
            
            # Synthesize chord
            chord_audio = self.synthesize_chord(chord, chord_duration, params.instruments[-1])
            
            # Add to track
            end_sample = min(current_sample + len(chord_audio), duration_samples)
            harmony_track[current_sample:end_sample] += chord_audio[:end_sample - current_sample]
            
            current_sample += chord_duration
            chord_index += 1
        
        return harmony_track * 0.3
    
    def synthesize_drum_hit(self, strength: float) -> np.ndarray:
        """
        Synthesize a drum hit
        """
        duration = 0.1
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # Generate noise burst
        hit = np.random.normal(0, 0.1, len(t))
        
        # Apply envelope
        envelope = np.exp(-10 * t) * strength
        hit *= envelope
        
        # Add some low frequency content for kick
        if strength > 0.7:
            kick = np.sin(2 * np.pi * 60 * t) * np.exp(-5 * t)
            hit += kick * 0.5
        
        return hit
    
    def synthesize_note(self, frequency: float, duration: float, instrument: str) -> np.ndarray:
        """
        Synthesize a musical note
        """
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # Get instrument parameters
        inst_params = self.instruments.get(instrument, self.instruments["piano"])
        
        # Generate base waveform
        if instrument == "piano":
            # Multiple harmonics for piano
            note = np.sin(2 * np.pi * frequency * t)
            note += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
            note += 0.25 * np.sin(2 * np.pi * frequency * 3 * t)
        elif instrument == "strings":
            # Sawtooth-like for strings
            note = signal.sawtooth(2 * np.pi * frequency * t) * 0.3
            note += np.sin(2 * np.pi * frequency * t) * 0.7
        elif instrument == "flute":
            # Pure sine for flute
            note = np.sin(2 * np.pi * frequency * t)
            # Add slight vibrato
            vibrato = np.sin(2 * np.pi * 5 * t) * 0.01
            note *= (1 + vibrato)
        else:
            # Default sine wave
            note = np.sin(2 * np.pi * frequency * t)
        
        # Apply ADSR envelope
        envelope = self.apply_adsr(t, inst_params)
        note *= envelope
        
        return note
    
    def synthesize_chord(self, frequencies: List[float], duration: int, instrument: str) -> np.ndarray:
        """
        Synthesize a chord
        """
        chord = np.zeros(duration)
        
        for freq in frequencies:
            note_duration = duration / self.sample_rate
            note = self.synthesize_note(freq, note_duration, instrument)
            chord[:len(note)] += note
        
        # Normalize
        chord /= len(frequencies)
        
        return chord
    
    def apply_adsr(self, t: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply ADSR envelope
        """
        total_time = t[-1]
        attack_time = params["attack"]
        decay_time = params["decay"]
        sustain_level = params["sustain"]
        release_time = params["release"]
        
        envelope = np.zeros_like(t)
        
        # Attack
        attack_mask = t < attack_time
        envelope[attack_mask] = t[attack_mask] / attack_time
        
        # Decay
        decay_mask = (t >= attack_time) & (t < attack_time + decay_time)
        decay_t = t[decay_mask] - attack_time
        envelope[decay_mask] = 1 - (1 - sustain_level) * (decay_t / decay_time)
        
        # Sustain
        sustain_end = total_time - release_time
        sustain_mask = (t >= attack_time + decay_time) & (t < sustain_end)
        envelope[sustain_mask] = sustain_level
        
        # Release
        release_mask = t >= sustain_end
        release_t = t[release_mask] - sustain_end
        envelope[release_mask] = sustain_level * (1 - release_t / release_time)
        
        return np.clip(envelope, 0, 1)
    
    def map_pitch_to_notes(self, pitch_contour: List[float], scale: List[int], base_freq: float) -> List[Tuple[float, float]]:
        """
        Map pitch contour to musical notes
        """
        notes = []
        
        # Quantize pitch to scale degrees
        for pitch in pitch_contour[:20]:  # Limit to 20 notes
            # Find nearest scale degree
            scale_degree = int(round(pitch)) % len(scale)
            note_number = scale[scale_degree]
            
            # Calculate frequency
            frequency = base_freq * (2 ** (note_number / 12))
            
            # Random duration between 0.25 and 1 second
            duration = np.random.uniform(0.25, 1.0)
            
            notes.append((frequency, duration))
        
        return notes
    
    def generate_random_melody(self, params: MusicParameters, scale: List[int], base_freq: float) -> List[Tuple[float, float]]:
        """
        Generate a random melody following the scale
        """
        notes = []
        num_notes = int(params.duration * 2)  # Approximately 2 notes per second
        
        for _ in range(num_notes):
            # Random scale degree
            scale_degree = np.random.choice(scale)
            
            # Random octave variation
            octave = np.random.choice([0, 0, 0, 1, -1])  # Mostly same octave
            
            # Calculate frequency
            frequency = base_freq * (2 ** (scale_degree / 12)) * (2 ** octave)
            
            # Random duration
            duration = np.random.choice([0.25, 0.5, 0.5, 1.0])
            
            notes.append((frequency, duration))
        
        return notes
    
    def get_chord_progression(self, key: str) -> List[List[float]]:
        """
        Get a simple chord progression
        """
        # I-V-vi-IV progression in C major
        base_freq = 261.63  # C4
        
        chords = [
            [base_freq, base_freq * 5/4, base_freq * 3/2],  # C major
            [base_freq * 3/2, base_freq * 15/8, base_freq * 9/8],  # G major
            [base_freq * 5/3, base_freq * 5/4, base_freq * 3/2],  # A minor
            [base_freq * 4/3, base_freq * 5/3, base_freq],  # F major
        ]
        
        return chords
    
    def mix_tracks(self, tracks: List[np.ndarray]) -> np.ndarray:
        """
        Mix multiple audio tracks
        """
        # Find the length of the longest track
        max_length = max(len(track) for track in tracks)
        
        # Pad tracks to same length
        padded_tracks = []
        for track in tracks:
            if len(track) < max_length:
                padded = np.pad(track, (0, max_length - len(track)), mode='constant')
                padded_tracks.append(padded)
            else:
                padded_tracks.append(track)
        
        # Mix tracks
        mixed = np.sum(padded_tracks, axis=0)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val * 0.8
        
        return mixed
    
    def apply_effects(self, audio: np.ndarray, params: MusicParameters) -> np.ndarray:
        """
        Apply audio effects
        """
        # Apply reverb
        if params.style in ["classical", "ambient"]:
            audio = self.apply_reverb(audio, amount=0.3)
        
        # Apply compression
        audio = self.apply_compression(audio)
        
        # Apply EQ
        audio = self.apply_eq(audio, params.style)
        
        return audio
    
    def apply_reverb(self, audio: np.ndarray, amount: float = 0.2) -> np.ndarray:
        """
        Apply simple reverb effect
        """
        # Simple delay-based reverb
        delay_samples = int(0.05 * self.sample_rate)
        decay = 0.6
        
        reverb = np.zeros_like(audio)
        
        for i in range(3):
            delay = delay_samples * (i + 1)
            gain = decay ** (i + 1) * amount
            
            if delay < len(audio):
                reverb[delay:] += audio[:-delay] * gain
        
        return audio + reverb
    
    def apply_compression(self, audio: np.ndarray, threshold: float = 0.7, ratio: float = 4) -> np.ndarray:
        """
        Apply dynamic range compression
        """
        compressed = audio.copy()
        
        # Find samples above threshold
        above_threshold = np.abs(compressed) > threshold
        
        # Compress samples above threshold
        compressed[above_threshold] = threshold + (compressed[above_threshold] - threshold) / ratio
        
        return compressed
    
    def apply_eq(self, audio: np.ndarray, style: str) -> np.ndarray:
        """
        Apply equalization based on style
        """
        # Simple high-pass filter for clarity
        if style in ["pop", "classical"]:
            # High-pass at 100 Hz
            sos = signal.butter(2, 100, 'hp', fs=self.sample_rate, output='sos')
            audio = signal.sosfilt(sos, audio)
        
        return audio
    
    def select_key(self, language: str) -> str:
        """
        Select musical key based on language
        """
        key_mappings = {
            "zh-CN": "C_pentatonic",
            "ja-JP": "D_pentatonic", 
            "ko-KR": "G_pentatonic",
            "es-ES": "A_minor",
            "fr-FR": "F_major",
            "de-DE": "D_minor",
            "en-US": "C_major"
        }
        
        return key_mappings.get(language, "C_major")
    
    def generate_metadata(self, params: MusicParameters, features: Dict) -> Dict[str, Any]:
        """
        Generate metadata for the generated music
        """
        return {
            "duration": params.duration,
            "tempo": params.tempo,
            "key": params.key,
            "time_signature": params.time_signature,
            "style": params.style,
            "instruments": params.instruments,
            "language_mapping": {
                "rhythm_consistency": features.get("rhythm", {}).get("rhythm_consistency", 0),
                "tone_accuracy": features.get("tone", {}).get("tone_accuracy", 0),
                "focus_areas": params.focus_areas
            },
            "sample_rate": self.sample_rate,
            "bit_depth": 16
        }
    
    def save_audio(self, audio: np.ndarray, filename: str, metadata: Dict[str, Any]) -> str:
        """
        Save audio to file with metadata
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert to 16-bit PCM
        audio_int16 = np.int16(audio * 32767)
        
        # Save audio file
        sf.write(filename, audio_int16, self.sample_rate, subtype='PCM_16')
        
        # Save metadata
        metadata_filename = filename.replace('.wav', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filename