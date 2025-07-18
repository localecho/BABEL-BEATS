#!/usr/bin/env python3
"""
Language processing module for BABEL-BEATS
Handles speech analysis, rhythm extraction, and tone detection
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
import base64
import io
import json
from dataclasses import dataclass
from scipy import signal
import logging

logger = logging.getLogger(__name__)


@dataclass
class LanguageFeatures:
    """Container for extracted language features"""
    rhythm: Dict[str, Any]
    tone: Dict[str, Any]
    pronunciation: Dict[str, Any]
    overall_score: float
    recommendations: List[str]


class LanguageProcessor:
    """
    Advanced language processing for speech analysis
    """
    
    def __init__(self):
        self.sample_rate = 22050
        self.frame_length = 2048
        self.hop_length = 512
        
        # Language-specific configurations
        self.language_configs = {
            "zh-CN": {
                "tonal": True,
                "tone_count": 4,
                "rhythm_pattern": "syllable-timed"
            },
            "en-US": {
                "tonal": False,
                "stress_based": True,
                "rhythm_pattern": "stress-timed"
            },
            "ja-JP": {
                "tonal": False,
                "pitch_accent": True,
                "rhythm_pattern": "mora-timed"
            },
            "es-ES": {
                "tonal": False,
                "syllable_based": True,
                "rhythm_pattern": "syllable-timed"
            }
        }
    
    def decode_audio(self, audio_base64: str) -> Tuple[np.ndarray, int]:
        """
        Decode base64 audio to numpy array
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_bytes)
            audio_data, sample_rate = sf.read(audio_buffer)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=self.sample_rate
                )
            
            return audio_data, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            raise ValueError(f"Failed to decode audio: {str(e)}")
    
    def analyze(self, audio_base64: str, language: str, text: Optional[str] = None) -> LanguageFeatures:
        """
        Perform comprehensive language analysis
        """
        # Decode audio
        audio, sr = self.decode_audio(audio_base64)
        
        # Extract features based on language
        language_config = self.language_configs.get(language, {})
        
        # Rhythm analysis
        rhythm_features = self.extract_rhythm_features(audio, language_config)
        
        # Tone analysis (for tonal languages)
        tone_features = self.extract_tone_features(audio, language_config) if language_config.get("tonal") else {}
        
        # Pronunciation analysis
        pronunciation_features = self.analyze_pronunciation(audio, language, text)
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(rhythm_features, tone_features, pronunciation_features)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            rhythm_features, tone_features, pronunciation_features, language
        )
        
        return LanguageFeatures(
            rhythm=rhythm_features,
            tone=tone_features,
            pronunciation=pronunciation_features,
            overall_score=overall_score,
            recommendations=recommendations
        )
    
    def extract_rhythm_features(self, audio: np.ndarray, config: Dict) -> Dict[str, Any]:
        """
        Extract rhythm and timing features
        """
        # Get tempo
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        
        # Get onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        
        # Detect onsets
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            units='time'
        )
        
        # Calculate inter-onset intervals
        ioi = np.diff(onsets) if len(onsets) > 1 else np.array([])
        
        # Rhythm consistency (coefficient of variation)
        rhythm_consistency = 1 - (np.std(ioi) / np.mean(ioi)) if len(ioi) > 0 else 0
        
        # Syllable rate estimation
        syllable_rate = len(onsets) / (len(audio) / self.sample_rate) if len(audio) > 0 else 0
        
        # Beat strength
        beat_strength = np.mean(onset_env[beats]) if len(beats) > 0 else 0
        
        return {
            "tempo": float(tempo),
            "beat_times": beats.tolist() if isinstance(beats, np.ndarray) else [],
            "onset_times": onsets.tolist(),
            "rhythm_consistency": float(rhythm_consistency),
            "syllable_rate": float(syllable_rate),
            "beat_strength": float(beat_strength),
            "inter_onset_intervals": ioi.tolist()
        }
    
    def extract_tone_features(self, audio: np.ndarray, config: Dict) -> Dict[str, Any]:
        """
        Extract tone features for tonal languages
        """
        # Extract pitch using YIN algorithm
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Remove unvoiced segments
        voiced_flag = f0 > 0
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) == 0:
            return {
                "pitch_contour": [],
                "tone_patterns": [],
                "tone_accuracy": 0.0
            }
        
        # Normalize pitch to semitones
        f0_semitones = 12 * np.log2(f0_voiced / np.median(f0_voiced))
        
        # Detect tone patterns (simplified)
        tone_patterns = self.detect_tone_patterns(f0_semitones, config)
        
        # Calculate tone accuracy (mock - would need reference)
        tone_accuracy = self.calculate_tone_accuracy(tone_patterns, config)
        
        return {
            "pitch_contour": f0_semitones.tolist(),
            "pitch_mean": float(np.mean(f0_voiced)),
            "pitch_std": float(np.std(f0_voiced)),
            "tone_patterns": tone_patterns,
            "tone_accuracy": tone_accuracy,
            "voiced_ratio": float(np.sum(voiced_flag) / len(f0))
        }
    
    def detect_tone_patterns(self, pitch_contour: np.ndarray, config: Dict) -> List[str]:
        """
        Detect tone patterns in pitch contour
        """
        if len(pitch_contour) < 10:
            return []
        
        patterns = []
        
        # Simple pattern detection based on pitch movement
        for i in range(0, len(pitch_contour) - 10, 5):
            segment = pitch_contour[i:i+10]
            trend = np.polyfit(range(len(segment)), segment, 1)[0]
            
            if trend > 0.5:
                patterns.append("rising")
            elif trend < -0.5:
                patterns.append("falling")
            elif np.std(segment) < 1:
                patterns.append("level")
            else:
                # Check for dipping pattern
                mid_point = len(segment) // 2
                if segment[mid_point] < segment[0] and segment[-1] > segment[mid_point]:
                    patterns.append("dipping")
                else:
                    patterns.append("complex")
        
        return patterns
    
    def calculate_tone_accuracy(self, patterns: List[str], config: Dict) -> float:
        """
        Calculate tone accuracy score
        """
        if not patterns:
            return 0.0
        
        # For tonal languages, check pattern distribution
        if config.get("tone_count"):
            # Expected distribution for Mandarin (simplified)
            expected_dist = {
                "level": 0.25,
                "rising": 0.25,
                "dipping": 0.25,
                "falling": 0.25
            }
            
            # Calculate actual distribution
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Compare distributions
            total_patterns = len(patterns)
            accuracy = 0
            for pattern, expected_ratio in expected_dist.items():
                actual_ratio = pattern_counts.get(pattern, 0) / total_patterns
                accuracy += 1 - abs(expected_ratio - actual_ratio)
            
            return accuracy / len(expected_dist)
        
        return 0.7  # Default for non-tonal languages
    
    def analyze_pronunciation(self, audio: np.ndarray, language: str, text: Optional[str]) -> Dict[str, Any]:
        """
        Analyze pronunciation quality
        """
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Calculate spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        
        # Voice quality metrics
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # Clarity score based on spectral features
        clarity_score = self.calculate_clarity_score(spectral_centroid, spectral_rolloff, zcr)
        
        # Mock phoneme accuracy (would require ASR in production)
        phoneme_accuracy = 0.75 + np.random.random() * 0.15
        
        # Identify problem sounds (mock)
        problem_sounds = self.identify_problem_sounds(language)
        
        return {
            "phoneme_accuracy": float(phoneme_accuracy),
            "clarity_score": float(clarity_score),
            "problem_sounds": problem_sounds,
            "spectral_features": {
                "centroid_mean": float(np.mean(spectral_centroid)),
                "centroid_std": float(np.std(spectral_centroid)),
                "rolloff_mean": float(np.mean(spectral_rolloff))
            },
            "voice_quality": {
                "zero_crossing_rate": float(np.mean(zcr)),
                "voiced_segments": float(np.sum(zcr < 0.1) / len(zcr[0]))
            }
        }
    
    def calculate_clarity_score(self, centroid: np.ndarray, rolloff: np.ndarray, zcr: np.ndarray) -> float:
        """
        Calculate speech clarity score
        """
        # Higher centroid = clearer speech
        centroid_score = np.clip(np.mean(centroid) / 4000, 0, 1)
        
        # Higher rolloff = more high-frequency content
        rolloff_score = np.clip(np.mean(rolloff) / 8000, 0, 1)
        
        # Lower ZCR variance = more stable speech
        zcr_score = 1 - np.clip(np.std(zcr), 0, 1)
        
        # Weighted average
        clarity_score = (centroid_score * 0.4 + rolloff_score * 0.3 + zcr_score * 0.3)
        
        return clarity_score
    
    def identify_problem_sounds(self, language: str) -> List[str]:
        """
        Identify problematic sounds for language learners
        """
        # Common problem sounds by target language
        problem_sounds_db = {
            "en-US": ["θ", "ð", "r", "l", "æ"],
            "zh-CN": ["zh", "ch", "sh", "r", "ü"],
            "ja-JP": ["r", "l", "ts", "fu"],
            "es-ES": ["rr", "j", "ll", "ñ"],
            "fr-FR": ["r", "u", "eu", "en", "on"],
            "de-DE": ["ü", "ö", "ch", "r"],
            "ko-KR": ["ㅡ", "ㅓ", "ㅂ/ㅍ", "ㄱ/ㅋ"]
        }
        
        return problem_sounds_db.get(language, [])[:3]  # Return top 3
    
    def calculate_overall_score(self, rhythm: Dict, tone: Dict, pronunciation: Dict) -> float:
        """
        Calculate overall language proficiency score
        """
        scores = []
        
        # Rhythm score
        if rhythm:
            rhythm_score = rhythm.get("rhythm_consistency", 0) * 0.7 + \
                          min(rhythm.get("beat_strength", 0), 1) * 0.3
            scores.append(rhythm_score)
        
        # Tone score (if applicable)
        if tone and tone.get("tone_accuracy"):
            scores.append(tone["tone_accuracy"])
        
        # Pronunciation score
        if pronunciation:
            pron_score = pronunciation.get("phoneme_accuracy", 0) * 0.6 + \
                        pronunciation.get("clarity_score", 0) * 0.4
            scores.append(pron_score)
        
        # Return weighted average
        return float(np.mean(scores)) if scores else 0.0
    
    def generate_recommendations(self, rhythm: Dict, tone: Dict, pronunciation: Dict, language: str) -> List[str]:
        """
        Generate personalized learning recommendations
        """
        recommendations = []
        
        # Rhythm recommendations
        if rhythm.get("rhythm_consistency", 1) < 0.7:
            recommendations.append("Practice speaking with a metronome to improve rhythm consistency")
        
        if rhythm.get("syllable_rate", 0) > 6:
            recommendations.append("Try to slow down your speech for better clarity")
        
        # Tone recommendations (for tonal languages)
        if tone and self.language_configs.get(language, {}).get("tonal"):
            if tone.get("tone_accuracy", 1) < 0.7:
                recommendations.append("Focus on tone practice, especially rising and falling tones")
        
        # Pronunciation recommendations
        if pronunciation:
            if pronunciation.get("clarity_score", 1) < 0.7:
                recommendations.append("Work on articulation and enunciation for clearer speech")
            
            problem_sounds = pronunciation.get("problem_sounds", [])
            if problem_sounds:
                recommendations.append(f"Practice these challenging sounds: {', '.join(problem_sounds[:2])}")
        
        # Language-specific recommendations
        if language == "zh-CN":
            recommendations.append("Practice tone pairs to improve tonal accuracy")
        elif language == "en-US":
            recommendations.append("Focus on stress patterns in multi-syllable words")
        
        return recommendations[:4]  # Return top 4 recommendations