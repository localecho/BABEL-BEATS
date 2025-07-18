#!/usr/bin/env python3
"""
Advanced Language Processing Module for BABEL-BEATS
Implements state-of-the-art speech recognition, phoneme analysis, and pronunciation assessment
"""

import numpy as np
import torch
import torchaudio
import whisper
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any, Union
import base64
import io
import json
import asyncio
from dataclasses import dataclass, field
from scipy import signal
import logging
import time
from pathlib import Path
import tempfile
import subprocess
import re

# Advanced audio processing
import parselmouth
import crepe
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
import epitran
import panphon
from g2p_en import G2p

# Deep learning
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier

# Audio features
from python_speech_features import mfcc, logfbank
import webrtcvad
from pyannote.audio import Pipeline as PyannotePipeline

# Metrics
from jiwer import wer, cer
import edit_distance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdvancedLanguageFeatures:
    """Enhanced container for language analysis results"""
    # Core features
    transcription: str
    language: str
    confidence: float
    
    # Phoneme analysis
    phonemes: List[str]
    phoneme_timestamps: List[Tuple[float, float]]
    phoneme_scores: List[float]
    
    # Prosody features
    pitch_contour: np.ndarray
    pitch_confidence: np.ndarray
    energy_contour: np.ndarray
    rhythm_metrics: Dict[str, float]
    
    # Pronunciation assessment
    pronunciation_score: float
    fluency_score: float
    completeness_score: float
    accent_score: float
    
    # Detailed feedback
    word_scores: List[Dict[str, Any]]
    problem_segments: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Advanced features
    emotion: Optional[str] = None
    speaking_rate: Optional[float] = None
    voice_quality: Optional[Dict[str, float]] = None
    comparison_data: Optional[Dict[str, Any]] = None


class AdvancedLanguageProcessor:
    """
    State-of-the-art language processor with multiple AI models
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize with GPU support if available"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Advanced Language Processor on {self.device}")
        
        # Sample rates
        self.sample_rate = 16000  # Standard for speech models
        
        # Initialize models lazily
        self._whisper_model = None
        self._wav2vec2_model = None
        self._wav2vec2_processor = None
        self._phoneme_model = None
        self._emotion_classifier = None
        self._g2p = None
        self._epitran = {}  # Language-specific epitran models
        self._panphon = panphon.PanPhon()
        
        # Voice activity detector
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        
        # Language configurations with phoneme sets
        self.language_configs = self._load_language_configs()
        
        # Cache for models
        self.model_cache = {}
        
    def _load_language_configs(self) -> Dict[str, Dict]:
        """Load comprehensive language configurations"""
        return {
            "en": {
                "name": "English",
                "whisper_code": "en",
                "wav2vec2_model": "facebook/wav2vec2-large-960h-lv60-self",
                "phoneme_backend": "espeak",
                "epitran_code": "eng-Latn",
                "tonal": False,
                "stress_timed": True,
                "phoneme_set": "arpabet"
            },
            "zh": {
                "name": "Chinese (Mandarin)",
                "whisper_code": "zh",
                "wav2vec2_model": "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt",
                "phoneme_backend": "espeak",
                "epitran_code": "cmn-Hans",
                "tonal": True,
                "tone_count": 5,
                "stress_timed": False,
                "phoneme_set": "pinyin"
            },
            "es": {
                "name": "Spanish",
                "whisper_code": "es",
                "wav2vec2_model": "facebook/wav2vec2-large-xlsr-53-spanish",
                "phoneme_backend": "espeak",
                "epitran_code": "spa-Latn",
                "tonal": False,
                "stress_timed": False,
                "phoneme_set": "ipa"
            },
            "fr": {
                "name": "French",
                "whisper_code": "fr",
                "wav2vec2_model": "facebook/wav2vec2-large-xlsr-53-french",
                "phoneme_backend": "espeak",
                "epitran_code": "fra-Latn",
                "tonal": False,
                "stress_timed": False,
                "phoneme_set": "ipa"
            },
            "de": {
                "name": "German",
                "whisper_code": "de",
                "wav2vec2_model": "facebook/wav2vec2-large-xlsr-53-german",
                "phoneme_backend": "espeak",
                "epitran_code": "deu-Latn",
                "tonal": False,
                "stress_timed": True,
                "phoneme_set": "ipa"
            },
            "ja": {
                "name": "Japanese",
                "whisper_code": "ja",
                "wav2vec2_model": "ydshieh/wav2vec2-large-xlsr-53-japanese",
                "phoneme_backend": "espeak",
                "epitran_code": "jpn-Hira",
                "tonal": False,
                "pitch_accent": True,
                "mora_timed": True,
                "phoneme_set": "ipa"
            },
            "ko": {
                "name": "Korean",
                "whisper_code": "ko",
                "wav2vec2_model": "kresnik/wav2vec2-large-xlsr-korean",
                "phoneme_backend": "espeak",
                "epitran_code": "kor-Hang",
                "tonal": False,
                "stress_timed": False,
                "phoneme_set": "ipa"
            }
        }
    
    @property
    def whisper_model(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            logger.info("Loading Whisper model...")
            self._whisper_model = whisper.load_model("base", device=self.device)
        return self._whisper_model
    
    @property
    def wav2vec2_model(self):
        """Lazy load Wav2Vec2 model"""
        if self._wav2vec2_model is None:
            logger.info("Loading Wav2Vec2 model...")
            model_name = "facebook/wav2vec2-large-960h-lv60-self"
            self._wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
            self._wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        return self._wav2vec2_model, self._wav2vec2_processor
    
    @property
    def g2p(self):
        """Lazy load G2P model"""
        if self._g2p is None:
            logger.info("Loading G2P model...")
            self._g2p = G2p()
        return self._g2p
    
    def get_epitran(self, language: str):
        """Get language-specific epitran model"""
        if language not in self._epitran:
            lang_config = self.language_configs.get(language, {})
            epitran_code = lang_config.get("epitran_code", "eng-Latn")
            self._epitran[language] = epitran.Epitran(epitran_code)
        return self._epitran[language]
    
    async def analyze_comprehensive(
        self,
        audio_input: Union[str, np.ndarray],
        language: Optional[str] = None,
        reference_text: Optional[str] = None,
        native_speaker_audio: Optional[Union[str, np.ndarray]] = None
    ) -> AdvancedLanguageFeatures:
        """
        Perform comprehensive language analysis with all advanced features
        """
        start_time = time.time()
        
        # Load and preprocess audio
        audio, sr = self._load_audio(audio_input)
        
        # Detect language if not provided
        if language is None:
            language = await self._detect_language(audio)
        
        # Get language configuration
        lang_config = self.language_configs.get(language, self.language_configs["en"])
        
        # Run all analyses in parallel
        tasks = [
            self._transcribe_with_whisper(audio, language),
            self._extract_phonemes(audio, language, reference_text),
            self._analyze_prosody(audio, lang_config),
            self._assess_pronunciation(audio, language, reference_text),
            self._analyze_voice_quality(audio),
            self._detect_emotion(audio)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Unpack results
        transcription_data = results[0]
        phoneme_data = results[1]
        prosody_data = results[2]
        pronunciation_data = results[3]
        voice_quality = results[4]
        emotion = results[5]
        
        # Compare with native speaker if provided
        comparison_data = None
        if native_speaker_audio is not None:
            comparison_data = await self._compare_with_native(
                audio, native_speaker_audio, language
            )
        
        # Generate advanced recommendations
        recommendations = self._generate_advanced_recommendations(
            transcription_data,
            phoneme_data,
            prosody_data,
            pronunciation_data,
            lang_config
        )
        
        # Calculate speaking rate
        speaking_rate = self._calculate_speaking_rate(
            phoneme_data["phonemes"],
            len(audio) / sr
        )
        
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        
        return AdvancedLanguageFeatures(
            transcription=transcription_data["text"],
            language=language,
            confidence=transcription_data["confidence"],
            phonemes=phoneme_data["phonemes"],
            phoneme_timestamps=phoneme_data["timestamps"],
            phoneme_scores=phoneme_data["scores"],
            pitch_contour=prosody_data["pitch_contour"],
            pitch_confidence=prosody_data["pitch_confidence"],
            energy_contour=prosody_data["energy_contour"],
            rhythm_metrics=prosody_data["rhythm_metrics"],
            pronunciation_score=pronunciation_data["overall_score"],
            fluency_score=pronunciation_data["fluency_score"],
            completeness_score=pronunciation_data["completeness_score"],
            accent_score=pronunciation_data["accent_score"],
            word_scores=pronunciation_data["word_scores"],
            problem_segments=pronunciation_data["problem_segments"],
            recommendations=recommendations,
            emotion=emotion,
            speaking_rate=speaking_rate,
            voice_quality=voice_quality,
            comparison_data=comparison_data
        )
    
    def _load_audio(self, audio_input: Union[str, np.ndarray]) -> Tuple[np.ndarray, int]:
        """Load audio from base64 string or numpy array"""
        if isinstance(audio_input, str):
            # Decode base64
            audio_bytes = base64.b64decode(audio_input)
            audio_buffer = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_buffer)
        else:
            audio = audio_input
            sr = self.sample_rate
        
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to 16kHz for speech models
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio, self.sample_rate
    
    async def _detect_language(self, audio: np.ndarray) -> str:
        """Detect language using Whisper"""
        # Use first 30 seconds for language detection
        sample_audio = audio[:30 * self.sample_rate]
        
        # Detect language
        result = self.whisper_model.detect_language(sample_audio)
        detected_lang = max(result, key=result.get)
        
        # Map to our language codes
        lang_mapping = {
            "en": "en", "zh": "zh", "es": "es", "fr": "fr",
            "de": "de", "ja": "ja", "ko": "ko"
        }
        
        return lang_mapping.get(detected_lang, "en")
    
    async def _transcribe_with_whisper(
        self, 
        audio: np.ndarray, 
        language: str
    ) -> Dict[str, Any]:
        """Transcribe audio using Whisper with word-level timestamps"""
        lang_code = self.language_configs.get(language, {}).get("whisper_code", "en")
        
        # Transcribe with word timestamps
        result = self.whisper_model.transcribe(
            audio,
            language=lang_code,
            word_timestamps=True,
            task="transcribe"
        )
        
        # Extract word-level information
        words = []
        if "words" in result:
            for word_info in result["words"]:
                words.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "probability": word_info.get("probability", 1.0)
                })
        
        return {
            "text": result["text"],
            "language": result.get("language", language),
            "words": words,
            "confidence": np.mean([w["probability"] for w in words]) if words else 0.0
        }
    
    async def _extract_phonemes(
        self,
        audio: np.ndarray,
        language: str,
        reference_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract phonemes with forced alignment"""
        # Get transcription if no reference provided
        if reference_text is None:
            transcription = await self._transcribe_with_whisper(audio, language)
            reference_text = transcription["text"]
        
        # Convert text to phonemes
        lang_config = self.language_configs.get(language, {})
        backend = lang_config.get("phoneme_backend", "espeak")
        
        try:
            # Get phonemes using phonemizer
            phonemes = phonemize(
                reference_text,
                language=self._get_phonemizer_language(language),
                backend=backend,
                strip=True,
                preserve_punctuation=False,
                with_stress=True,
                language_switch='remove-flags'
            )
            
            # Split into individual phonemes
            phoneme_list = phonemes.strip().split()
            
            # Perform forced alignment to get timestamps
            alignment_data = await self._force_align_phonemes(
                audio, phoneme_list, language
            )
            
            # Calculate phoneme-level scores
            phoneme_scores = self._calculate_phoneme_scores(
                audio, alignment_data, language
            )
            
            return {
                "phonemes": phoneme_list,
                "timestamps": alignment_data["timestamps"],
                "scores": phoneme_scores,
                "alignment_score": alignment_data["confidence"]
            }
            
        except Exception as e:
            logger.error(f"Phoneme extraction failed: {e}")
            # Fallback to basic phoneme extraction
            return self._fallback_phoneme_extraction(reference_text, language)
    
    def _get_phonemizer_language(self, language: str) -> str:
        """Map language code to phonemizer language"""
        mapping = {
            "en": "en-us",
            "zh": "cmn",
            "es": "es",
            "fr": "fr-fr",
            "de": "de",
            "ja": "ja",
            "ko": "ko"
        }
        return mapping.get(language, "en-us")
    
    async def _force_align_phonemes(
        self,
        audio: np.ndarray,
        phonemes: List[str],
        language: str
    ) -> Dict[str, Any]:
        """Perform forced alignment using wav2vec2"""
        model, processor = self.wav2vec2_model
        
        # Process audio
        inputs = processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode to get alignment
        # This is simplified - in production, use proper CTC alignment
        timestamps = []
        frame_duration = len(audio) / logits.shape[1]
        
        for i, phoneme in enumerate(phonemes):
            start_time = i * frame_duration * len(phonemes) / logits.shape[1]
            end_time = (i + 1) * frame_duration * len(phonemes) / logits.shape[1]
            timestamps.append((start_time, end_time))
        
        return {
            "timestamps": timestamps,
            "confidence": 0.85  # Placeholder
        }
    
    def _calculate_phoneme_scores(
        self,
        audio: np.ndarray,
        alignment_data: Dict[str, Any],
        language: str
    ) -> List[float]:
        """Calculate pronunciation scores for each phoneme"""
        scores = []
        
        for start, end in alignment_data["timestamps"]:
            # Extract phoneme segment
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            segment = audio[start_sample:end_sample]
            
            if len(segment) > 0:
                # Calculate acoustic features
                mfcc_features = mfcc(segment, self.sample_rate)
                
                # Simple scoring based on energy and spectral properties
                energy = np.mean(np.square(segment))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                    y=segment, sr=self.sample_rate
                ))
                
                # Normalize and combine scores
                energy_score = np.clip(energy * 10, 0, 1)
                spectral_score = np.clip(spectral_centroid / 4000, 0, 1)
                
                score = 0.6 * energy_score + 0.4 * spectral_score
                scores.append(float(score))
            else:
                scores.append(0.0)
        
        return scores
    
    async def _analyze_prosody(
        self,
        audio: np.ndarray,
        lang_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze prosodic features including pitch, rhythm, and stress"""
        # Extract pitch using CREPE
        time_stamps, frequency, confidence, activation = crepe.predict(
            audio,
            self.sample_rate,
            viterbi=True,
            model_capacity="full"
        )
        
        # Filter out unvoiced segments
        voiced_mask = confidence > 0.5
        voiced_freq = frequency[voiced_mask]
        voiced_time = time_stamps[voiced_mask]
        
        # Calculate energy contour
        frame_length = 2048
        hop_length = 512
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Calculate rhythm metrics
        rhythm_metrics = self._calculate_rhythm_metrics(
            audio, lang_config, voiced_time, voiced_freq
        )
        
        # Analyze stress patterns for stress-timed languages
        if lang_config.get("stress_timed", False):
            stress_pattern = self._analyze_stress_pattern(
                audio, energy, voiced_freq
            )
            rhythm_metrics["stress_pattern"] = stress_pattern
        
        # Analyze tone patterns for tonal languages
        if lang_config.get("tonal", False):
            tone_analysis = self._analyze_tones(
                voiced_freq, voiced_time, lang_config
            )
            rhythm_metrics.update(tone_analysis)
        
        return {
            "pitch_contour": frequency,
            "pitch_confidence": confidence,
            "energy_contour": energy,
            "rhythm_metrics": rhythm_metrics,
            "voiced_segments": voiced_mask
        }
    
    def _calculate_rhythm_metrics(
        self,
        audio: np.ndarray,
        lang_config: Dict[str, Any],
        pitch_times: np.ndarray,
        pitch_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive rhythm metrics"""
        # Use Praat for advanced rhythm analysis
        sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
        
        # Extract syllable nuclei
        syllable_nuclei = self._extract_syllable_nuclei(sound)
        
        # Calculate various rhythm metrics
        metrics = {}
        
        if len(syllable_nuclei) > 1:
            # Inter-syllable intervals
            intervals = np.diff(syllable_nuclei)
            
            # nPVI (normalized Pairwise Variability Index)
            if len(intervals) > 1:
                npvi = self._calculate_npvi(intervals)
                metrics["nPVI"] = float(npvi)
            
            # rPVI (raw Pairwise Variability Index)
            rpvi = self._calculate_rpvi(intervals)
            metrics["rPVI"] = float(rpvi)
            
            # %V (percentage of vocalic intervals)
            metrics["percent_V"] = self._calculate_percent_v(audio, sound)
            
            # ΔC (standard deviation of consonantal intervals)
            metrics["delta_C"] = float(np.std(intervals))
            
            # Syllable rate
            duration = len(audio) / self.sample_rate
            metrics["syllable_rate"] = len(syllable_nuclei) / duration
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        metrics["tempo"] = float(tempo)
        
        # Speech rate variability
        if len(pitch_times) > 10:
            local_rates = []
            window = 10
            for i in range(len(pitch_times) - window):
                local_duration = pitch_times[i + window] - pitch_times[i]
                local_rate = window / local_duration if local_duration > 0 else 0
                local_rates.append(local_rate)
            
            if local_rates:
                metrics["speech_rate_variability"] = float(np.std(local_rates))
        
        return metrics
    
    def _extract_syllable_nuclei(self, sound: parselmouth.Sound) -> np.ndarray:
        """Extract syllable nuclei using Praat algorithm"""
        # This is a simplified version - full implementation would use
        # Praat's syllable nuclei detection algorithm
        intensity = sound.to_intensity()
        
        # Find peaks in intensity (simplified syllable detection)
        intensity_values = intensity.values[0]
        peaks, _ = signal.find_peaks(
            intensity_values,
            height=np.max(intensity_values) * 0.3,
            distance=int(0.1 * intensity.time_step)  # Min 100ms between syllables
        )
        
        # Convert to time stamps
        time_stamps = peaks * intensity.time_step + intensity.start_time
        
        return time_stamps
    
    def _calculate_npvi(self, intervals: np.ndarray) -> float:
        """Calculate normalized Pairwise Variability Index"""
        if len(intervals) < 2:
            return 0.0
        
        npvi_sum = 0
        for i in range(len(intervals) - 1):
            diff = abs(intervals[i] - intervals[i + 1])
            avg = (intervals[i] + intervals[i + 1]) / 2
            if avg > 0:
                npvi_sum += diff / avg
        
        return 100 * npvi_sum / (len(intervals) - 1)
    
    def _calculate_rpvi(self, intervals: np.ndarray) -> float:
        """Calculate raw Pairwise Variability Index"""
        if len(intervals) < 2:
            return 0.0
        
        rpvi_sum = 0
        for i in range(len(intervals) - 1):
            rpvi_sum += abs(intervals[i] - intervals[i + 1])
        
        return rpvi_sum / (len(intervals) - 1)
    
    def _calculate_percent_v(self, audio: np.ndarray, sound: parselmouth.Sound) -> float:
        """Calculate percentage of vocalic intervals"""
        # Use VAD to detect voiced segments
        frame_duration = 30  # ms
        frame_length = int(self.sample_rate * frame_duration / 1000)
        
        voiced_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length]
            # Convert to 16-bit PCM for VAD
            frame_int16 = (frame * 32767).astype(np.int16)
            is_speech = self.vad.is_speech(frame_int16.tobytes(), self.sample_rate)
            
            if is_speech:
                voiced_frames += 1
            total_frames += 1
        
        return (voiced_frames / total_frames * 100) if total_frames > 0 else 0
    
    def _analyze_stress_pattern(
        self,
        audio: np.ndarray,
        energy: np.ndarray,
        pitch: np.ndarray
    ) -> List[int]:
        """Analyze stress patterns in speech"""
        # Find peaks in energy that correspond to stressed syllables
        peaks, properties = signal.find_peaks(
            energy,
            height=np.mean(energy),
            prominence=np.std(energy) * 0.5
        )
        
        # Create binary stress pattern
        stress_pattern = []
        for i in range(len(energy)):
            if i in peaks:
                stress_pattern.append(1)  # Stressed
            else:
                stress_pattern.append(0)  # Unstressed
        
        return stress_pattern
    
    def _analyze_tones(
        self,
        pitch_contour: np.ndarray,
        time_stamps: np.ndarray,
        lang_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze tones for tonal languages"""
        tone_count = lang_config.get("tone_count", 4)
        
        # Segment pitch contour into tone-bearing units
        tone_segments = self._segment_tones(pitch_contour, time_stamps)
        
        # Classify each segment
        tone_classifications = []
        tone_scores = []
        
        for segment in tone_segments:
            if len(segment) > 0:
                # Normalize pitch
                normalized = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                
                # Simple tone classification based on contour shape
                tone_type, confidence = self._classify_tone_contour(
                    normalized, tone_count
                )
                
                tone_classifications.append(tone_type)
                tone_scores.append(confidence)
        
        return {
            "tone_sequence": tone_classifications,
            "tone_confidence": tone_scores,
            "tone_accuracy": np.mean(tone_scores) if tone_scores else 0.0
        }
    
    def _segment_tones(
        self,
        pitch_contour: np.ndarray,
        time_stamps: np.ndarray
    ) -> List[np.ndarray]:
        """Segment pitch contour into tone-bearing units"""
        # Simple segmentation based on pitch stability
        segments = []
        current_segment = []
        
        for i in range(1, len(pitch_contour)):
            pitch_change = abs(pitch_contour[i] - pitch_contour[i-1])
            
            if pitch_change < 20:  # Stable pitch
                current_segment.append(pitch_contour[i])
            else:
                if len(current_segment) > 5:
                    segments.append(np.array(current_segment))
                current_segment = [pitch_contour[i]]
        
        if len(current_segment) > 5:
            segments.append(np.array(current_segment))
        
        return segments
    
    def _classify_tone_contour(
        self,
        contour: np.ndarray,
        tone_count: int
    ) -> Tuple[int, float]:
        """Classify tone based on pitch contour shape"""
        if len(contour) < 3:
            return 0, 0.0
        
        # Calculate contour features
        start_pitch = np.mean(contour[:3])
        end_pitch = np.mean(contour[-3:])
        mid_pitch = np.mean(contour[len(contour)//2-1:len(contour)//2+2])
        
        # Simple classification for Mandarin tones
        if tone_count == 4 or tone_count == 5:
            # Tone 1: High level
            if abs(end_pitch - start_pitch) < 0.5 and start_pitch > 0.5:
                return 1, 0.8
            
            # Tone 2: Rising
            elif end_pitch - start_pitch > 1.0:
                return 2, 0.7
            
            # Tone 3: Dipping
            elif mid_pitch < start_pitch and mid_pitch < end_pitch:
                return 3, 0.7
            
            # Tone 4: Falling
            elif start_pitch - end_pitch > 1.0:
                return 4, 0.7
            
            # Tone 0: Neutral (if 5 tones)
            else:
                return 0, 0.5
        
        return 0, 0.0
    
    async def _assess_pronunciation(
        self,
        audio: np.ndarray,
        language: str,
        reference_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive pronunciation assessment"""
        # Get phoneme alignment
        phoneme_data = await self._extract_phonemes(audio, language, reference_text)
        
        # Calculate GOP (Goodness of Pronunciation) scores
        gop_scores = self._calculate_gop_scores(
            audio, phoneme_data, language
        )
        
        # Assess fluency
        fluency_score = self._assess_fluency(
            audio, phoneme_data, language
        )
        
        # Assess completeness
        completeness_score = 1.0  # Placeholder
        if reference_text:
            completeness_score = self._assess_completeness(
                phoneme_data["phonemes"], reference_text, language
            )
        
        # Assess accent
        accent_score = await self._assess_accent(audio, language)
        
        # Calculate word-level scores
        word_scores = self._calculate_word_scores(
            phoneme_data, gop_scores
        )
        
        # Identify problem segments
        problem_segments = self._identify_problem_segments(
            phoneme_data, gop_scores, threshold=0.6
        )
        
        # Overall pronunciation score
        overall_score = np.mean([
            np.mean(gop_scores),
            fluency_score,
            completeness_score,
            accent_score
        ])
        
        return {
            "overall_score": float(overall_score),
            "fluency_score": float(fluency_score),
            "completeness_score": float(completeness_score),
            "accent_score": float(accent_score),
            "gop_scores": gop_scores,
            "word_scores": word_scores,
            "problem_segments": problem_segments
        }
    
    def _calculate_gop_scores(
        self,
        audio: np.ndarray,
        phoneme_data: Dict[str, Any],
        language: str
    ) -> List[float]:
        """Calculate Goodness of Pronunciation scores"""
        scores = []
        
        for i, (start, end) in enumerate(phoneme_data["timestamps"]):
            # Extract segment
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            segment = audio[start_sample:end_sample]
            
            if len(segment) > 0:
                # Extract acoustic features
                mfcc_features = mfcc(segment, self.sample_rate, nfcc=13)
                
                # Simple GOP score based on acoustic model confidence
                # In production, this would use a trained acoustic model
                score = phoneme_data["scores"][i] if i < len(phoneme_data["scores"]) else 0.5
                
                # Adjust based on segment quality
                if len(segment) < 0.01 * self.sample_rate:  # Very short segment
                    score *= 0.8
                
                scores.append(float(score))
            else:
                scores.append(0.0)
        
        return scores
    
    def _assess_fluency(
        self,
        audio: np.ndarray,
        phoneme_data: Dict[str, Any],
        language: str
    ) -> float:
        """Assess speech fluency"""
        # Calculate pause statistics
        timestamps = phoneme_data["timestamps"]
        pauses = []
        
        for i in range(1, len(timestamps)):
            gap = timestamps[i][0] - timestamps[i-1][1]
            if gap > 0.1:  # Pause longer than 100ms
                pauses.append(gap)
        
        # Fluency metrics
        total_duration = len(audio) / self.sample_rate
        speaking_duration = sum(end - start for start, end in timestamps)
        pause_duration = sum(pauses)
        
        # Calculate fluency score
        speaking_ratio = speaking_duration / total_duration if total_duration > 0 else 0
        pause_ratio = pause_duration / total_duration if total_duration > 0 else 0
        
        # Penalize for too many or too long pauses
        pause_penalty = 1.0
        if len(pauses) > 0:
            avg_pause = np.mean(pauses)
            if avg_pause > 0.5:  # Long pauses
                pause_penalty *= 0.8
            if len(pauses) / (len(timestamps) + 1) > 0.3:  # Too many pauses
                pause_penalty *= 0.9
        
        fluency_score = speaking_ratio * pause_penalty
        
        return np.clip(fluency_score, 0, 1)
    
    def _assess_completeness(
        self,
        phonemes: List[str],
        reference_text: str,
        language: str
    ) -> float:
        """Assess how complete the pronunciation is"""
        # Convert reference text to expected phonemes
        expected_phonemes = self._text_to_phonemes(reference_text, language)
        
        # Calculate edit distance
        distance = edit_distance.SequenceMatcher(
            a=phonemes,
            b=expected_phonemes
        ).distance()
        
        # Normalize by length
        max_length = max(len(phonemes), len(expected_phonemes))
        if max_length == 0:
            return 1.0
        
        completeness = 1.0 - (distance / max_length)
        
        return np.clip(completeness, 0, 1)
    
    def _text_to_phonemes(self, text: str, language: str) -> List[str]:
        """Convert text to phonemes for comparison"""
        try:
            lang_config = self.language_configs.get(language, {})
            backend = lang_config.get("phoneme_backend", "espeak")
            
            phonemes = phonemize(
                text,
                language=self._get_phonemizer_language(language),
                backend=backend,
                strip=True,
                preserve_punctuation=False
            )
            
            return phonemes.strip().split()
        except:
            # Fallback
            return text.split()
    
    async def _assess_accent(self, audio: np.ndarray, language: str) -> float:
        """Assess accent/native-likeness"""
        # This would use a trained accent classifier
        # For now, return a placeholder
        return 0.75
    
    def _calculate_word_scores(
        self,
        phoneme_data: Dict[str, Any],
        gop_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Calculate pronunciation scores at word level"""
        # This is simplified - would need word boundaries from forced alignment
        word_scores = []
        
        # Group phonemes into words (simplified)
        phonemes_per_word = 5  # Rough estimate
        
        for i in range(0, len(gop_scores), phonemes_per_word):
            word_phoneme_scores = gop_scores[i:i+phonemes_per_word]
            if word_phoneme_scores:
                word_score = np.mean(word_phoneme_scores)
                word_scores.append({
                    "word_index": len(word_scores),
                    "score": float(word_score),
                    "phoneme_indices": list(range(i, min(i+phonemes_per_word, len(gop_scores))))
                })
        
        return word_scores
    
    def _identify_problem_segments(
        self,
        phoneme_data: Dict[str, Any],
        gop_scores: List[float],
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Identify segments that need improvement"""
        problem_segments = []
        
        for i, score in enumerate(gop_scores):
            if score < threshold and i < len(phoneme_data["phonemes"]):
                start, end = phoneme_data["timestamps"][i]
                
                problem_segments.append({
                    "phoneme": phoneme_data["phonemes"][i],
                    "index": i,
                    "score": float(score),
                    "start_time": float(start),
                    "end_time": float(end),
                    "severity": "high" if score < 0.4 else "medium"
                })
        
        return problem_segments
    
    async def _analyze_voice_quality(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze voice quality metrics"""
        # Calculate various voice quality measures
        
        # Jitter (pitch variation)
        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=self.sample_rate)
        voiced = f0 > 0
        if np.sum(voiced) > 10:
            f0_voiced = f0[voiced]
            periods = 1 / f0_voiced
            if len(periods) > 1:
                period_diffs = np.abs(np.diff(periods))
                jitter = np.mean(period_diffs) / np.mean(periods)
            else:
                jitter = 0
        else:
            jitter = 0
        
        # Shimmer (amplitude variation)
        amplitude = np.abs(audio)
        peaks, _ = signal.find_peaks(amplitude, distance=int(0.01 * self.sample_rate))
        if len(peaks) > 1:
            peak_values = amplitude[peaks]
            amp_diffs = np.abs(np.diff(peak_values))
            shimmer = np.mean(amp_diffs) / np.mean(peak_values)
        else:
            shimmer = 0
        
        # Harmonic-to-Noise Ratio (HNR)
        hnr = self._calculate_hnr(audio)
        
        # Voice breaks
        voice_breaks = self._detect_voice_breaks(f0)
        
        return {
            "jitter": float(jitter),
            "shimmer": float(shimmer),
            "hnr": float(hnr),
            "voice_breaks": float(voice_breaks),
            "overall_quality": float(1.0 - (jitter + shimmer + voice_breaks) / 3)
        }
    
    def _calculate_hnr(self, audio: np.ndarray) -> float:
        """Calculate Harmonic-to-Noise Ratio"""
        # Simplified HNR calculation
        # In production, use proper HNR algorithm
        
        # Autocorrelation
        correlation = np.correlate(audio, audio, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # Find first peak after lag 0
        min_period = int(0.002 * self.sample_rate)  # 2ms
        max_period = int(0.02 * self.sample_rate)   # 20ms
        
        if max_period < len(correlation):
            peak_lag = np.argmax(correlation[min_period:max_period]) + min_period
            
            if peak_lag > 0:
                harmonic_energy = correlation[peak_lag]
                total_energy = correlation[0]
                
                if total_energy > 0:
                    hnr = 10 * np.log10(harmonic_energy / (total_energy - harmonic_energy + 1e-10))
                    return np.clip(hnr, 0, 40)
        
        return 0
    
    def _detect_voice_breaks(self, f0: np.ndarray) -> float:
        """Detect voice breaks in pitch contour"""
        voiced = f0 > 0
        
        # Find transitions from voiced to unvoiced
        breaks = 0
        for i in range(1, len(voiced)):
            if voiced[i-1] and not voiced[i]:
                breaks += 1
        
        # Normalize by duration
        duration = len(f0) / 100  # Assuming 100Hz pitch tracking
        breaks_per_second = breaks / duration if duration > 0 else 0
        
        return breaks_per_second
    
    async def _detect_emotion(self, audio: np.ndarray) -> Optional[str]:
        """Detect emotion from speech"""
        # This would use a trained emotion classifier
        # For now, return placeholder
        return "neutral"
    
    async def _compare_with_native(
        self,
        learner_audio: np.ndarray,
        native_audio: Union[str, np.ndarray],
        language: str
    ) -> Dict[str, Any]:
        """Compare learner's speech with native speaker"""
        # Load native speaker audio
        native_audio, _ = self._load_audio(native_audio)
        
        # Extract features for both
        learner_features = await self._extract_comparison_features(learner_audio, language)
        native_features = await self._extract_comparison_features(native_audio, language)
        
        # Compare features
        comparison = {
            "pitch_similarity": self._compare_pitch_contours(
                learner_features["pitch"],
                native_features["pitch"]
            ),
            "rhythm_similarity": self._compare_rhythm(
                learner_features["rhythm"],
                native_features["rhythm"]
            ),
            "spectral_similarity": self._compare_spectral_features(
                learner_features["spectral"],
                native_features["spectral"]
            ),
            "duration_ratio": len(learner_audio) / len(native_audio)
        }
        
        # Overall similarity score
        comparison["overall_similarity"] = np.mean([
            comparison["pitch_similarity"],
            comparison["rhythm_similarity"],
            comparison["spectral_similarity"]
        ])
        
        return comparison
    
    async def _extract_comparison_features(
        self,
        audio: np.ndarray,
        language: str
    ) -> Dict[str, Any]:
        """Extract features for comparison"""
        # Pitch features
        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=self.sample_rate)
        
        # Rhythm features
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        
        # Spectral features
        mfcc_features = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        return {
            "pitch": f0,
            "rhythm": {
                "tempo": tempo,
                "onset_strength": onset_env
            },
            "spectral": mfcc_features
        }
    
    def _compare_pitch_contours(self, pitch1: np.ndarray, pitch2: np.ndarray) -> float:
        """Compare two pitch contours"""
        # Dynamic Time Warping for pitch comparison
        from scipy.spatial.distance import euclidean
        from fastdtw import fastdtw
        
        # Remove unvoiced segments
        voiced1 = pitch1[pitch1 > 0]
        voiced2 = pitch2[pitch2 > 0]
        
        if len(voiced1) == 0 or len(voiced2) == 0:
            return 0.0
        
        # Normalize pitch to semitones
        semi1 = 12 * np.log2(voiced1 / np.median(voiced1))
        semi2 = 12 * np.log2(voiced2 / np.median(voiced2))
        
        # DTW distance
        distance, _ = fastdtw(semi1, semi2, dist=euclidean)
        
        # Normalize and convert to similarity
        normalized_distance = distance / max(len(semi1), len(semi2))
        similarity = np.exp(-normalized_distance / 10)  # Exponential decay
        
        return float(similarity)
    
    def _compare_rhythm(self, rhythm1: Dict, rhythm2: Dict) -> float:
        """Compare rhythm features"""
        # Tempo similarity
        tempo_diff = abs(rhythm1["tempo"] - rhythm2["tempo"])
        tempo_similarity = np.exp(-tempo_diff / 20)
        
        # Onset pattern similarity
        onset1 = rhythm1["onset_strength"]
        onset2 = rhythm2["onset_strength"]
        
        # Normalize lengths
        min_len = min(len(onset1), len(onset2))
        onset1_norm = onset1[:min_len]
        onset2_norm = onset2[:min_len]
        
        # Correlation
        if len(onset1_norm) > 0:
            correlation = np.corrcoef(onset1_norm, onset2_norm)[0, 1]
            onset_similarity = (correlation + 1) / 2  # Map to [0, 1]
        else:
            onset_similarity = 0
        
        return float(0.5 * tempo_similarity + 0.5 * onset_similarity)
    
    def _compare_spectral_features(self, mfcc1: np.ndarray, mfcc2: np.ndarray) -> float:
        """Compare spectral features"""
        # Average MFCC vectors
        mean_mfcc1 = np.mean(mfcc1, axis=1)
        mean_mfcc2 = np.mean(mfcc2, axis=1)
        
        # Cosine similarity
        similarity = np.dot(mean_mfcc1, mean_mfcc2) / (
            np.linalg.norm(mean_mfcc1) * np.linalg.norm(mean_mfcc2) + 1e-8
        )
        
        return float((similarity + 1) / 2)  # Map to [0, 1]
    
    def _calculate_speaking_rate(
        self,
        phonemes: List[str],
        duration: float
    ) -> float:
        """Calculate speaking rate in phonemes per second"""
        if duration > 0:
            return len(phonemes) / duration
        return 0
    
    def _generate_advanced_recommendations(
        self,
        transcription_data: Dict[str, Any],
        phoneme_data: Dict[str, Any],
        prosody_data: Dict[str, Any],
        pronunciation_data: Dict[str, Any],
        lang_config: Dict[str, Any]
    ) -> List[str]:
        """Generate detailed, actionable recommendations"""
        recommendations = []
        
        # Pronunciation recommendations
        if pronunciation_data["overall_score"] < 0.7:
            problem_phonemes = [
                seg["phoneme"] for seg in pronunciation_data["problem_segments"][:3]
            ]
            if problem_phonemes:
                recommendations.append(
                    f"Focus on improving these sounds: {', '.join(problem_phonemes)}. "
                    f"Practice minimal pairs and use tongue twisters."
                )
        
        # Fluency recommendations
        if pronunciation_data["fluency_score"] < 0.7:
            recommendations.append(
                "Work on fluency by practicing shadowing exercises with native speakers. "
                "Try to reduce pauses between words."
            )
        
        # Prosody recommendations
        rhythm_metrics = prosody_data["rhythm_metrics"]
        
        if lang_config.get("stress_timed") and rhythm_metrics.get("nPVI", 0) < 40:
            recommendations.append(
                "Your rhythm is too syllable-timed. Practice stress-timing by emphasizing "
                "content words and reducing function words."
            )
        
        if lang_config.get("tonal") and rhythm_metrics.get("tone_accuracy", 0) < 0.7:
            recommendations.append(
                "Practice tone patterns using tone pair drills. Focus on the contrast "
                "between rising and falling tones."
            )
        
        # Speaking rate recommendations
        if hasattr(self, '_last_speaking_rate'):
            if self._last_speaking_rate < 2.5:  # Too slow
                recommendations.append(
                    "Try to increase your speaking speed to sound more natural. "
                    "Practice with gradually faster recordings."
                )
            elif self._last_speaking_rate > 5.0:  # Too fast
                recommendations.append(
                    "Slow down your speech for better clarity. "
                    "Focus on articulating each sound clearly."
                )
        
        # Voice quality recommendations
        if hasattr(self, '_last_voice_quality'):
            vq = self._last_voice_quality
            if vq.get("jitter", 0) > 0.02:
                recommendations.append(
                    "Your voice shows irregularity. Practice breathing exercises "
                    "and maintain steady airflow while speaking."
                )
        
        # Limit to top 5 recommendations
        return recommendations[:5]
    
    def _fallback_phoneme_extraction(
        self,
        text: str,
        language: str
    ) -> Dict[str, Any]:
        """Fallback method for phoneme extraction"""
        # Simple character-based fallback
        phonemes = list(text.replace(" ", " | "))
        
        # Create dummy timestamps
        total_duration = len(text) * 0.1  # Assume 0.1s per character
        timestamps = []
        current_time = 0
        
        for phoneme in phonemes:
            duration = 0.1 if phoneme != "|" else 0.05
            timestamps.append((current_time, current_time + duration))
            current_time += duration
        
        # Create dummy scores
        scores = [0.7] * len(phonemes)
        
        return {
            "phonemes": phonemes,
            "timestamps": timestamps,
            "scores": scores,
            "alignment_score": 0.5
        }


class PhonemeAligner:
    """Specialized class for phoneme-level forced alignment"""
    
    def __init__(self, language: str):
        self.language = language
        self.aligner = None  # Would initialize MFA or similar
        
    async def align(
        self,
        audio: np.ndarray,
        text: str,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Perform forced alignment to get phoneme boundaries"""
        # This would use Montreal Forced Aligner or similar
        # For now, return placeholder
        
        words = text.split()
        word_boundaries = []
        phoneme_boundaries = []
        
        # Simple linear distribution (placeholder)
        duration = len(audio) / sample_rate
        time_per_word = duration / len(words) if words else 0
        
        current_time = 0
        for word in words:
            word_start = current_time
            word_end = current_time + time_per_word
            word_boundaries.append({
                "word": word,
                "start": word_start,
                "end": word_end
            })
            
            # Fake phonemes for word
            phoneme_count = len(word) // 2 + 1
            phoneme_duration = time_per_word / phoneme_count
            
            for i in range(phoneme_count):
                phon_start = word_start + i * phoneme_duration
                phon_end = phon_start + phoneme_duration
                phoneme_boundaries.append({
                    "phoneme": f"P{i}",
                    "start": phon_start,
                    "end": phon_end
                })
            
            current_time = word_end
        
        return {
            "words": word_boundaries,
            "phonemes": phoneme_boundaries
        }


class PronunciationCoach:
    """Interactive pronunciation coaching system"""
    
    def __init__(self, processor: AdvancedLanguageProcessor):
        self.processor = processor
        self.session_history = []
        
    async def analyze_and_coach(
        self,
        audio: Union[str, np.ndarray],
        language: str,
        target_text: str
    ) -> Dict[str, Any]:
        """Provide real-time pronunciation coaching"""
        # Analyze pronunciation
        features = await self.processor.analyze_comprehensive(
            audio, language, target_text
        )
        
        # Generate coaching feedback
        coaching = {
            "overall_feedback": self._generate_overall_feedback(features),
            "specific_corrections": self._generate_corrections(features),
            "practice_exercises": self._suggest_exercises(features, language),
            "visual_guides": self._create_visual_guides(features),
            "next_steps": self._recommend_next_steps(features)
        }
        
        # Track progress
        self.session_history.append({
            "timestamp": time.time(),
            "scores": {
                "pronunciation": features.pronunciation_score,
                "fluency": features.fluency_score,
                "accent": features.accent_score
            }
        })
        
        return coaching
    
    def _generate_overall_feedback(self, features: AdvancedLanguageFeatures) -> str:
        """Generate encouraging overall feedback"""
        score = features.pronunciation_score
        
        if score > 0.9:
            return "Excellent pronunciation! You sound very close to a native speaker."
        elif score > 0.8:
            return "Great job! Your pronunciation is very good with just minor areas to improve."
        elif score > 0.7:
            return "Good pronunciation! Let's work on a few specific sounds to make it even better."
        elif score > 0.6:
            return "You're making good progress! Focus on the highlighted areas for improvement."
        else:
            return "Keep practicing! Let's break this down into smaller parts."
    
    def _generate_corrections(
        self,
        features: AdvancedLanguageFeatures
    ) -> List[Dict[str, Any]]:
        """Generate specific pronunciation corrections"""
        corrections = []
        
        for segment in features.problem_segments[:5]:  # Top 5 problems
            correction = {
                "phoneme": segment["phoneme"],
                "issue": self._identify_issue(segment),
                "instruction": self._get_correction_instruction(segment["phoneme"]),
                "audio_example": self._get_audio_example(segment["phoneme"]),
                "visual_guide": self._get_visual_guide(segment["phoneme"])
            }
            corrections.append(correction)
        
        return corrections
    
    def _identify_issue(self, segment: Dict[str, Any]) -> str:
        """Identify specific pronunciation issue"""
        score = segment["score"]
        phoneme = segment["phoneme"]
        
        if score < 0.3:
            return f"The sound '{phoneme}' is missing or very unclear"
        elif score < 0.5:
            return f"The sound '{phoneme}' needs significant improvement"
        else:
            return f"The sound '{phoneme}' is slightly off"
    
    def _get_correction_instruction(self, phoneme: str) -> str:
        """Get specific instruction for correcting a phoneme"""
        # Database of phoneme-specific instructions
        instructions = {
            "θ": "Place your tongue between your teeth and blow air out gently",
            "ð": "Same as 'th' but add voice - your throat should vibrate",
            "r": "Curl your tongue back without touching the roof of your mouth",
            "l": "Touch the tip of your tongue to the ridge behind your upper teeth",
            # Add more phoneme instructions
        }
        
        return instructions.get(phoneme, f"Focus on the correct position for '{phoneme}'")
    
    def _get_audio_example(self, phoneme: str) -> Optional[str]:
        """Get audio example for phoneme (would return actual audio URL)"""
        return f"/audio/phonemes/{phoneme}.mp3"
    
    def _get_visual_guide(self, phoneme: str) -> Optional[str]:
        """Get visual guide for phoneme (would return actual image/video URL)"""
        return f"/visuals/phonemes/{phoneme}.gif"
    
    def _suggest_exercises(
        self,
        features: AdvancedLanguageFeatures,
        language: str
    ) -> List[Dict[str, Any]]:
        """Suggest targeted practice exercises"""
        exercises = []
        
        # Minimal pairs for problem phonemes
        for segment in features.problem_segments[:3]:
            phoneme = segment["phoneme"]
            exercises.append({
                "type": "minimal_pairs",
                "title": f"Practice '{phoneme}' sounds",
                "pairs": self._get_minimal_pairs(phoneme, language),
                "instructions": f"Practice distinguishing and producing '{phoneme}'"
            })
        
        # Rhythm exercises
        if features.fluency_score < 0.7:
            exercises.append({
                "type": "rhythm",
                "title": "Rhythm and Flow Practice",
                "content": "Shadow native speakers to improve your rhythm",
                "audio_samples": ["/audio/rhythm/sample1.mp3"]
            })
        
        # Tone exercises for tonal languages
        if self.processor.language_configs.get(language, {}).get("tonal"):
            exercises.append({
                "type": "tone_drills",
                "title": "Tone Practice",
                "patterns": self._get_tone_patterns(language),
                "instructions": "Practice these tone combinations"
            })
        
        return exercises
    
    def _get_minimal_pairs(self, phoneme: str, language: str) -> List[Tuple[str, str]]:
        """Get minimal pairs for practicing specific phoneme"""
        # This would have a comprehensive database
        minimal_pairs = {
            "r": [("right", "light"), ("pray", "play"), ("crown", "clown")],
            "l": [("light", "right"), ("play", "pray"), ("clown", "crown")],
            "θ": [("think", "sink"), ("path", "pass"), ("tooth", "toot")],
            # Add more
        }
        
        return minimal_pairs.get(phoneme, [])
    
    def _get_tone_patterns(self, language: str) -> List[Dict[str, Any]]:
        """Get tone practice patterns for tonal languages"""
        if language == "zh":  # Chinese
            return [
                {"pattern": "1-1", "example": "māmā", "meaning": "mother"},
                {"pattern": "2-3", "example": "méiyǒu", "meaning": "don't have"},
                {"pattern": "3-3", "example": "nǐhǎo", "meaning": "hello"},
                {"pattern": "4-4", "example": "zàijiàn", "meaning": "goodbye"}
            ]
        return []
    
    def _create_visual_guides(
        self,
        features: AdvancedLanguageFeatures
    ) -> Dict[str, Any]:
        """Create visual guides for pronunciation"""
        return {
            "mouth_positions": [
                {
                    "phoneme": seg["phoneme"],
                    "image": f"/images/mouth/{seg['phoneme']}.png",
                    "description": self._get_mouth_description(seg["phoneme"])
                }
                for seg in features.problem_segments[:3]
            ],
            "pitch_contour": {
                "learner": features.pitch_contour.tolist(),
                "target": self._get_target_pitch_contour(features.language)
            },
            "waveform_comparison": {
                "learner": features.comparison_data.get("waveform") if features.comparison_data else None,
                "native": "native_waveform_data"
            }
        }
    
    def _get_mouth_description(self, phoneme: str) -> str:
        """Get description of mouth position for phoneme"""
        descriptions = {
            "θ": "Tongue between teeth, lips slightly open",
            "r": "Tongue curled back, lips slightly rounded",
            "l": "Tongue tip touching alveolar ridge",
            # Add more
        }
        return descriptions.get(phoneme, "")
    
    def _get_target_pitch_contour(self, language: str) -> List[float]:
        """Get target pitch contour for comparison"""
        # This would return actual native speaker pitch data
        return []
    
    def _recommend_next_steps(
        self,
        features: AdvancedLanguageFeatures
    ) -> List[str]:
        """Recommend next learning steps"""
        steps = []
        
        # Check progress history
        if len(self.session_history) > 1:
            improvement = (
                features.pronunciation_score - 
                self.session_history[-2]["scores"]["pronunciation"]
            )
            
            if improvement > 0.05:
                steps.append("Great progress! Continue with more challenging texts.")
            elif improvement < -0.05:
                steps.append("Take a break and come back refreshed.")
            else:
                steps.append("Try focusing on one specific sound at a time.")
        
        # Specific recommendations based on scores
        if features.pronunciation_score > 0.85:
            steps.append("Move on to conversational practice with native speakers.")
        elif features.pronunciation_score > 0.7:
            steps.append("Practice with tongue twisters to improve accuracy.")
        else:
            steps.append("Focus on individual sounds before full sentences.")
        
        if features.fluency_score < 0.7:
            steps.append("Practice reading aloud for 10 minutes daily.")
        
        if features.accent_score < 0.7:
            steps.append("Listen to native speakers and try to mimic their intonation.")
        
        return steps[:3]  # Top 3 recommendations