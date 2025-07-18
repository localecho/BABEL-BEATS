#!/usr/bin/env python3
"""
Phoneme Alignment Service for BABEL-BEATS
High-precision phoneme-level analysis using Montreal Forced Aligner and other tools
"""

import os
import tempfile
import shutil
import subprocess
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend, FestivalBackend, SegmentsBackend
from phonemizer.separator import Separator
import g2p_en
import epitran
import panphon
from praatio import textgrid
import parselmouth
from scipy import signal, stats
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles
import hashlib
from datetime import datetime
import redis
from typing_extensions import Literal

logger = logging.getLogger(__name__)


@dataclass
class PhonemeSegment:
    """Represents a single phoneme with timing and acoustic features"""
    phoneme: str
    start_time: float
    end_time: float
    duration: float
    confidence: float
    formants: List[float]  # F1, F2, F3
    pitch: Optional[float]
    intensity: Optional[float]
    voice_quality: Dict[str, float]
    articulation_features: Dict[str, Any]
    ipa_symbol: str
    manner: str  # stop, fricative, vowel, etc.
    place: str   # bilabial, alveolar, etc.
    voicing: bool
    stress_level: int  # 0=unstressed, 1=primary, 2=secondary


@dataclass
class PhonemeAlignment:
    """Complete phoneme alignment results"""
    segments: List[PhonemeSegment]
    text: str
    language: str
    total_duration: float
    average_phoneme_duration: float
    speech_rate: float  # phonemes per second
    articulation_rate: float  # excluding pauses
    pause_ratio: float
    phoneme_inventory: Dict[str, int]
    alignment_confidence: float
    prosodic_features: Dict[str, Any]
    timing_patterns: Dict[str, Any]


class PhonemeAlignmentService:
    """
    Advanced phoneme alignment service with multiple backends
    Supports 100+ languages with high precision
    """
    
    def __init__(self, cache_dir: str = "/tmp/phoneme_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize backends
        self._init_backends()
        
        # Redis for caching
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        logger.info("Phoneme Alignment Service initialized")
    
    def _init_backends(self):
        """Initialize various phoneme processing backends"""
        # G2P for English
        self.g2p_en = g2p_en.G2p()
        
        # Epitran for IPA conversion
        self.epitran_backends = {
            'eng': epitran.Epitran('eng-Latn'),
            'spa': epitran.Epitran('spa-Latn'),
            'fra': epitran.Epitran('fra-Latn'),
            'deu': epitran.Epitran('deu-Latn'),
            'ita': epitran.Epitran('ita-Latn'),
            'por': epitran.Epitran('por-Latn'),
            'cmn': epitran.Epitran('cmn-Hans'),
            'jpn': epitran.Epitran('jpn-Hira'),
            'kor': epitran.Epitran('kor-Hang'),
            'ara': epitran.Epitran('ara-Arab'),
            'hin': epitran.Epitran('hin-Deva'),
            'rus': epitran.Epitran('rus-Cyrl'),
        }
        
        # Panphon for phonological features
        self.panphon = panphon.Panphon()
        
        # Wav2Vec2 for phoneme recognition
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.wav2vec2_model.eval()
        
        # Phonemizer backends
        self.phonemizer_separator = Separator(
            phone=' ',
            syllable='|',
            word=' _ '
        )
        
        logger.info("All phoneme processing backends initialized")
    
    async def align_phonemes(
        self,
        audio_path: Union[str, np.ndarray],
        text: str,
        language: str,
        use_mfa: bool = True,
        detailed_features: bool = True,
        cache_results: bool = True
    ) -> PhonemeAlignment:
        """
        Perform high-precision phoneme alignment
        
        Args:
            audio_path: Path to audio file or numpy array
            text: Transcript text
            language: Language code (ISO 639-1)
            use_mfa: Use Montreal Forced Aligner for alignment
            detailed_features: Extract detailed acoustic features
            cache_results: Cache results for faster retrieval
        
        Returns:
            PhonemeAlignment object with detailed timing and features
        """
        # Check cache first
        if cache_results and isinstance(audio_path, str):
            cache_key = self._get_cache_key(audio_path, text, language)
            cached = await self._get_cached_alignment(cache_key)
            if cached:
                logger.info("Returning cached alignment")
                return cached
        
        # Load audio
        if isinstance(audio_path, str):
            audio, sr = librosa.load(audio_path, sr=16000)
        else:
            audio = audio_path
            sr = 16000
        
        # Get phoneme sequence
        phonemes = await self._get_phonemes(text, language)
        
        # Perform alignment
        if use_mfa and self._is_mfa_available():
            segments = await self._align_with_mfa(audio, sr, text, language)
        else:
            segments = await self._align_with_wav2vec2(audio, sr, phonemes, language)
        
        # Extract detailed features if requested
        if detailed_features:
            segments = await self._extract_detailed_features(audio, sr, segments)
        
        # Calculate alignment metrics
        alignment = self._create_alignment_object(segments, text, language, audio, sr)
        
        # Cache results
        if cache_results and isinstance(audio_path, str):
            await self._cache_alignment(cache_key, alignment)
        
        return alignment
    
    async def _get_phonemes(self, text: str, language: str) -> List[str]:
        """Get phoneme sequence from text"""
        lang_code = self._get_lang_code(language)
        
        # Use language-specific backends
        if language.startswith('en'):
            # Use G2P for English
            phonemes = self.g2p_en(text)
            # Filter out non-phoneme tokens
            phonemes = [p for p in phonemes if p != ' ' and not p.isdigit()]
        else:
            # Use phonemizer for other languages
            try:
                backend = self._get_phonemizer_backend(language)
                phonemes_str = phonemize(
                    text,
                    language=lang_code,
                    backend=backend,
                    separator=self.phonemizer_separator,
                    strip=True,
                    preserve_punctuation=False,
                    njobs=1
                )
                phonemes = phonemes_str.split()
            except Exception as e:
                logger.warning(f"Phonemizer failed for {language}, using epitran: {e}")
                # Fallback to epitran
                phonemes = self._get_epitran_phonemes(text, language)
        
        return phonemes
    
    def _get_epitran_phonemes(self, text: str, language: str) -> List[str]:
        """Get phonemes using epitran"""
        lang_map = {
            'en': 'eng', 'es': 'spa', 'fr': 'fra', 'de': 'deu',
            'it': 'ita', 'pt': 'por', 'zh': 'cmn', 'ja': 'jpn',
            'ko': 'kor', 'ar': 'ara', 'hi': 'hin', 'ru': 'rus'
        }
        
        lang_code = lang_map.get(language[:2], 'eng')
        if lang_code in self.epitran_backends:
            epi = self.epitran_backends[lang_code]
            ipa = epi.transliterate(text)
            # Split into individual phonemes
            phonemes = [p for p in ipa if p.strip()]
            return phonemes
        else:
            # Default to character-level
            return list(text.replace(' ', ''))
    
    async def _align_with_mfa(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        language: str
    ) -> List[PhonemeSegment]:
        """Align using Montreal Forced Aligner"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Save audio and text
            audio_path = os.path.join(temp_dir, "audio.wav")
            text_path = os.path.join(temp_dir, "transcript.txt")
            
            sf.write(audio_path, audio, sr)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Run MFA
            output_dir = os.path.join(temp_dir, "aligned")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get appropriate acoustic model
            acoustic_model = self._get_mfa_model(language)
            
            cmd = [
                "mfa", "align",
                temp_dir,
                acoustic_model,
                acoustic_model,  # Using same model for dict
                output_dir,
                "--clean",
                "--overwrite"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"MFA failed: {result.stderr}")
                raise RuntimeError("MFA alignment failed")
            
            # Parse TextGrid output
            textgrid_path = os.path.join(output_dir, "audio.TextGrid")
            segments = self._parse_textgrid(textgrid_path)
            
            return segments
            
        finally:
            shutil.rmtree(temp_dir)
    
    async def _align_with_wav2vec2(
        self,
        audio: np.ndarray,
        sr: int,
        phonemes: List[str],
        language: str
    ) -> List[PhonemeSegment]:
        """Align using Wav2Vec2 model"""
        # Prepare audio
        input_values = self.wav2vec2_processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_values
        
        # Get model predictions
        with torch.no_grad():
            logits = self.wav2vec2_model(input_values).logits
        
        # Get predicted phonemes
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.wav2vec2_processor.batch_decode(predicted_ids)
        
        # Perform alignment with DTW
        segments = await self._dtw_alignment(
            audio, sr, phonemes, transcription[0], logits
        )
        
        return segments
    
    async def _dtw_alignment(
        self,
        audio: np.ndarray,
        sr: int,
        ref_phonemes: List[str],
        pred_phonemes: str,
        logits: torch.Tensor
    ) -> List[PhonemeSegment]:
        """Dynamic Time Warping alignment between reference and predicted phonemes"""
        from dtw import dtw
        
        # Convert predictions to phoneme sequence
        pred_phones = pred_phonemes.replace(' ', '').replace('|', '')
        
        # Create cost matrix
        cost_matrix = np.zeros((len(ref_phonemes), len(pred_phones)))
        
        for i, ref_phone in enumerate(ref_phonemes):
            for j, pred_phone in enumerate(pred_phones):
                # Use phonological distance
                distance = self._phonological_distance(ref_phone, pred_phone)
                cost_matrix[i, j] = distance
        
        # Perform DTW
        alignment = dtw(cost_matrix, keep_internals=True)
        
        # Extract segments with timing
        segments = []
        frame_duration = len(audio) / sr / logits.shape[1]
        
        for i, (ref_idx, pred_idx) in enumerate(zip(alignment.index1, alignment.index2)):
            if i < len(ref_phonemes):
                phoneme = ref_phonemes[ref_idx]
                start_time = pred_idx * frame_duration
                end_time = (pred_idx + 1) * frame_duration if i < len(alignment.index1) - 1 else len(audio) / sr
                
                # Get confidence from logits
                confidence = torch.softmax(logits[0, pred_idx], dim=0).max().item()
                
                segment = PhonemeSegment(
                    phoneme=phoneme,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    confidence=confidence,
                    formants=[],
                    pitch=None,
                    intensity=None,
                    voice_quality={},
                    articulation_features={},
                    ipa_symbol=phoneme,
                    manner="unknown",
                    place="unknown",
                    voicing=False,
                    stress_level=0
                )
                segments.append(segment)
        
        return segments
    
    def _phonological_distance(self, phone1: str, phone2: str) -> float:
        """Calculate phonological distance between two phonemes"""
        try:
            # Get feature vectors
            features1 = self.panphon.fts(phone1)
            features2 = self.panphon.fts(phone2)
            
            if features1 and features2:
                # Hamming distance between feature vectors
                dist = self.panphon.hammond_distance(
                    features1.to_vector(),
                    features2.to_vector()
                )
                return dist
            else:
                # Fallback to simple comparison
                return 0.0 if phone1 == phone2 else 1.0
        except:
            return 0.0 if phone1 == phone2 else 1.0
    
    async def _extract_detailed_features(
        self,
        audio: np.ndarray,
        sr: int,
        segments: List[PhonemeSegment]
    ) -> List[PhonemeSegment]:
        """Extract detailed acoustic features for each phoneme"""
        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio, sr)
        
        # Extract formants
        formant = sound.to_formant_burg()
        
        # Extract pitch
        pitch = sound.to_pitch()
        
        # Extract intensity
        intensity = sound.to_intensity()
        
        # Process each segment
        enhanced_segments = []
        for segment in segments:
            # Extract segment audio
            start_sample = int(segment.start_time * sr)
            end_sample = int(segment.end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) > 0:
                # Get formants
                formants = []
                for i in range(1, 4):  # F1, F2, F3
                    f = formant.get_value_at_time(
                        i, segment.start_time + segment.duration / 2
                    )
                    formants.append(f if f else 0.0)
                
                # Get pitch
                pitch_value = pitch.get_value_at_time(
                    segment.start_time + segment.duration / 2
                )
                
                # Get intensity
                intensity_value = intensity.get_value(
                    segment.start_time + segment.duration / 2
                )
                
                # Voice quality analysis
                voice_quality = self._analyze_voice_quality(
                    segment_audio, sr
                )
                
                # Get articulation features
                articulation = self._get_articulation_features(
                    segment.phoneme
                )
                
                # Update segment
                segment.formants = formants
                segment.pitch = pitch_value
                segment.intensity = intensity_value
                segment.voice_quality = voice_quality
                segment.articulation_features = articulation
                
                # Get IPA and features
                ipa_data = self._get_ipa_features(segment.phoneme)
                segment.ipa_symbol = ipa_data['ipa']
                segment.manner = ipa_data['manner']
                segment.place = ipa_data['place']
                segment.voicing = ipa_data['voicing']
                
            enhanced_segments.append(segment)
        
        return enhanced_segments
    
    def _analyze_voice_quality(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Analyze voice quality metrics"""
        if len(audio) < sr * 0.02:  # Need at least 20ms
            return {
                'jitter': 0.0,
                'shimmer': 0.0,
                'hnr': 0.0,
                'cepstral_peak': 0.0
            }
        
        sound = parselmouth.Sound(audio, sr)
        
        # Jitter (pitch variation)
        point_process = parselmouth.praat.call(
            sound, "To PointProcess (periodic, cc)...", 75, 600
        )
        jitter = parselmouth.praat.call(
            point_process, "Get jitter (local)...", 0, 0, 0.0001, 0.02, 1.3
        ) if point_process else 0.0
        
        # Shimmer (amplitude variation)
        shimmer = parselmouth.praat.call(
            [sound, point_process], "Get shimmer (local)...", 0, 0, 0.0001, 0.02, 1.3, 1.6
        ) if point_process else 0.0
        
        # HNR (Harmonics-to-Noise Ratio)
        harmonicity = sound.to_harmonicity()
        hnr = harmonicity.get_value(harmonicity.get_time_from_index(1))
        
        # Cepstral peak prominence (CPP)
        # Simplified version
        cepstrum = np.fft.ifft(np.log(np.abs(np.fft.fft(audio)) + 1e-10)).real
        cepstral_peak = np.max(np.abs(cepstrum[20:200])) if len(cepstrum) > 200 else 0.0
        
        return {
            'jitter': float(jitter) if jitter else 0.0,
            'shimmer': float(shimmer) if shimmer else 0.0,
            'hnr': float(hnr) if hnr else 0.0,
            'cepstral_peak': float(cepstral_peak)
        }
    
    def _get_articulation_features(self, phoneme: str) -> Dict[str, Any]:
        """Get articulation features for phoneme"""
        try:
            # Get panphon features
            features = self.panphon.fts(phoneme)
            if features:
                return {
                    'high': features.match(['+', 'hi']),
                    'low': features.match(['+', 'lo']),
                    'front': features.match(['+', 'front']),
                    'back': features.match(['+', 'back']),
                    'round': features.match(['+', 'round']),
                    'voiced': features.match(['+', 'voi']),
                    'nasal': features.match(['+', 'nas']),
                    'continuant': features.match(['+', 'cont']),
                    'strident': features.match(['+', 'strid']),
                    'lateral': features.match(['+', 'lat'])
                }
        except:
            pass
        
        # Fallback
        return {
            'high': False, 'low': False, 'front': False, 'back': False,
            'round': False, 'voiced': False, 'nasal': False,
            'continuant': False, 'strident': False, 'lateral': False
        }
    
    def _get_ipa_features(self, phoneme: str) -> Dict[str, Any]:
        """Get IPA symbol and phonological features"""
        # Common phoneme mappings
        manner_map = {
            'p': 'stop', 'b': 'stop', 't': 'stop', 'd': 'stop',
            'k': 'stop', 'g': 'stop', 'f': 'fricative', 'v': 'fricative',
            's': 'fricative', 'z': 'fricative', 'θ': 'fricative',
            'ð': 'fricative', 'ʃ': 'fricative', 'ʒ': 'fricative',
            'm': 'nasal', 'n': 'nasal', 'ŋ': 'nasal', 'l': 'liquid',
            'r': 'liquid', 'w': 'glide', 'j': 'glide', 'h': 'fricative',
            'a': 'vowel', 'e': 'vowel', 'i': 'vowel', 'o': 'vowel',
            'u': 'vowel', 'æ': 'vowel', 'ɪ': 'vowel', 'ʊ': 'vowel'
        }
        
        place_map = {
            'p': 'bilabial', 'b': 'bilabial', 'm': 'bilabial',
            'f': 'labiodental', 'v': 'labiodental', 'θ': 'dental',
            'ð': 'dental', 't': 'alveolar', 'd': 'alveolar',
            'n': 'alveolar', 's': 'alveolar', 'z': 'alveolar',
            'l': 'alveolar', 'r': 'alveolar', 'ʃ': 'postalveolar',
            'ʒ': 'postalveolar', 'j': 'palatal', 'k': 'velar',
            'g': 'velar', 'ŋ': 'velar', 'w': 'velar', 'h': 'glottal'
        }
        
        voiced_set = {'b', 'd', 'g', 'v', 'ð', 'z', 'ʒ', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j'}
        
        # Determine features
        ipa = phoneme.lower()
        manner = manner_map.get(ipa, 'unknown')
        place = place_map.get(ipa, 'unknown')
        voicing = ipa in voiced_set
        
        return {
            'ipa': ipa,
            'manner': manner,
            'place': place,
            'voicing': voicing
        }
    
    def _create_alignment_object(
        self,
        segments: List[PhonemeSegment],
        text: str,
        language: str,
        audio: np.ndarray,
        sr: int
    ) -> PhonemeAlignment:
        """Create complete alignment object with metrics"""
        if not segments:
            return PhonemeAlignment(
                segments=[],
                text=text,
                language=language,
                total_duration=0.0,
                average_phoneme_duration=0.0,
                speech_rate=0.0,
                articulation_rate=0.0,
                pause_ratio=0.0,
                phoneme_inventory={},
                alignment_confidence=0.0,
                prosodic_features={},
                timing_patterns={}
            )
        
        # Calculate metrics
        total_duration = len(audio) / sr
        phoneme_durations = [s.duration for s in segments]
        
        # Speech rate (phonemes per second)
        speech_rate = len(segments) / total_duration
        
        # Articulation rate (excluding pauses)
        non_pause_segments = [s for s in segments if s.phoneme not in ['<sil>', 'sp', '']]
        non_pause_duration = sum(s.duration for s in non_pause_segments)
        articulation_rate = len(non_pause_segments) / non_pause_duration if non_pause_duration > 0 else 0
        
        # Pause ratio
        pause_duration = total_duration - non_pause_duration
        pause_ratio = pause_duration / total_duration
        
        # Phoneme inventory
        phoneme_inventory = {}
        for segment in segments:
            phoneme = segment.phoneme
            phoneme_inventory[phoneme] = phoneme_inventory.get(phoneme, 0) + 1
        
        # Average confidence
        confidences = [s.confidence for s in segments if s.confidence > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Prosodic features
        prosodic_features = self._extract_prosodic_features(segments)
        
        # Timing patterns
        timing_patterns = self._analyze_timing_patterns(segments)
        
        return PhonemeAlignment(
            segments=segments,
            text=text,
            language=language,
            total_duration=total_duration,
            average_phoneme_duration=np.mean(phoneme_durations),
            speech_rate=speech_rate,
            articulation_rate=articulation_rate,
            pause_ratio=pause_ratio,
            phoneme_inventory=phoneme_inventory,
            alignment_confidence=avg_confidence,
            prosodic_features=prosodic_features,
            timing_patterns=timing_patterns
        )
    
    def _extract_prosodic_features(
        self,
        segments: List[PhonemeSegment]
    ) -> Dict[str, Any]:
        """Extract prosodic features from aligned segments"""
        # Pitch contour
        pitches = [s.pitch for s in segments if s.pitch]
        pitch_mean = np.mean(pitches) if pitches else 0
        pitch_std = np.std(pitches) if pitches else 0
        pitch_range = max(pitches) - min(pitches) if pitches else 0
        
        # Intensity contour
        intensities = [s.intensity for s in segments if s.intensity]
        intensity_mean = np.mean(intensities) if intensities else 0
        intensity_std = np.std(intensities) if intensities else 0
        
        # Duration patterns
        vowel_durations = [s.duration for s in segments if s.manner == 'vowel']
        consonant_durations = [s.duration for s in segments if s.manner != 'vowel']
        
        # Rhythm metrics
        duration_deltas = np.diff([s.duration for s in segments])
        rhythm_variability = np.std(duration_deltas) if len(duration_deltas) > 0 else 0
        
        return {
            'pitch': {
                'mean': pitch_mean,
                'std': pitch_std,
                'range': pitch_range,
                'contour': self._classify_pitch_contour(pitches)
            },
            'intensity': {
                'mean': intensity_mean,
                'std': intensity_std,
                'dynamic_range': max(intensities) - min(intensities) if intensities else 0
            },
            'duration': {
                'vowel_mean': np.mean(vowel_durations) if vowel_durations else 0,
                'consonant_mean': np.mean(consonant_durations) if consonant_durations else 0,
                'vowel_consonant_ratio': (
                    np.mean(vowel_durations) / np.mean(consonant_durations)
                    if vowel_durations and consonant_durations else 1.0
                )
            },
            'rhythm': {
                'variability': rhythm_variability,
                'isochrony': 1 - rhythm_variability  # Simplified isochrony measure
            }
        }
    
    def _classify_pitch_contour(self, pitches: List[float]) -> str:
        """Classify pitch contour pattern"""
        if not pitches or len(pitches) < 3:
            return "unknown"
        
        # Fit polynomial to pitch contour
        x = np.arange(len(pitches))
        coeffs = np.polyfit(x, pitches, 2)
        
        # Classify based on coefficients
        if abs(coeffs[0]) < 0.01:  # Nearly linear
            if coeffs[1] > 0.5:
                return "rising"
            elif coeffs[1] < -0.5:
                return "falling"
            else:
                return "level"
        else:
            if coeffs[0] > 0:
                return "rise-fall"
            else:
                return "fall-rise"
    
    def _analyze_timing_patterns(
        self,
        segments: List[PhonemeSegment]
    ) -> Dict[str, Any]:
        """Analyze timing patterns in phoneme sequences"""
        if not segments:
            return {}
        
        # Inter-phoneme intervals
        intervals = []
        for i in range(1, len(segments)):
            interval = segments[i].start_time - segments[i-1].end_time
            intervals.append(interval)
        
        # Clustering for rhythm groups
        if intervals:
            # Simple k-means for rhythm groups
            from sklearn.cluster import KMeans
            intervals_array = np.array(intervals).reshape(-1, 1)
            
            try:
                kmeans = KMeans(n_clusters=min(3, len(intervals)), random_state=42)
                clusters = kmeans.fit_predict(intervals_array)
                rhythm_groups = {
                    'short': float(kmeans.cluster_centers_[0][0]),
                    'medium': float(kmeans.cluster_centers_[1][0]) if len(kmeans.cluster_centers_) > 1 else 0,
                    'long': float(kmeans.cluster_centers_[2][0]) if len(kmeans.cluster_centers_) > 2 else 0
                }
            except:
                rhythm_groups = {'short': 0.0, 'medium': 0.0, 'long': 0.0}
        else:
            rhythm_groups = {'short': 0.0, 'medium': 0.0, 'long': 0.0}
        
        # Phoneme transition patterns
        transitions = {}
        for i in range(1, len(segments)):
            transition = f"{segments[i-1].manner}->{segments[i].manner}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        return {
            'inter_phoneme_intervals': {
                'mean': np.mean(intervals) if intervals else 0,
                'std': np.std(intervals) if intervals else 0,
                'min': min(intervals) if intervals else 0,
                'max': max(intervals) if intervals else 0
            },
            'rhythm_groups': rhythm_groups,
            'transition_patterns': transitions,
            'coarticulation_score': self._calculate_coarticulation(segments)
        }
    
    def _calculate_coarticulation(self, segments: List[PhonemeSegment]) -> float:
        """Calculate coarticulation score based on formant transitions"""
        if len(segments) < 2:
            return 0.0
        
        transitions = []
        for i in range(1, len(segments)):
            if segments[i-1].formants and segments[i].formants:
                # Calculate formant transition smoothness
                f1_diff = abs(segments[i].formants[0] - segments[i-1].formants[0])
                f2_diff = abs(segments[i].formants[1] - segments[i-1].formants[1])
                
                # Normalize by duration
                transition_duration = segments[i].start_time - segments[i-1].end_time
                if transition_duration > 0:
                    smoothness = 1 / (1 + (f1_diff + f2_diff) * transition_duration)
                    transitions.append(smoothness)
        
        return np.mean(transitions) if transitions else 0.0
    
    def _get_cache_key(self, audio_path: str, text: str, language: str) -> str:
        """Generate cache key for alignment"""
        content = f"{audio_path}:{text}:{language}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _get_cached_alignment(self, cache_key: str) -> Optional[PhonemeAlignment]:
        """Retrieve cached alignment"""
        try:
            cached_json = self.redis_client.get(f"phoneme_alignment:{cache_key}")
            if cached_json:
                data = json.loads(cached_json)
                # Reconstruct PhonemeSegment objects
                segments = []
                for seg_data in data['segments']:
                    segment = PhonemeSegment(**seg_data)
                    segments.append(segment)
                
                # Reconstruct PhonemeAlignment
                data['segments'] = segments
                return PhonemeAlignment(**data)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    async def _cache_alignment(self, cache_key: str, alignment: PhonemeAlignment):
        """Cache alignment results"""
        try:
            # Convert to JSON-serializable format
            data = asdict(alignment)
            # Convert segment objects
            data['segments'] = [asdict(seg) for seg in alignment.segments]
            
            # Cache with 1-hour expiry
            self.redis_client.setex(
                f"phoneme_alignment:{cache_key}",
                3600,
                json.dumps(data)
            )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _is_mfa_available(self) -> bool:
        """Check if Montreal Forced Aligner is available"""
        try:
            result = subprocess.run(
                ["mfa", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def _get_mfa_model(self, language: str) -> str:
        """Get appropriate MFA model for language"""
        model_map = {
            'en': 'english_us_arpa',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'zh': 'mandarin',
            'ja': 'japanese',
            'ko': 'korean',
            'ru': 'russian',
            'ar': 'arabic',
            'hi': 'hindi'
        }
        
        lang_code = language[:2]
        return model_map.get(lang_code, 'english_us_arpa')
    
    def _get_lang_code(self, language: str) -> str:
        """Convert language code to phonemizer format"""
        lang_map = {
            'en': 'en-us', 'es': 'es', 'fr': 'fr-fr', 'de': 'de',
            'it': 'it', 'pt': 'pt', 'zh': 'zh', 'ja': 'ja',
            'ko': 'ko', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi'
        }
        return lang_map.get(language[:2], 'en-us')
    
    def _get_phonemizer_backend(self, language: str) -> str:
        """Get appropriate phonemizer backend for language"""
        # ESpeak supports most languages
        return 'espeak'
    
    def _parse_textgrid(self, textgrid_path: str) -> List[PhonemeSegment]:
        """Parse MFA TextGrid output"""
        segments = []
        
        try:
            tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)
            
            # Get phone tier
            phone_tier = None
            for tier in tg.tiers:
                if tier.name.lower() in ['phones', 'phone']:
                    phone_tier = tier
                    break
            
            if phone_tier:
                for interval in phone_tier.entries:
                    if interval.label and interval.label != '':
                        segment = PhonemeSegment(
                            phoneme=interval.label,
                            start_time=interval.start,
                            end_time=interval.end,
                            duration=interval.end - interval.start,
                            confidence=1.0,  # MFA alignments are high confidence
                            formants=[],
                            pitch=None,
                            intensity=None,
                            voice_quality={},
                            articulation_features={},
                            ipa_symbol=interval.label,
                            manner="unknown",
                            place="unknown",
                            voicing=False,
                            stress_level=0
                        )
                        segments.append(segment)
        except Exception as e:
            logger.error(f"Error parsing TextGrid: {e}")
        
        return segments
    
    async def analyze_pronunciation_quality(
        self,
        learner_audio: Union[str, np.ndarray],
        native_audio: Union[str, np.ndarray],
        text: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Compare learner pronunciation with native speaker
        
        Returns detailed pronunciation assessment
        """
        # Align both recordings
        learner_alignment = await self.align_phonemes(
            learner_audio, text, language
        )
        native_alignment = await self.align_phonemes(
            native_audio, text, language
        )
        
        # Compare alignments
        comparison = self._compare_alignments(
            learner_alignment, native_alignment
        )
        
        # Calculate scores
        scores = self._calculate_pronunciation_scores(comparison)
        
        # Generate feedback
        feedback = self._generate_pronunciation_feedback(
            comparison, scores, language
        )
        
        return {
            'scores': scores,
            'comparison': comparison,
            'feedback': feedback,
            'learner_alignment': learner_alignment,
            'native_alignment': native_alignment
        }
    
    def _compare_alignments(
        self,
        learner: PhonemeAlignment,
        native: PhonemeAlignment
    ) -> Dict[str, Any]:
        """Compare learner and native alignments"""
        comparisons = []
        
        # Align the segments using DTW
        if len(learner.segments) > 0 and len(native.segments) > 0:
            # Create timing-based alignment
            for i, learner_seg in enumerate(learner.segments):
                # Find corresponding native segment
                native_seg = None
                min_distance = float('inf')
                
                for nat_seg in native.segments:
                    if nat_seg.phoneme == learner_seg.phoneme:
                        time_diff = abs(
                            (learner_seg.start_time + learner_seg.end_time) / 2 -
                            (nat_seg.start_time + nat_seg.end_time) / 2
                        )
                        if time_diff < min_distance:
                            min_distance = time_diff
                            native_seg = nat_seg
                
                if native_seg:
                    comparison = {
                        'phoneme': learner_seg.phoneme,
                        'learner': learner_seg,
                        'native': native_seg,
                        'duration_ratio': learner_seg.duration / native_seg.duration,
                        'pitch_difference': (
                            learner_seg.pitch - native_seg.pitch
                            if learner_seg.pitch and native_seg.pitch else None
                        ),
                        'formant_differences': self._compare_formants(
                            learner_seg.formants, native_seg.formants
                        ),
                        'timing_offset': learner_seg.start_time - native_seg.start_time
                    }
                    comparisons.append(comparison)
        
        return {
            'segment_comparisons': comparisons,
            'overall_timing': {
                'learner_duration': learner.total_duration,
                'native_duration': native.total_duration,
                'duration_ratio': learner.total_duration / native.total_duration
            },
            'speech_rate': {
                'learner': learner.speech_rate,
                'native': native.speech_rate,
                'ratio': learner.speech_rate / native.speech_rate
            },
            'prosody_comparison': self._compare_prosody(learner, native)
        }
    
    def _compare_formants(
        self,
        learner_formants: List[float],
        native_formants: List[float]
    ) -> Dict[str, float]:
        """Compare formant frequencies"""
        if not learner_formants or not native_formants:
            return {'f1_diff': 0, 'f2_diff': 0, 'f3_diff': 0}
        
        return {
            'f1_diff': learner_formants[0] - native_formants[0] if len(learner_formants) > 0 else 0,
            'f2_diff': learner_formants[1] - native_formants[1] if len(learner_formants) > 1 else 0,
            'f3_diff': learner_formants[2] - native_formants[2] if len(learner_formants) > 2 else 0
        }
    
    def _compare_prosody(
        self,
        learner: PhonemeAlignment,
        native: PhonemeAlignment
    ) -> Dict[str, Any]:
        """Compare prosodic features"""
        learner_prosody = learner.prosodic_features
        native_prosody = native.prosodic_features
        
        return {
            'pitch': {
                'mean_difference': (
                    learner_prosody['pitch']['mean'] - native_prosody['pitch']['mean']
                ),
                'range_ratio': (
                    learner_prosody['pitch']['range'] / native_prosody['pitch']['range']
                    if native_prosody['pitch']['range'] > 0 else 1.0
                ),
                'contour_match': (
                    learner_prosody['pitch']['contour'] == native_prosody['pitch']['contour']
                )
            },
            'rhythm': {
                'variability_difference': (
                    learner_prosody['rhythm']['variability'] -
                    native_prosody['rhythm']['variability']
                )
            },
            'intensity': {
                'dynamic_range_ratio': (
                    learner_prosody['intensity']['dynamic_range'] /
                    native_prosody['intensity']['dynamic_range']
                    if native_prosody['intensity']['dynamic_range'] > 0 else 1.0
                )
            }
        }
    
    def _calculate_pronunciation_scores(
        self,
        comparison: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate pronunciation quality scores"""
        scores = {}
        
        # Segmental accuracy
        if comparison['segment_comparisons']:
            duration_scores = []
            formant_scores = []
            
            for comp in comparison['segment_comparisons']:
                # Duration score (closer to 1.0 is better)
                duration_score = 1 - abs(1 - comp['duration_ratio'])
                duration_score = max(0, min(1, duration_score))
                duration_scores.append(duration_score)
                
                # Formant accuracy
                if comp['formant_differences']:
                    f_diffs = comp['formant_differences']
                    # Normalize differences (typical range ~200Hz)
                    f_score = 1 - (
                        abs(f_diffs['f1_diff']) + abs(f_diffs['f2_diff'])
                    ) / 400
                    f_score = max(0, min(1, f_score))
                    formant_scores.append(f_score)
            
            scores['segmental_accuracy'] = np.mean(duration_scores) if duration_scores else 0
            scores['vowel_accuracy'] = np.mean(formant_scores) if formant_scores else 0
        else:
            scores['segmental_accuracy'] = 0
            scores['vowel_accuracy'] = 0
        
        # Prosody score
        prosody_comp = comparison['prosody_comparison']
        pitch_score = 1 - min(1, abs(prosody_comp['pitch']['mean_difference']) / 50)
        rhythm_score = 1 - min(1, abs(prosody_comp['rhythm']['variability_difference']))
        
        scores['prosody'] = (pitch_score + rhythm_score) / 2
        
        # Timing score
        timing_ratio = comparison['overall_timing']['duration_ratio']
        timing_score = 1 - abs(1 - timing_ratio)
        timing_score = max(0, min(1, timing_score))
        scores['timing'] = timing_score
        
        # Overall score
        scores['overall'] = np.mean([
            scores['segmental_accuracy'],
            scores['vowel_accuracy'],
            scores['prosody'],
            scores['timing']
        ])
        
        return scores
    
    def _generate_pronunciation_feedback(
        self,
        comparison: Dict[str, Any],
        scores: Dict[str, float],
        language: str
    ) -> List[str]:
        """Generate actionable pronunciation feedback"""
        feedback = []
        
        # Overall performance
        if scores['overall'] > 0.8:
            feedback.append("Excellent pronunciation! You sound very close to a native speaker.")
        elif scores['overall'] > 0.6:
            feedback.append("Good pronunciation with room for improvement in specific areas.")
        else:
            feedback.append("Keep practicing! Focus on the specific areas mentioned below.")
        
        # Timing feedback
        if scores['timing'] < 0.7:
            if comparison['overall_timing']['duration_ratio'] > 1.2:
                feedback.append("Try to speak a bit faster to match native speech rate.")
            else:
                feedback.append("You're speaking too quickly. Slow down for clearer pronunciation.")
        
        # Segmental feedback
        if scores['segmental_accuracy'] < 0.7:
            # Find problematic phonemes
            problem_phonemes = []
            for comp in comparison['segment_comparisons']:
                if abs(1 - comp['duration_ratio']) > 0.3:
                    problem_phonemes.append(comp['phoneme'])
            
            if problem_phonemes:
                feedback.append(
                    f"Focus on these sounds: {', '.join(set(problem_phonemes[:3]))}"
                )
        
        # Prosody feedback
        if scores['prosody'] < 0.7:
            pitch_diff = comparison['prosody_comparison']['pitch']['mean_difference']
            if abs(pitch_diff) > 20:
                if pitch_diff > 0:
                    feedback.append("Try to lower your overall pitch slightly.")
                else:
                    feedback.append("Try to raise your overall pitch slightly.")
            
            if not comparison['prosody_comparison']['pitch']['contour_match']:
                feedback.append("Work on matching the intonation pattern of native speakers.")
        
        # Language-specific feedback
        if language.startswith('zh') and scores['prosody'] < 0.8:
            feedback.append("Pay special attention to tone accuracy - it's crucial in Mandarin.")
        elif language.startswith('en') and scores['vowel_accuracy'] < 0.7:
            feedback.append("Focus on English vowel sounds, especially 'æ', 'ʌ', and 'ɜː'.")
        
        return feedback
    
    async def cleanup(self):
        """Clean up resources"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        if hasattr(self, 'redis_client'):
            self.redis_client.close()


# Example usage
async def main():
    service = PhonemeAlignmentService()
    
    # Example: Align phonemes
    audio_path = "path/to/audio.wav"
    text = "Hello world, this is a test"
    language = "en"
    
    alignment = await service.align_phonemes(audio_path, text, language)
    
    print(f"Total duration: {alignment.total_duration:.2f}s")
    print(f"Speech rate: {alignment.speech_rate:.1f} phonemes/s")
    print(f"Alignment confidence: {alignment.alignment_confidence:.2%}")
    
    # Show first few phonemes
    for segment in alignment.segments[:5]:
        print(f"{segment.phoneme}: {segment.start_time:.3f}-{segment.end_time:.3f}s")
    
    await service.cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())