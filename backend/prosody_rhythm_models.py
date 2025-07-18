#!/usr/bin/env python3
"""
Deep Learning Models for Prosody and Rhythm Analysis
Advanced neural networks for language rhythm and intonation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import librosa
from scipy import signal
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
import torchaudio.transforms as T
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProsodyFeatures:
    """Container for prosody analysis results"""
    pitch_contour: np.ndarray
    pitch_mean: float
    pitch_std: float
    pitch_range: float
    intonation_pattern: str
    stress_patterns: List[int]
    boundary_tones: List[str]
    prosodic_phrases: List[Tuple[float, float]]
    emotional_tone: str
    speaking_style: str
    confidence: float


@dataclass
class RhythmFeatures:
    """Container for rhythm analysis results"""
    tempo: float
    beat_strength: float
    syncopation: float
    isochrony: float
    rhythm_class: str  # stress-timed, syllable-timed, mora-timed
    timing_patterns: np.ndarray
    accent_patterns: List[int]
    pause_patterns: List[Tuple[float, float]]
    fluency_score: float
    naturalness_score: float


class ProsodyEncoder(nn.Module):
    """
    Deep neural network for prosody feature extraction
    Uses multi-scale CNN + BiLSTM architecture
    """
    
    def __init__(
        self,
        input_dim: int = 80,  # Mel-spectrogram bins
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-scale CNN for local prosodic features
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # BiLSTM for sequential modeling
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention for long-range dependencies
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Prosody-specific heads
        self.pitch_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Pitch prediction
        )
        
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # No stress, primary, secondary
        )
        
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # None, minor, major, final
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # Neutral, happy, sad, angry, fear, surprise, disgust
        )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input features [batch, time, features]
            lengths: Sequence lengths for masking
        
        Returns:
            Dictionary of prosodic predictions
        """
        batch_size, seq_len, _ = x.shape
        
        # CNN processing (requires channel-first)
        x_conv = x.transpose(1, 2)  # [batch, features, time]
        
        # Multi-scale convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x_conv)
            # Align temporal dimension
            conv_out = F.interpolate(
                conv_out,
                size=seq_len // 2,
                mode='linear',
                align_corners=False
            )
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(conv_outputs, dim=1)  # [batch, 384, time/2]
        multi_scale = multi_scale.transpose(1, 2)  # [batch, time/2, 384]
        
        # Fusion
        fused = self.fusion(multi_scale)  # [batch, time/2, hidden]
        
        # LSTM with packing for efficiency
        if lengths is not None:
            # Adjust lengths for pooling
            adj_lengths = lengths // 2
            packed = pack_padded_sequence(
                fused, adj_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(fused)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        features = lstm_out + attn_out
        
        # Prosody predictions
        pitch = self.pitch_head(features).squeeze(-1)
        stress = self.stress_head(features)
        boundary = self.boundary_head(features)
        emotion = self.emotion_head(features)
        
        # Global emotion from pooled features
        if lengths is not None:
            # Masked pooling
            mask = torch.arange(features.size(1)).expand(
                batch_size, -1
            ).to(features.device) < adj_lengths.unsqueeze(1)
            masked_features = features * mask.unsqueeze(-1)
            global_features = masked_features.sum(dim=1) / adj_lengths.unsqueeze(1)
        else:
            global_features = features.mean(dim=1)
        
        global_emotion = self.emotion_head(global_features)
        
        return {
            'pitch': pitch,
            'stress': stress,
            'boundary': boundary,
            'emotion': emotion,
            'global_emotion': global_emotion,
            'features': features
        }


class RhythmAnalyzer(nn.Module):
    """
    Neural network for rhythm analysis
    Combines traditional DSP with deep learning
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        
        # Rhythm feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Temporal convolutions for rhythm patterns
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
            nn.ReLU()
        )
        
        # GRU for temporal dependencies
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Rhythm classification head
        self.rhythm_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # stress-timed, syllable-timed, mora-timed
        )
        
        # Tempo regression head
        self.tempo_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Beat tracking head
        self.beat_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Beat probability
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for rhythm analysis
        
        Args:
            x: Input rhythm features [batch, time, features]
        
        Returns:
            Dictionary of rhythm predictions
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature extraction with reshaping for BatchNorm
        x_reshaped = x.reshape(-1, x.size(-1))
        features = self.feature_extractor(x_reshaped)
        features = features.reshape(batch_size, seq_len, -1)
        
        # TCN processing
        tcn_in = features.transpose(1, 2)
        tcn_out = self.tcn(tcn_in)
        tcn_out = tcn_out.transpose(1, 2)
        
        # Residual connection
        features = features + tcn_out
        
        # GRU processing
        gru_out, _ = self.gru(features)
        
        # Global pooling for classification
        global_features = gru_out.mean(dim=1)
        
        # Predictions
        rhythm_class = self.rhythm_classifier(global_features)
        tempo = self.tempo_head(global_features)
        beats = self.beat_head(gru_out).squeeze(-1)
        
        return {
            'rhythm_class': rhythm_class,
            'tempo': tempo,
            'beats': beats,
            'features': gru_out
        }


class ProsodyRhythmAnalyzer:
    """
    Complete prosody and rhythm analysis system
    Combines neural models with signal processing
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        
        # Initialize models
        self.prosody_model = ProsodyEncoder().to(self.device)
        self.rhythm_model = RhythmAnalyzer().to(self.device)
        
        # Load pretrained weights if available
        if model_path:
            self.load_models(model_path)
        
        # Feature processors
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,
            n_mels=80
        ).to(self.device)
        
        # Wav2Vec2 for advanced features
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        ).to(self.device)
        
        # Feature scalers
        self.prosody_scaler = StandardScaler()
        self.rhythm_scaler = StandardScaler()
        
        # Set models to eval mode
        self.prosody_model.eval()
        self.rhythm_model.eval()
        self.wav2vec2_model.eval()
        
        logger.info(f"Prosody/Rhythm analyzer initialized on {device}")
    
    def analyze_prosody(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000
    ) -> ProsodyFeatures:
        """
        Analyze prosodic features of speech
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
        
        Returns:
            ProsodyFeatures object
        """
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            # Extract mel-spectrogram
            mel_spec = self.mel_transform(audio_tensor)
            mel_spec = mel_spec.transpose(1, 2)  # [batch, time, mels]
            
            # Extract Wav2Vec2 features
            inputs = self.wav2vec2_processor(
                audio_tensor.squeeze(0).cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            wav2vec2_features = self.wav2vec2_model(
                inputs.input_values.to(self.device)
            ).last_hidden_state
            
            # Align features temporally
            if wav2vec2_features.size(1) != mel_spec.size(1):
                wav2vec2_features = F.interpolate(
                    wav2vec2_features.transpose(1, 2),
                    size=mel_spec.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            # Combine features
            combined_features = torch.cat([
                mel_spec,
                wav2vec2_features[:, :mel_spec.size(1), :80]  # Use first 80 dims
            ], dim=-1)
            
            # Prosody prediction
            prosody_outputs = self.prosody_model(combined_features)
            
            # Extract pitch contour using traditional method for validation
            pitch_contour = self._extract_pitch_contour(
                audio_tensor.squeeze(0).cpu().numpy(),
                sample_rate
            )
            
            # Post-process predictions
            pitch_pred = prosody_outputs['pitch'].squeeze(0).cpu().numpy()
            stress_pred = prosody_outputs['stress'].argmax(dim=-1).squeeze(0).cpu().numpy()
            boundary_pred = prosody_outputs['boundary'].argmax(dim=-1).squeeze(0).cpu().numpy()
            emotion_pred = prosody_outputs['global_emotion'].argmax(dim=-1).item()
            
            # Combine neural predictions with signal processing
            if pitch_contour is not None and len(pitch_contour) > 0:
                # Interpolate neural pitch to match extracted pitch
                pitch_combined = self._combine_pitch_estimates(
                    pitch_pred, pitch_contour
                )
                pitch_mean = np.nanmean(pitch_combined)
                pitch_std = np.nanstd(pitch_combined)
                pitch_range = np.nanmax(pitch_combined) - np.nanmin(pitch_combined)
            else:
                pitch_combined = pitch_pred
                pitch_mean = np.mean(pitch_pred)
                pitch_std = np.std(pitch_pred)
                pitch_range = np.max(pitch_pred) - np.min(pitch_pred)
            
            # Detect intonation pattern
            intonation_pattern = self._classify_intonation(pitch_combined)
            
            # Extract prosodic phrases
            prosodic_phrases = self._detect_prosodic_phrases(boundary_pred)
            
            # Map emotion prediction
            emotion_map = [
                'neutral', 'happy', 'sad', 'angry',
                'fearful', 'surprised', 'disgusted'
            ]
            emotional_tone = emotion_map[emotion_pred]
            
            # Determine speaking style
            speaking_style = self._classify_speaking_style(
                pitch_std, pitch_range, stress_pred
            )
            
            # Calculate confidence
            confidence = self._calculate_prosody_confidence(prosody_outputs)
        
        return ProsodyFeatures(
            pitch_contour=pitch_combined,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            pitch_range=pitch_range,
            intonation_pattern=intonation_pattern,
            stress_patterns=stress_pred.tolist(),
            boundary_tones=self._decode_boundaries(boundary_pred),
            prosodic_phrases=prosodic_phrases,
            emotional_tone=emotional_tone,
            speaking_style=speaking_style,
            confidence=confidence
        )
    
    def analyze_rhythm(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000
    ) -> RhythmFeatures:
        """
        Analyze rhythm features of speech
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
        
        Returns:
            RhythmFeatures object
        """
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        audio_tensor = audio_tensor.to(self.device)
        audio_np = audio_tensor.squeeze(0).cpu().numpy()
        
        with torch.no_grad():
            # Extract rhythm features using traditional methods
            tempo, beats = librosa.beat.beat_track(
                y=audio_np, sr=sample_rate
            )
            
            # Onset detection for rhythm
            onset_env = librosa.onset.onset_strength(
                y=audio_np, sr=sample_rate
            )
            
            # Extract tempogram
            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env,
                sr=sample_rate,
                hop_length=512
            )
            
            # Neural rhythm features
            rhythm_features = self._extract_rhythm_features(
                audio_np, sample_rate, onset_env, tempogram
            )
            
            # Convert to tensor
            rhythm_tensor = torch.from_numpy(rhythm_features).float()
            if rhythm_tensor.dim() == 2:
                rhythm_tensor = rhythm_tensor.unsqueeze(0)
            rhythm_tensor = rhythm_tensor.to(self.device)
            
            # Neural rhythm analysis
            rhythm_outputs = self.rhythm_model(rhythm_tensor)
            
            # Post-process predictions
            rhythm_class_pred = rhythm_outputs['rhythm_class'].argmax(dim=-1).item()
            tempo_pred = rhythm_outputs['tempo'].item()
            beat_probs = rhythm_outputs['beats'].squeeze(0).cpu().numpy()
            
            # Combine predictions
            tempo_combined = (tempo + tempo_pred * 60) / 2  # Neural output is normalized
            
            # Calculate rhythm metrics
            beat_strength = self._calculate_beat_strength(onset_env, beats)
            syncopation = self._calculate_syncopation(onset_env, beats)
            isochrony = self._calculate_isochrony(beats)
            
            # Map rhythm class
            rhythm_classes = ['stress-timed', 'syllable-timed', 'mora-timed']
            rhythm_class = rhythm_classes[rhythm_class_pred]
            
            # Extract timing patterns
            timing_patterns = self._extract_timing_patterns(onset_env)
            
            # Detect accents
            accent_patterns = self._detect_accent_patterns(
                audio_np, sample_rate, beats
            )
            
            # Detect pauses
            pause_patterns = self._detect_pauses(audio_np, sample_rate)
            
            # Calculate fluency and naturalness
            fluency_score = self._calculate_fluency(pause_patterns, timing_patterns)
            naturalness_score = self._calculate_naturalness(
                isochrony, syncopation, beat_strength
            )
        
        return RhythmFeatures(
            tempo=tempo_combined,
            beat_strength=beat_strength,
            syncopation=syncopation,
            isochrony=isochrony,
            rhythm_class=rhythm_class,
            timing_patterns=timing_patterns,
            accent_patterns=accent_patterns,
            pause_patterns=pause_patterns,
            fluency_score=fluency_score,
            naturalness_score=naturalness_score
        )
    
    def _extract_pitch_contour(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Optional[np.ndarray]:
        """Extract pitch contour using CREPE"""
        try:
            import crepe
            
            # CREPE pitch detection
            time, frequency, confidence, _ = crepe.predict(
                audio, sr, viterbi=True, step_size=10
            )
            
            # Filter by confidence
            frequency[confidence < 0.5] = np.nan
            
            return frequency
        except ImportError:
            # Fallback to librosa
            f0 = librosa.yin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            return f0
    
    def _combine_pitch_estimates(
        self,
        neural_pitch: np.ndarray,
        signal_pitch: np.ndarray
    ) -> np.ndarray:
        """Combine neural and signal-based pitch estimates"""
        # Interpolate to same length
        if len(neural_pitch) != len(signal_pitch):
            x_neural = np.linspace(0, 1, len(neural_pitch))
            x_signal = np.linspace(0, 1, len(signal_pitch))
            
            # Interpolate neural to signal length
            neural_interp = np.interp(x_signal, x_neural, neural_pitch)
            
            # Weighted combination
            combined = 0.7 * signal_pitch + 0.3 * neural_interp
        else:
            combined = 0.7 * signal_pitch + 0.3 * neural_pitch
        
        return combined
    
    def _classify_intonation(self, pitch_contour: np.ndarray) -> str:
        """Classify intonation pattern"""
        if len(pitch_contour) < 10:
            return "unknown"
        
        # Remove NaN values
        valid_pitch = pitch_contour[~np.isnan(pitch_contour)]
        if len(valid_pitch) < 10:
            return "unknown"
        
        # Fit polynomial
        x = np.linspace(0, 1, len(valid_pitch))
        coeffs = np.polyfit(x, valid_pitch, 2)
        
        # Classify based on shape
        if coeffs[0] > 10:  # Strong upward curve
            return "rise-fall"
        elif coeffs[0] < -10:  # Strong downward curve
            return "fall-rise"
        elif coeffs[1] > 20:  # Linear rise
            return "rising"
        elif coeffs[1] < -20:  # Linear fall
            return "falling"
        else:
            return "level"
    
    def _detect_prosodic_phrases(
        self,
        boundaries: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Detect prosodic phrase boundaries"""
        phrases = []
        start_idx = 0
        
        # Boundary types: 0=none, 1=minor, 2=major, 3=final
        for i, boundary in enumerate(boundaries):
            if boundary >= 2:  # Major or final boundary
                if i > start_idx:
                    # Convert indices to time
                    start_time = start_idx * 0.01  # Assuming 10ms frame shift
                    end_time = i * 0.01
                    phrases.append((start_time, end_time))
                start_idx = i + 1
        
        # Add final phrase
        if start_idx < len(boundaries):
            phrases.append((start_idx * 0.01, len(boundaries) * 0.01))
        
        return phrases
    
    def _classify_speaking_style(
        self,
        pitch_std: float,
        pitch_range: float,
        stress_patterns: np.ndarray
    ) -> str:
        """Classify speaking style based on prosodic features"""
        # Count stress frequency
        stress_freq = np.sum(stress_patterns > 0) / len(stress_patterns)
        
        if pitch_std < 20 and pitch_range < 50:
            return "monotone"
        elif pitch_std > 40 and pitch_range > 100:
            if stress_freq > 0.3:
                return "emphatic"
            else:
                return "expressive"
        elif stress_freq > 0.4:
            return "rhythmic"
        else:
            return "conversational"
    
    def _decode_boundaries(self, boundaries: np.ndarray) -> List[str]:
        """Decode boundary predictions to labels"""
        boundary_map = {
            0: "none",
            1: "minor",
            2: "major", 
            3: "final"
        }
        
        decoded = []
        for b in boundaries:
            decoded.append(boundary_map.get(b, "none"))
        
        return decoded
    
    def _calculate_prosody_confidence(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate confidence score for prosody predictions"""
        # Get softmax probabilities
        stress_probs = F.softmax(outputs['stress'], dim=-1)
        boundary_probs = F.softmax(outputs['boundary'], dim=-1)
        emotion_probs = F.softmax(outputs['global_emotion'], dim=-1)
        
        # Calculate average max probability
        stress_conf = stress_probs.max(dim=-1)[0].mean().item()
        boundary_conf = boundary_probs.max(dim=-1)[0].mean().item()
        emotion_conf = emotion_probs.max(dim=-1)[0].item()
        
        # Weighted average
        confidence = (stress_conf + boundary_conf + emotion_conf) / 3
        
        return confidence
    
    def _extract_rhythm_features(
        self,
        audio: np.ndarray,
        sr: int,
        onset_env: np.ndarray,
        tempogram: np.ndarray
    ) -> np.ndarray:
        """Extract rhythm features for neural model"""
        # Frame-wise features
        n_frames = tempogram.shape[1]
        features = []
        
        for i in range(n_frames):
            frame_features = []
            
            # Tempogram features
            tempo_frame = tempogram[:, i]
            frame_features.extend([
                np.mean(tempo_frame),
                np.std(tempo_frame),
                np.max(tempo_frame),
                np.argmax(tempo_frame)
            ])
            
            # Onset strength
            onset_idx = min(i, len(onset_env) - 1)
            frame_features.append(onset_env[onset_idx])
            
            # Local energy
            start_sample = i * 512  # hop_length
            end_sample = start_sample + 2048  # frame_length
            if end_sample < len(audio):
                frame_audio = audio[start_sample:end_sample]
                energy = np.sqrt(np.mean(frame_audio ** 2))
                zcr = librosa.feature.zero_crossing_rate(frame_audio)[0, 0]
                frame_features.extend([energy, zcr])
            else:
                frame_features.extend([0.0, 0.0])
            
            # Spectral features
            if end_sample < len(audio):
                spectrum = np.abs(np.fft.rfft(frame_audio * np.hanning(len(frame_audio))))
                spectral_centroid = np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-10)
                spectral_rolloff = np.searchsorted(
                    np.cumsum(spectrum),
                    0.85 * np.sum(spectrum)
                )
                frame_features.extend([spectral_centroid, spectral_rolloff])
            else:
                frame_features.extend([0.0, 0.0])
            
            features.append(frame_features)
        
        features_array = np.array(features)
        
        # Pad to fixed size (128 features per frame)
        if features_array.shape[1] < 128:
            padding = np.zeros((features_array.shape[0], 128 - features_array.shape[1]))
            features_array = np.hstack([features_array, padding])
        
        return features_array
    
    def _calculate_beat_strength(
        self,
        onset_env: np.ndarray,
        beats: np.ndarray
    ) -> float:
        """Calculate average beat strength"""
        if len(beats) == 0:
            return 0.0
        
        beat_strengths = []
        for beat in beats:
            if beat < len(onset_env):
                beat_strengths.append(onset_env[beat])
        
        return np.mean(beat_strengths) if beat_strengths else 0.0
    
    def _calculate_syncopation(
        self,
        onset_env: np.ndarray,
        beats: np.ndarray
    ) -> float:
        """Calculate syncopation score"""
        if len(beats) < 2:
            return 0.0
        
        # Expected beat positions
        beat_interval = np.median(np.diff(beats))
        expected_beats = np.arange(beats[0], len(onset_env), beat_interval)
        
        # Find onsets between beats
        syncopation_score = 0.0
        for i in range(len(expected_beats) - 1):
            start = int(expected_beats[i])
            end = int(expected_beats[i + 1])
            
            if end < len(onset_env):
                between_beats = onset_env[start:end]
                # Check for strong onsets off the beat
                offbeat_strength = np.max(between_beats[1:-1]) if len(between_beats) > 2 else 0
                syncopation_score += offbeat_strength
        
        return syncopation_score / (len(expected_beats) - 1)
    
    def _calculate_isochrony(self, beats: np.ndarray) -> float:
        """Calculate isochrony (regularity) of beats"""
        if len(beats) < 3:
            return 0.0
        
        intervals = np.diff(beats)
        if len(intervals) == 0:
            return 0.0
        
        # Coefficient of variation (lower = more isochronous)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        
        # Convert to score (0-1, higher = more regular)
        isochrony = 1 / (1 + cv)
        
        return isochrony
    
    def _extract_timing_patterns(
        self,
        onset_env: np.ndarray
    ) -> np.ndarray:
        """Extract timing patterns from onset envelope"""
        # Find peaks in onset envelope
        peaks, properties = signal.find_peaks(
            onset_env,
            height=np.mean(onset_env),
            distance=10
        )
        
        if len(peaks) < 2:
            return np.array([])
        
        # Inter-onset intervals
        intervals = np.diff(peaks)
        
        return intervals
    
    def _detect_accent_patterns(
        self,
        audio: np.ndarray,
        sr: int,
        beats: np.ndarray
    ) -> List[int]:
        """Detect accent patterns in speech"""
        # Use energy and spectral features to detect accents
        accents = []
        
        for beat in beats:
            start_sample = int(beat * sr / 100)  # Convert to samples
            end_sample = start_sample + int(0.1 * sr)  # 100ms window
            
            if end_sample < len(audio):
                segment = audio[start_sample:end_sample]
                
                # Energy-based accent detection
                energy = np.sqrt(np.mean(segment ** 2))
                spectral_centroid = np.mean(
                    librosa.feature.spectral_centroid(y=segment, sr=sr)
                )
                
                # Simple threshold (could be learned)
                if energy > 0.1 and spectral_centroid > 2000:
                    accents.append(1)  # Accented
                else:
                    accents.append(0)  # Unaccented
        
        return accents
    
    def _detect_pauses(
        self,
        audio: np.ndarray,
        sr: int,
        energy_threshold: float = 0.01
    ) -> List[Tuple[float, float]]:
        """Detect pauses in speech"""
        # Frame-based energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        frames = librosa.util.frame(
            audio,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Calculate frame energy
        frame_energy = np.sqrt(np.mean(frames ** 2, axis=0))
        
        # Detect low-energy regions
        is_pause = frame_energy < energy_threshold
        
        # Group consecutive pause frames
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, p in enumerate(is_pause):
            if p and not in_pause:
                pause_start = i * hop_length / sr
                in_pause = True
            elif not p and in_pause:
                pause_end = i * hop_length / sr
                if pause_end - pause_start > 0.1:  # Minimum 100ms pause
                    pauses.append((pause_start, pause_end))
                in_pause = False
        
        return pauses
    
    def _calculate_fluency(
        self,
        pauses: List[Tuple[float, float]],
        timing_patterns: np.ndarray
    ) -> float:
        """Calculate fluency score"""
        if len(timing_patterns) == 0:
            return 0.5
        
        # Pause frequency and duration
        total_pause_duration = sum(p[1] - p[0] for p in pauses)
        pause_frequency = len(pauses)
        
        # Timing regularity
        timing_cv = np.std(timing_patterns) / (np.mean(timing_patterns) + 1e-10)
        
        # Fluency decreases with more/longer pauses and irregular timing
        fluency = 1.0
        fluency -= min(0.3, total_pause_duration / 10)  # Penalize long pauses
        fluency -= min(0.2, pause_frequency / 20)       # Penalize frequent pauses
        fluency -= min(0.3, timing_cv)                  # Penalize irregular timing
        
        return max(0.0, fluency)
    
    def _calculate_naturalness(
        self,
        isochrony: float,
        syncopation: float,
        beat_strength: float
    ) -> float:
        """Calculate naturalness score"""
        # Natural speech has moderate isochrony, low syncopation, clear beats
        naturalness = 0.0
        
        # Isochrony contribution (peak at moderate values)
        if 0.3 < isochrony < 0.7:
            naturalness += 0.4
        else:
            naturalness += 0.2
        
        # Low syncopation is more natural
        naturalness += 0.3 * (1 - min(1, syncopation))
        
        # Clear beat strength
        naturalness += 0.3 * min(1, beat_strength)
        
        return naturalness
    
    def save_models(self, path: str):
        """Save model weights"""
        torch.save({
            'prosody_model': self.prosody_model.state_dict(),
            'rhythm_model': self.rhythm_model.state_dict(),
            'prosody_scaler': self.prosody_scaler,
            'rhythm_scaler': self.rhythm_scaler
        }, path)
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.prosody_model.load_state_dict(checkpoint['prosody_model'])
        self.rhythm_model.load_state_dict(checkpoint['rhythm_model'])
        
        if 'prosody_scaler' in checkpoint:
            self.prosody_scaler = checkpoint['prosody_scaler']
        if 'rhythm_scaler' in checkpoint:
            self.rhythm_scaler = checkpoint['rhythm_scaler']
        
        logger.info(f"Models loaded from {path}")


# Example usage
def main():
    # Initialize analyzer
    analyzer = ProsodyRhythmAnalyzer()
    
    # Load test audio
    audio, sr = librosa.load("test_speech.wav", sr=16000)
    
    # Analyze prosody
    prosody = analyzer.analyze_prosody(audio, sr)
    print(f"Pitch mean: {prosody.pitch_mean:.1f} Hz")
    print(f"Intonation: {prosody.intonation_pattern}")
    print(f"Emotion: {prosody.emotional_tone}")
    print(f"Style: {prosody.speaking_style}")
    
    # Analyze rhythm
    rhythm = analyzer.analyze_rhythm(audio, sr)
    print(f"\nTempo: {rhythm.tempo:.1f} BPM")
    print(f"Rhythm class: {rhythm.rhythm_class}")
    print(f"Fluency: {rhythm.fluency_score:.2f}")
    print(f"Naturalness: {rhythm.naturalness_score:.2f}")


if __name__ == "__main__":
    main()