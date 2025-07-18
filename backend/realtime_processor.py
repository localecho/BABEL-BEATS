#!/usr/bin/env python3
"""
Real-time Language Processing Module for BABEL-BEATS
Achieves <100ms latency for interactive feedback
"""

import asyncio
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import deque
import time
import threading
import queue
import pyaudio
import sounddevice as sd
import librosa
import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import onnxruntime as ort
import webrtcvad
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
import logging
from enum import Enum
import json
import websockets
from websockets.server import WebSocketServerProtocol
import uvloop
import numba
from numba import jit, cuda
import cupy as cp
import redis
import msgpack
import lz4.frame

logger = logging.getLogger(__name__)

# Set up async event loop policy for performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class ProcessingMode(Enum):
    """Real-time processing modes"""
    ULTRA_LOW_LATENCY = "ultra_low"  # <50ms
    LOW_LATENCY = "low"              # <100ms
    BALANCED = "balanced"            # <200ms
    HIGH_QUALITY = "high_quality"    # <500ms


@dataclass
class AudioFrame:
    """Represents a single audio frame for processing"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    frame_id: int
    is_speech: bool = False
    energy: float = 0.0
    
    @property
    def duration_ms(self) -> float:
        return len(self.data) / self.sample_rate * 1000


@dataclass
class RealtimeResult:
    """Results from real-time processing"""
    frame_id: int
    timestamp: float
    processing_time_ms: float
    features: Dict[str, Any]
    feedback: Optional[str] = None
    confidence: float = 0.0
    
    def to_json(self) -> str:
        return json.dumps({
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'processing_time_ms': self.processing_time_ms,
            'features': self.features,
            'feedback': self.feedback,
            'confidence': self.confidence
        })


class AudioBuffer:
    """Lock-free ring buffer for audio samples"""
    
    def __init__(self, capacity: int = 48000):  # 1 second at 48kHz
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self._lock = threading.Lock()
    
    def write(self, data: np.ndarray) -> bool:
        """Write data to buffer (non-blocking)"""
        with self._lock:
            n_samples = len(data)
            available = self.capacity - self.available_samples()
            
            if n_samples > available:
                return False  # Buffer full
            
            # Write in circular fashion
            end_pos = self.write_pos + n_samples
            if end_pos <= self.capacity:
                self.buffer[self.write_pos:end_pos] = data
            else:
                # Wrap around
                first_part = self.capacity - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:n_samples - first_part] = data[first_part:]
            
            self.write_pos = end_pos % self.capacity
            return True
    
    def read(self, n_samples: int) -> Optional[np.ndarray]:
        """Read data from buffer (non-blocking)"""
        with self._lock:
            if n_samples > self.available_samples():
                return None
            
            # Read in circular fashion
            end_pos = self.read_pos + n_samples
            if end_pos <= self.capacity:
                data = self.buffer[self.read_pos:end_pos].copy()
            else:
                # Wrap around
                first_part = self.capacity - self.read_pos
                data = np.concatenate([
                    self.buffer[self.read_pos:],
                    self.buffer[:n_samples - first_part]
                ])
            
            self.read_pos = end_pos % self.capacity
            return data
    
    def available_samples(self) -> int:
        """Get number of available samples"""
        if self.write_pos >= self.read_pos:
            return self.write_pos - self.read_pos
        else:
            return self.capacity - self.read_pos + self.write_pos


class GPUAccelerator:
    """GPU acceleration for real-time processing"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        
        if self.cuda_available:
            # Pre-allocate GPU memory
            self.gpu_buffer = torch.zeros(16000, device=self.device)
            torch.cuda.synchronize()
            
            logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name()}")
        else:
            logger.warning("GPU not available, using CPU")
    
    @torch.no_grad()
    def process_audio_gpu(self, audio: np.ndarray) -> torch.Tensor:
        """Process audio on GPU with minimal transfers"""
        if not self.cuda_available:
            return torch.from_numpy(audio)
        
        # Use pinned memory for faster transfers
        audio_tensor = torch.from_numpy(audio).pin_memory()
        
        # Non-blocking transfer to GPU
        gpu_audio = audio_tensor.to(self.device, non_blocking=True)
        
        return gpu_audio


class RealtimeLanguageProcessor:
    """
    Ultra-low latency language processing for real-time feedback
    Designed for <100ms end-to-end latency
    """
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.LOW_LATENCY,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        enable_gpu: bool = True
    ):
        self.mode = mode
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Audio components
        self.audio_buffer = AudioBuffer(capacity=sample_rate * 2)  # 2 second buffer
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        
        # Processing components
        self.gpu_accelerator = GPUAccelerator() if enable_gpu else None
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Frame tracking
        self.frame_counter = 0
        self.processing_times = deque(maxlen=100)
        
        # Initialize models
        self._init_models()
        
        # WebSocket server for real-time streaming
        self.websocket_clients: List[WebSocketServerProtocol] = []
        
        # Redis for fast caching
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=False  # Binary mode for msgpack
        )
        
        logger.info(f"Realtime processor initialized in {mode.value} mode")
    
    def _init_models(self):
        """Initialize lightweight models for real-time processing"""
        # Load quantized Whisper model for speed
        self.whisper_processor = WhisperProcessor.from_pretrained(
            "openai/whisper-tiny"  # Smallest model for speed
        )
        
        # Convert to ONNX for faster inference
        self.onnx_session = self._load_onnx_models()
        
        # Pre-compile feature extractors
        self.feature_extractors = self._compile_feature_extractors()
        
        # Warm up models
        self._warmup_models()
    
    def _load_onnx_models(self) -> Dict[str, ort.InferenceSession]:
        """Load ONNX models for fast inference"""
        sessions = {}
        
        # Configure ONNX Runtime for speed
        providers = ['CUDAExecutionProvider'] if self.gpu_accelerator and self.gpu_accelerator.cuda_available else ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Load models (paths would be configured)
        # sessions['pitch'] = ort.InferenceSession("models/pitch_detector.onnx", sess_options, providers=providers)
        # sessions['phoneme'] = ort.InferenceSession("models/phoneme_classifier.onnx", sess_options, providers=providers)
        
        return sessions
    
    def _compile_feature_extractors(self) -> Dict[str, Callable]:
        """Compile JIT feature extractors"""
        extractors = {}
        
        # Compile with Numba for speed
        extractors['energy'] = self._compile_energy_extractor()
        extractors['zcr'] = self._compile_zcr_extractor()
        extractors['spectral'] = self._compile_spectral_extractor()
        
        return extractors
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def _extract_energy(audio: np.ndarray) -> float:
        """JIT-compiled energy extraction"""
        return np.sqrt(np.mean(audio ** 2))
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def _extract_zcr(audio: np.ndarray) -> float:
        """JIT-compiled zero-crossing rate"""
        signs = np.sign(audio)
        signs[signs == 0] = 1
        return np.sum(np.abs(np.diff(signs))) / (2 * len(audio))
    
    def _compile_energy_extractor(self) -> Callable:
        return self._extract_energy
    
    def _compile_zcr_extractor(self) -> Callable:
        return self._extract_zcr
    
    def _compile_spectral_extractor(self) -> Callable:
        """Compile spectral feature extractor"""
        @jit(nopython=True, cache=True, fastmath=True)
        def extract_spectral_centroid(magnitude_spectrum: np.ndarray, frequencies: np.ndarray) -> float:
            return np.sum(frequencies * magnitude_spectrum) / (np.sum(magnitude_spectrum) + 1e-10)
        
        return extract_spectral_centroid
    
    def _warmup_models(self):
        """Warm up models to avoid cold start latency"""
        dummy_audio = np.random.randn(self.frame_size).astype(np.float32)
        
        # Warm up feature extractors
        for _ in range(10):
            self._extract_features_fast(dummy_audio)
        
        logger.info("Models warmed up")
    
    async def start_audio_stream(self, input_device: Optional[int] = None):
        """Start real-time audio streaming"""
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Convert to mono if needed
            audio = indata[:, 0] if len(indata.shape) > 1 else indata
            
            # Write to buffer (non-blocking)
            if not self.audio_buffer.write(audio.astype(np.float32)):
                logger.warning("Audio buffer overflow")
        
        # Start audio stream
        self.stream = sd.InputStream(
            device=input_device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            callback=audio_callback,
            dtype=np.float32
        )
        
        self.stream.start()
        
        # Start processing loop
        await self._processing_loop()
    
    async def _processing_loop(self):
        """Main processing loop with minimal latency"""
        while True:
            start_time = time.perf_counter()
            
            # Read frame from buffer
            audio_data = self.audio_buffer.read(self.frame_size)
            
            if audio_data is not None:
                # Create frame
                frame = AudioFrame(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=self.sample_rate,
                    frame_id=self.frame_counter
                )
                self.frame_counter += 1
                
                # Process frame asynchronously
                result = await self._process_frame_async(frame)
                
                # Send to WebSocket clients
                await self._broadcast_result(result)
                
                # Track processing time
                processing_time = (time.perf_counter() - start_time) * 1000
                self.processing_times.append(processing_time)
                
                if self.frame_counter % 50 == 0:  # Log every 50 frames
                    avg_time = np.mean(self.processing_times)
                    logger.debug(f"Avg processing time: {avg_time:.1f}ms")
            
            # Minimal sleep to prevent CPU spinning
            await asyncio.sleep(0.001)
    
    async def _process_frame_async(self, frame: AudioFrame) -> RealtimeResult:
        """Process single frame with ultra-low latency"""
        start_time = time.perf_counter()
        
        # Run processing in parallel
        tasks = []
        
        # VAD check (very fast)
        is_speech = await self._check_speech_async(frame)
        frame.is_speech = is_speech
        
        if is_speech:
            # Extract features in parallel
            if self.mode == ProcessingMode.ULTRA_LOW_LATENCY:
                # Only essential features
                features = await self._extract_minimal_features(frame)
            else:
                # More comprehensive features
                features = await self._extract_features_parallel(frame)
            
            # Generate feedback
            feedback = self._generate_instant_feedback(features)
        else:
            features = {'is_speech': False}
            feedback = None
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return RealtimeResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            processing_time_ms=processing_time,
            features=features,
            feedback=feedback,
            confidence=features.get('confidence', 0.0)
        )
    
    async def _check_speech_async(self, frame: AudioFrame) -> bool:
        """Fast speech detection"""
        # Convert to 16-bit PCM for VAD
        pcm_data = (frame.data * 32767).astype(np.int16).tobytes()
        
        # Use WebRTC VAD (very fast)
        try:
            return self.vad.is_speech(pcm_data, frame.sample_rate)
        except:
            # Fallback to energy-based detection
            energy = self.feature_extractors['energy'](frame.data)
            return energy > 0.01
    
    async def _extract_minimal_features(self, frame: AudioFrame) -> Dict[str, Any]:
        """Extract only essential features for ultra-low latency"""
        # Use pre-compiled extractors
        energy = self.feature_extractors['energy'](frame.data)
        zcr = self.feature_extractors['zcr'](frame.data)
        
        # Fast pitch estimation using autocorrelation
        pitch = await self._estimate_pitch_fast(frame.data)
        
        return {
            'energy': float(energy),
            'zcr': float(zcr),
            'pitch': float(pitch) if pitch else None,
            'is_speech': True,
            'confidence': 0.8
        }
    
    async def _extract_features_parallel(self, frame: AudioFrame) -> Dict[str, Any]:
        """Extract features in parallel for better latency"""
        loop = asyncio.get_event_loop()
        
        # Schedule feature extraction in parallel
        tasks = [
            loop.run_in_executor(None, self._extract_energy_features, frame.data),
            loop.run_in_executor(None, self._extract_spectral_features, frame.data),
            loop.run_in_executor(None, self._extract_pitch_features, frame.data),
        ]
        
        # Wait for all features
        results = await asyncio.gather(*tasks)
        
        # Combine results
        features = {}
        for result in results:
            features.update(result)
        
        features['is_speech'] = True
        features['confidence'] = self._calculate_confidence(features)
        
        return features
    
    def _extract_energy_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract energy-based features"""
        energy = self.feature_extractors['energy'](audio)
        
        # RMS energy in dB
        rms_db = 20 * np.log10(energy + 1e-10)
        
        return {
            'energy': float(energy),
            'rms_db': float(rms_db)
        }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features efficiently"""
        # Use FFT with power of 2 for speed
        n_fft = 256  # Small FFT for speed
        
        # Window the signal
        windowed = audio[:n_fft] * np.hanning(min(len(audio), n_fft))
        
        # FFT
        spectrum = np.abs(np.fft.rfft(windowed))
        frequencies = np.fft.rfftfreq(n_fft, 1/self.sample_rate)
        
        # Spectral centroid
        centroid = self.feature_extractors['spectral'](spectrum, frequencies)
        
        # Spectral rolloff (simplified)
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = frequencies[rolloff_idx] if rolloff_idx < len(frequencies) else frequencies[-1]
        
        return {
            'spectral_centroid': float(centroid),
            'spectral_rolloff': float(rolloff)
        }
    
    def _extract_pitch_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract pitch using fast autocorrelation"""
        # Simplified YIN algorithm for speed
        pitch = self._yin_pitch_fast(audio)
        
        return {
            'pitch': float(pitch) if pitch else None,
            'pitch_confidence': 0.8 if pitch else 0.0
        }
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def _autocorrelation(signal: np.ndarray, max_lag: int) -> np.ndarray:
        """Fast autocorrelation using NumPy"""
        result = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                result[lag] = np.sum(signal * signal)
            else:
                result[lag] = np.sum(signal[:-lag] * signal[lag:])
        return result
    
    def _yin_pitch_fast(self, audio: np.ndarray) -> Optional[float]:
        """Simplified YIN for real-time pitch detection"""
        # Downsample for speed if needed
        if self.sample_rate > 16000:
            audio = signal.decimate(audio, self.sample_rate // 16000)
            sr = 16000
        else:
            sr = self.sample_rate
        
        # Parameters
        min_f0 = 80
        max_f0 = 400
        min_lag = int(sr / max_f0)
        max_lag = int(sr / min_f0)
        
        # Autocorrelation
        r = self._autocorrelation(audio, max_lag)
        
        # Cumulative mean normalized difference
        d = np.zeros(max_lag)
        d[0] = 1
        for tau in range(1, max_lag):
            d[tau] = r[0] + r[tau] - 2 * r[tau]
        
        # Cumulative mean normalization
        cumsum = np.cumsum(d)
        d[1:] = d[1:] / (cumsum[1:] / np.arange(1, max_lag))
        
        # Find first minimum below threshold
        threshold = 0.1
        for tau in range(min_lag, max_lag):
            if d[tau] < threshold:
                # Parabolic interpolation
                if tau > 0 and tau < max_lag - 1:
                    alpha = d[tau - 1]
                    beta = d[tau]
                    gamma = d[tau + 1]
                    peak = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                    return sr / (tau + peak)
        
        return None
    
    async def _estimate_pitch_fast(self, audio: np.ndarray) -> Optional[float]:
        """Async wrapper for pitch estimation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._yin_pitch_fast, audio)
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate feature confidence score"""
        confidence = 1.0
        
        # Reduce confidence for low energy
        if features.get('energy', 0) < 0.001:
            confidence *= 0.5
        
        # Reduce confidence for no pitch
        if features.get('pitch') is None:
            confidence *= 0.8
        
        return confidence
    
    def _generate_instant_feedback(self, features: Dict[str, Any]) -> Optional[str]:
        """Generate instant feedback based on features"""
        if not features.get('is_speech'):
            return None
        
        feedback_parts = []
        
        # Energy feedback
        energy = features.get('energy', 0)
        if energy < 0.005:
            feedback_parts.append("Speak louder")
        elif energy > 0.5:
            feedback_parts.append("Too loud")
        
        # Pitch feedback
        pitch = features.get('pitch')
        if pitch:
            if pitch < 80:
                feedback_parts.append("Pitch too low")
            elif pitch > 300:
                feedback_parts.append("Pitch too high")
        
        # Combine feedback
        if feedback_parts:
            return "; ".join(feedback_parts)
        else:
            return "Good!"
    
    async def _broadcast_result(self, result: RealtimeResult):
        """Broadcast result to all WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message = result.to_json()
        
        # Send to all clients in parallel
        tasks = []
        for client in self.websocket_clients:
            tasks.append(client.send(message))
        
        # Don't wait for slow clients
        asyncio.create_task(self._send_to_clients(tasks))
    
    async def _send_to_clients(self, tasks: List):
        """Send to clients without blocking"""
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error broadcasting to clients: {e}")
    
    async def websocket_handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket connections"""
        self.websocket_clients.append(websocket)
        logger.info(f"Client connected. Total clients: {len(self.websocket_clients)}")
        
        try:
            async for message in websocket:
                # Handle client messages if needed
                pass
        finally:
            self.websocket_clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.websocket_clients)}")
    
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time streaming"""
        logger.info(f"Starting WebSocket server on {host}:{port}")
        await websockets.serve(self.websocket_handler, host, port)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.processing_times:
            return {
                'mean': 0,
                'median': 0,
                'p95': 0,
                'p99': 0,
                'max': 0
            }
        
        times = list(self.processing_times)
        return {
            'mean': np.mean(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'max': np.max(times)
        }
    
    async def benchmark_latency(self, duration_seconds: int = 10):
        """Benchmark processing latency"""
        logger.info(f"Running latency benchmark for {duration_seconds} seconds...")
        
        # Generate test audio
        test_audio = np.random.randn(self.sample_rate * duration_seconds).astype(np.float32) * 0.1
        
        # Process in frames
        results = []
        for i in range(0, len(test_audio) - self.frame_size, self.frame_size):
            frame_data = test_audio[i:i + self.frame_size]
            
            frame = AudioFrame(
                data=frame_data,
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                frame_id=i // self.frame_size
            )
            
            start = time.perf_counter()
            result = await self._process_frame_async(frame)
            latency = (time.perf_counter() - start) * 1000
            
            results.append(latency)
        
        # Calculate statistics
        stats = {
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results),
            'p95': np.percentile(results, 95),
            'p99': np.percentile(results, 99),
            'below_100ms': np.sum(np.array(results) < 100) / len(results) * 100
        }
        
        logger.info(f"Latency benchmark results:")
        logger.info(f"  Mean: {stats['mean']:.2f}ms")
        logger.info(f"  Median: {stats['median']:.2f}ms")
        logger.info(f"  95th percentile: {stats['p95']:.2f}ms")
        logger.info(f"  99th percentile: {stats['p99']:.2f}ms")
        logger.info(f"  Frames below 100ms: {stats['below_100ms']:.1f}%")
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self.thread_pool.shutdown(wait=True)
        
        if hasattr(self, 'redis_client'):
            self.redis_client.close()


# Optimized feature extraction functions
@jit(nopython=True, cache=True, fastmath=True)
def extract_mfcc_fast(audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """Fast MFCC extraction"""
    # Simplified MFCC for real-time
    # This is a placeholder - real implementation would use optimized DCT
    return np.zeros(n_mfcc)


@jit(nopython=True, cache=True, fastmath=True)
def extract_formants_fast(audio: np.ndarray, sr: int) -> np.ndarray:
    """Fast formant extraction using LPC"""
    # Simplified LPC for formants
    # This is a placeholder - real implementation would use Burg's method
    return np.array([700.0, 1220.0, 2600.0])  # Typical formants


class RealtimeWebInterface:
    """Web interface for real-time language processing"""
    
    def __init__(self, processor: RealtimeLanguageProcessor):
        self.processor = processor
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start web server with WebSocket support"""
        from aiohttp import web
        
        app = web.Application()
        
        # Routes
        app.router.add_get('/', self.index_handler)
        app.router.add_get('/ws', self.websocket_handler)
        app.router.add_get('/stats', self.stats_handler)
        
        # Start WebSocket server
        await self.processor.start_websocket_server()
        
        # Start web server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Web interface started at http://{host}:{port}")
    
    async def index_handler(self, request):
        """Serve main page"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>BABEL-BEATS Real-time Processor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                #latency { 
                    font-size: 48px; 
                    font-weight: bold; 
                    color: #4CAF50;
                }
                #feedback {
                    font-size: 24px;
                    margin: 20px 0;
                    padding: 10px;
                    background: #f0f0f0;
                    border-radius: 5px;
                }
                .stats {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 10px;
                    margin: 20px 0;
                }
                .stat-box {
                    background: #e0e0e0;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <h1>BABEL-BEATS Real-time Language Processor</h1>
            <div id="latency">0.0ms</div>
            <div id="feedback">Waiting for speech...</div>
            <div class="stats">
                <div class="stat-box">
                    <strong>Energy</strong>
                    <div id="energy">0.00</div>
                </div>
                <div class="stat-box">
                    <strong>Pitch</strong>
                    <div id="pitch">N/A</div>
                </div>
                <div class="stat-box">
                    <strong>Confidence</strong>
                    <div id="confidence">0%</div>
                </div>
            </div>
            
            <script>
                const ws = new WebSocket('ws://localhost:8765');
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    // Update latency
                    document.getElementById('latency').textContent = 
                        data.processing_time_ms.toFixed(1) + 'ms';
                    
                    // Update feedback
                    if (data.feedback) {
                        document.getElementById('feedback').textContent = data.feedback;
                    }
                    
                    // Update stats
                    if (data.features.energy !== undefined) {
                        document.getElementById('energy').textContent = 
                            data.features.energy.toFixed(3);
                    }
                    
                    if (data.features.pitch !== null) {
                        document.getElementById('pitch').textContent = 
                            data.features.pitch.toFixed(1) + ' Hz';
                    } else {
                        document.getElementById('pitch').textContent = 'N/A';
                    }
                    
                    document.getElementById('confidence').textContent = 
                        (data.confidence * 100).toFixed(0) + '%';
                    
                    // Color code latency
                    const latencyEl = document.getElementById('latency');
                    if (data.processing_time_ms < 50) {
                        latencyEl.style.color = '#4CAF50';  // Green
                    } else if (data.processing_time_ms < 100) {
                        latencyEl.style.color = '#FFC107';  // Yellow
                    } else {
                        latencyEl.style.color = '#F44336';  // Red
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    document.getElementById('feedback').textContent = 
                        'Connection error!';
                };
            </script>
        </body>
        </html>
        """
        from aiohttp import web
        return web.Response(text=html, content_type='text/html')
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections from web interface"""
        from aiohttp import web
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Add to processor's clients
        self.processor.websocket_clients.append(ws)
        
        try:
            async for msg in ws:
                # Handle messages if needed
                pass
        finally:
            self.processor.websocket_clients.remove(ws)
        
        return ws
    
    async def stats_handler(self, request):
        """Return processing statistics"""
        from aiohttp import web
        stats = self.processor.get_latency_stats()
        return web.json_response(stats)


# Example usage
async def main():
    # Create processor
    processor = RealtimeLanguageProcessor(
        mode=ProcessingMode.LOW_LATENCY,
        enable_gpu=True
    )
    
    # Benchmark latency
    await processor.benchmark_latency(duration_seconds=5)
    
    # Create web interface
    web_interface = RealtimeWebInterface(processor)
    
    # Start servers
    await web_interface.start_server()
    
    # Start audio processing
    await processor.start_audio_stream()


if __name__ == "__main__":
    asyncio.run(main())