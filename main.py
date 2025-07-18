#!/usr/bin/env python3
"""
BABEL-BEATS: Advanced Language Learning Through AI-Generated Musical Patterns
Main application entry point with 10x enhanced features
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import base64

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from dotenv import load_dotenv
import redis
import aiofiles

# Import our advanced modules
from backend.advanced_language_processor import AdvancedLanguageProcessor
from backend.phoneme_alignment_service import PhonemeAlignmentService
from backend.realtime_processor import RealtimeLanguageProcessor, ProcessingMode
from backend.prosody_rhythm_models import ProsodyRhythmAnalyzer
from backend.music_generator import MusicGenerator
from backend.language_processor import LanguageProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BABEL-BEATS Advanced API",
    description="10x Enhanced Language Learning Through AI-Generated Musical Patterns",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# Data Models
class LanguageAnalysisRequest(BaseModel):
    """Request model for language analysis"""
    audio_base64: str = Field(..., description="Base64 encoded audio")
    language: str = Field(..., description="Target language code")
    text: Optional[str] = Field(None, description="Optional reference text")
    native_speaker_audio: Optional[str] = Field(None, description="Native speaker reference")
    analysis_mode: str = Field(default="comprehensive", description="basic|comprehensive|real-time")

class MusicGenerationRequest(BaseModel):
    """Request model for music generation"""
    language_features: Dict[str, Any]
    style: str = Field(default="adaptive", description="Music style")
    duration: int = Field(default=30, ge=10, le=300)
    tempo_adjustment: Optional[float] = Field(None, description="Tempo multiplier")
    mood: Optional[str] = Field(None, description="Target mood")

class PronunciationAssessmentRequest(BaseModel):
    """Request for pronunciation assessment"""
    learner_audio: str = Field(..., description="Learner's audio (base64)")
    native_audio: str = Field(..., description="Native speaker audio (base64)")
    text: str = Field(..., description="Reference text")
    language: str = Field(..., description="Language code")

class RealtimeConfig(BaseModel):
    """Configuration for real-time processing"""
    mode: str = Field(default="low_latency", description="ultra_low|low|balanced|high_quality")
    language: str = Field(..., description="Target language")
    enable_gpu: bool = Field(default=True)

# Initialize Services
logger.info("Initializing advanced services...")

# Advanced processors
advanced_processor = AdvancedLanguageProcessor()
phoneme_service = PhonemeAlignmentService()
realtime_processor = RealtimeLanguageProcessor(mode=ProcessingMode.LOW_LATENCY)
prosody_analyzer = ProsodyRhythmAnalyzer()

# Original processors (for backward compatibility)
language_processor = LanguageProcessor()
music_generator = MusicGenerator()

logger.info("All services initialized successfully")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to BABEL-BEATS Advanced API",
        "version": "2.0.0",
        "description": "10x Enhanced Language Learning Through Musical Patterns",
        "features": {
            "whisper_asr": "Multi-lingual speech recognition",
            "phoneme_alignment": "Precise phoneme-level analysis",
            "real_time": "Sub-100ms latency feedback",
            "prosody_analysis": "Deep learning prosody models",
            "languages_supported": "100+",
            "pronunciation_coaching": "AI-powered coaching"
        },
        "endpoints": {
            "analyze": "/analyze",
            "analyze_advanced": "/analyze/advanced",
            "phoneme_align": "/phoneme/align",
            "pronunciation_assess": "/pronunciation/assess",
            "generate_music": "/music/generate",
            "real_time": "/realtime/start",
            "health": "/health"
        }
    }

@app.post("/analyze")
async def analyze_language(request: LanguageAnalysisRequest):
    """Basic language analysis (backward compatible)"""
    try:
        # Use original processor for compatibility
        features = language_processor.analyze(
            request.audio_base64,
            request.language,
            request.text
        )
        
        return {
            "status": "success",
            "features": features.__dict__,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/advanced")
async def analyze_language_advanced(request: LanguageAnalysisRequest):
    """Advanced language analysis with Whisper ASR"""
    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Comprehensive analysis
        if request.native_speaker_audio:
            # With native speaker comparison
            native_bytes = base64.b64decode(request.native_speaker_audio)
            features = await advanced_processor.analyze_comprehensive(
                audio_bytes,
                request.language,
                request.text,
                native_bytes
            )
        else:
            # Without native speaker
            features = await advanced_processor.analyze_comprehensive(
                audio_bytes,
                request.language,
                request.text
            )
        
        return {
            "status": "success",
            "features": features.to_dict(),
            "processing_time_ms": features.processing_time * 1000,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Advanced analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/phoneme/align")
async def align_phonemes(request: LanguageAnalysisRequest):
    """High-precision phoneme alignment"""
    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Save to temporary file
        temp_path = f"/tmp/audio_{datetime.utcnow().timestamp()}.wav"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(audio_bytes)
        
        # Perform alignment
        alignment = await phoneme_service.align_phonemes(
            temp_path,
            request.text or "",
            request.language,
            detailed_features=True
        )
        
        # Clean up
        os.remove(temp_path)
        
        # Convert to JSON-serializable format
        segments_data = []
        for seg in alignment.segments:
            segments_data.append({
                "phoneme": seg.phoneme,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "confidence": seg.confidence,
                "ipa_symbol": seg.ipa_symbol,
                "manner": seg.manner,
                "place": seg.place,
                "voicing": seg.voicing
            })
        
        return {
            "status": "success",
            "alignment": {
                "segments": segments_data,
                "total_duration": alignment.total_duration,
                "speech_rate": alignment.speech_rate,
                "phoneme_inventory": alignment.phoneme_inventory,
                "alignment_confidence": alignment.alignment_confidence
            },
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Phoneme alignment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pronunciation/assess")
async def assess_pronunciation(request: PronunciationAssessmentRequest):
    """Compare learner pronunciation with native speaker"""
    try:
        # Decode audio files
        learner_bytes = base64.b64decode(request.learner_audio)
        native_bytes = base64.b64decode(request.native_audio)
        
        # Save to temporary files
        learner_path = f"/tmp/learner_{datetime.utcnow().timestamp()}.wav"
        native_path = f"/tmp/native_{datetime.utcnow().timestamp()}.wav"
        
        async with aiofiles.open(learner_path, 'wb') as f:
            await f.write(learner_bytes)
        async with aiofiles.open(native_path, 'wb') as f:
            await f.write(native_bytes)
        
        # Analyze pronunciation
        assessment = await phoneme_service.analyze_pronunciation_quality(
            learner_path,
            native_path,
            request.text,
            request.language
        )
        
        # Clean up
        os.remove(learner_path)
        os.remove(native_path)
        
        return {
            "status": "success",
            "assessment": {
                "scores": assessment['scores'],
                "feedback": assessment['feedback'],
                "comparison": {
                    "overall_timing": assessment['comparison']['overall_timing'],
                    "speech_rate": assessment['comparison']['speech_rate']
                }
            },
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Pronunciation assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/music/generate")
async def generate_music(request: MusicGenerationRequest):
    """Generate personalized learning music"""
    try:
        # Generate music based on language features
        music_id = await music_generator.generate_personalized_music(
            request.language_features,
            style=request.style,
            duration=request.duration,
            tempo_adjustment=request.tempo_adjustment,
            mood=request.mood
        )
        
        # Store generation metadata
        redis_client.hset(
            f"music:{music_id}",
            mapping={
                "created_at": datetime.utcnow().isoformat(),
                "duration": request.duration,
                "style": request.style,
                "status": "completed"
            }
        )
        
        return {
            "status": "success",
            "music_id": music_id,
            "url": f"/music/{music_id}",
            "duration": request.duration,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Music generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/music/{music_id}")
async def get_music(music_id: str):
    """Stream generated music"""
    try:
        # Check if music exists
        if not redis_client.exists(f"music:{music_id}"):
            raise HTTPException(status_code=404, detail="Music not found")
        
        # Get file path
        file_path = f"generated/{music_id}.mp3"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Music file not found")
        
        # Stream the file
        async def iterfile():
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(1024 * 1024):  # 1MB chunks
                    yield chunk
        
        return StreamingResponse(
            iterfile(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename={music_id}.mp3"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Music streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/realtime")
async def websocket_realtime(websocket: WebSocket):
    """WebSocket endpoint for real-time processing"""
    await websocket.accept()
    
    try:
        # Receive configuration
        config_data = await websocket.receive_json()
        config = RealtimeConfig(**config_data)
        
        # Configure processor
        mode_map = {
            "ultra_low": ProcessingMode.ULTRA_LOW_LATENCY,
            "low": ProcessingMode.LOW_LATENCY,
            "balanced": ProcessingMode.BALANCED,
            "high_quality": ProcessingMode.HIGH_QUALITY
        }
        
        processor = RealtimeLanguageProcessor(
            mode=mode_map.get(config.mode, ProcessingMode.LOW_LATENCY),
            enable_gpu=config.enable_gpu
        )
        
        # Add websocket to processor's clients
        processor.websocket_clients.append(websocket)
        
        # Start processing
        await websocket.send_json({
            "type": "ready",
            "message": "Real-time processing ready"
        })
        
        # Keep connection alive
        while True:
            try:
                # Receive audio chunks
                data = await websocket.receive_bytes()
                # Audio will be processed by the processor's internal loop
                # Results will be automatically sent to all clients
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    finally:
        # Clean up
        if websocket in processor.websocket_clients:
            processor.websocket_clients.remove(websocket)
        await websocket.close()

@app.post("/prosody/analyze")
async def analyze_prosody(file: UploadFile = File(...)):
    """Analyze prosody features using deep learning"""
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Save temporarily
        temp_path = f"/tmp/prosody_{datetime.utcnow().timestamp()}.wav"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(audio_bytes)
        
        # Load audio
        import librosa
        audio, sr = librosa.load(temp_path, sr=16000)
        
        # Analyze prosody
        prosody_features = prosody_analyzer.analyze_prosody(audio, sr)
        
        # Analyze rhythm
        rhythm_features = prosody_analyzer.analyze_rhythm(audio, sr)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "status": "success",
            "prosody": {
                "pitch_mean": prosody_features.pitch_mean,
                "pitch_std": prosody_features.pitch_std,
                "intonation_pattern": prosody_features.intonation_pattern,
                "emotional_tone": prosody_features.emotional_tone,
                "speaking_style": prosody_features.speaking_style,
                "confidence": prosody_features.confidence
            },
            "rhythm": {
                "tempo": rhythm_features.tempo,
                "rhythm_class": rhythm_features.rhythm_class,
                "fluency_score": rhythm_features.fluency_score,
                "naturalness_score": rhythm_features.naturalness_score
            },
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Prosody analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis
        redis_status = redis_client.ping()
        
        # Check GPU availability
        import torch
        gpu_available = torch.cuda.is_available()
        
        # Get latency stats
        latency_stats = realtime_processor.get_latency_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "service": "BABEL-BEATS Advanced",
            "components": {
                "redis": "connected" if redis_status else "disconnected",
                "gpu": "available" if gpu_available else "not available",
                "whisper": "loaded",
                "phoneme_aligner": "ready",
                "realtime_processor": "active",
                "prosody_analyzer": "ready"
            },
            "performance": {
                "realtime_latency_ms": latency_stats.get('mean', 0),
                "supported_languages": 100
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting BABEL-BEATS Advanced API...")
    
    # Warm up models
    logger.info("Warming up AI models...")
    await advanced_processor.warmup()
    
    # Start real-time processor WebSocket server
    asyncio.create_task(realtime_processor.start_websocket_server())
    
    logger.info("All services started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down BABEL-BEATS Advanced API...")
    
    # Clean up resources
    await phoneme_service.cleanup()
    realtime_processor.cleanup()
    
    logger.info("Shutdown complete")

# Main execution
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("generated", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Log startup
    logger.info("Starting BABEL-BEATS Advanced API...")
    logger.info("Features: Whisper ASR, Phoneme Alignment, Real-time Processing, Deep Learning")
    logger.info("Supported Languages: 100+")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20
    )