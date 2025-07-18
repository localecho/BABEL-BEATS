#!/usr/bin/env python3
"""
Enhanced BABEL-BEATS API with full backend integration
"""

import os
import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import json
import hashlib

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from dotenv import load_dotenv

# Import our modules
from language_processor import LanguageProcessor, LanguageFeatures
from music_generator import MusicGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./babel_beats.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# Metrics
request_count = Counter('babel_beats_requests_total', 'Total requests')
request_duration = Histogram('babel_beats_request_duration_seconds', 'Request duration')
active_users = Gauge('babel_beats_active_users', 'Active users')
generation_count = Counter('babel_beats_generations_total', 'Total music generations')

# Initialize FastAPI app
app = FastAPI(
    title="BABEL-BEATS API",
    description="Language learning through AI-generated musical patterns",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Database Models
class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(String, primary_key=True)
    user_id = Column(String)
    language = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    features = Column(JSON)
    overall_score = Column(Float)
    recommendations = Column(JSON)


class GeneratedMusic(Base):
    __tablename__ = "generated_music"
    
    id = Column(String, primary_key=True)
    analysis_id = Column(String)
    user_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String)
    metadata = Column(JSON)
    style = Column(String)
    duration = Column(Integer)


class UserProgress(Base):
    __tablename__ = "user_progress"
    
    user_id = Column(String, primary_key=True)
    language = Column(String)
    total_sessions = Column(Integer, default=0)
    total_practice_minutes = Column(Float, default=0)
    average_score = Column(Float, default=0)
    last_session = Column(DateTime)
    achievements = Column(JSON, default=[])
    streak_days = Column(Integer, default=0)


# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class AnalyzeRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio")
    language: str = Field(..., description="Target language code")
    text: Optional[str] = Field(None, description="Reference text")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis")
    user_id: Optional[str] = Field(None, description="User identifier")


class GenerateRequest(BaseModel):
    analysis_id: str = Field(..., description="Analysis result ID")
    style: str = Field(default="classical", description="Music style")
    duration: int = Field(default=30, ge=10, le=300, description="Duration in seconds")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")


class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: datetime
    language: str
    features: Dict[str, Any]
    overall_score: float
    recommendations: List[str]


class GenerationResponse(BaseModel):
    generation_id: str
    analysis_id: str
    audio_url: str
    metadata: Dict[str, Any]
    created_at: datetime


# Services
language_processor = LanguageProcessor()
music_generator = MusicGenerator()


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    
    # Validate token (implement your auth logic here)
    # For demo, just check if token exists
    if not token:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    # Get user from token
    user_id = hashlib.md5(token.encode()).hexdigest()[:8]
    return user_id


# Cache decorator
def cache_result(expiry: int = 300):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(cache_key, expiry, json.dumps(result, default=str))
            
            return result
        return wrapper
    return decorator


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "BABEL-BEATS API",
        "version": "1.0.0",
        "description": "Speak the universal language of music",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    try:
        # Check Redis
        redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow(),
        "services": {
            "database": db_status,
            "cache": redis_status,
            "language_processor": "healthy",
            "music_generator": "healthy"
        }
    }


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_speech(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Analyze speech and extract language features"""
    try:
        # Update metrics
        request_count.inc()
        
        # Process audio
        features = language_processor.analyze(
            request.audio_data,
            request.language,
            request.text
        )
        
        # Generate analysis ID
        analysis_id = f"ana_{uuid.uuid4().hex[:12]}"
        
        # Save to database
        analysis_result = AnalysisResult(
            id=analysis_id,
            user_id=user_id or request.user_id,
            language=request.language,
            features={
                "rhythm": features.rhythm,
                "tone": features.tone,
                "pronunciation": features.pronunciation
            },
            overall_score=features.overall_score,
            recommendations=features.recommendations
        )
        db.add(analysis_result)
        
        # Update user progress
        background_tasks.add_task(update_user_progress, user_id, request.language, features.overall_score, db)
        
        db.commit()
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=datetime.utcnow(),
            language=request.language,
            features={
                "rhythm": features.rhythm,
                "tone": features.tone,
                "pronunciation": features.pronunciation
            },
            overall_score=features.overall_score,
            recommendations=features.recommendations
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/generate", response_model=GenerationResponse)
async def generate_music(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Generate personalized music based on analysis"""
    try:
        # Update metrics
        generation_count.inc()
        
        # Get analysis result
        analysis = db.query(AnalysisResult).filter(
            AnalysisResult.id == request.analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Generate music
        audio, metadata = music_generator.generate(
            analysis.features,
            request.style,
            request.duration,
            request.preferences
        )
        
        # Generate ID and save audio
        generation_id = f"gen_{uuid.uuid4().hex[:12]}"
        filename = f"generated/{generation_id}.wav"
        filepath = music_generator.save_audio(audio, filename, metadata)
        
        # Save to database
        generated_music = GeneratedMusic(
            id=generation_id,
            analysis_id=request.analysis_id,
            user_id=user_id,
            file_path=filepath,
            metadata=metadata,
            style=request.style,
            duration=request.duration
        )
        db.add(generated_music)
        db.commit()
        
        # Cache the file path
        redis_client.setex(f"music:{generation_id}", 3600, filepath)
        
        return GenerationResponse(
            generation_id=generation_id,
            analysis_id=request.analysis_id,
            audio_url=f"/api/v1/audio/{generation_id}",
            metadata=metadata,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/api/v1/audio/{audio_id}")
async def get_audio(audio_id: str, user_id: str = Depends(verify_token)):
    """Retrieve generated audio file"""
    # Check cache first
    filepath = redis_client.get(f"music:{audio_id}")
    
    if not filepath:
        # Get from database
        db = SessionLocal()
        music = db.query(GeneratedMusic).filter(
            GeneratedMusic.id == audio_id
        ).first()
        db.close()
        
        if not music:
            raise HTTPException(status_code=404, detail="Audio not found")
        
        filepath = music.file_path
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(filepath, media_type="audio/wav")


@app.get("/api/v1/users/{user_id}/progress")
async def get_user_progress(
    user_id: str,
    auth_user_id: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get user's learning progress"""
    # Verify user can access this data
    if user_id != auth_user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get user progress
    progress = db.query(UserProgress).filter(
        UserProgress.user_id == user_id
    ).first()
    
    if not progress:
        return {
            "user_id": user_id,
            "overall_progress": {
                "level": 1,
                "total_practice_minutes": 0,
                "streak_days": 0,
                "languages": {}
            }
        }
    
    # Get recent analyses
    recent_analyses = db.query(AnalysisResult).filter(
        AnalysisResult.user_id == user_id
    ).order_by(AnalysisResult.created_at.desc()).limit(10).all()
    
    # Calculate statistics
    language_stats = {}
    for analysis in recent_analyses:
        lang = analysis.language
        if lang not in language_stats:
            language_stats[lang] = {
                "sessions": 0,
                "average_score": 0,
                "scores": []
            }
        language_stats[lang]["sessions"] += 1
        language_stats[lang]["scores"].append(analysis.overall_score)
    
    # Calculate averages
    for lang, stats in language_stats.items():
        stats["average_score"] = sum(stats["scores"]) / len(stats["scores"])
        del stats["scores"]
    
    return {
        "user_id": user_id,
        "overall_progress": {
            "level": calculate_level(progress.total_practice_minutes),
            "total_practice_minutes": progress.total_practice_minutes,
            "streak_days": progress.streak_days,
            "languages": language_stats
        },
        "achievements": progress.achievements or [],
        "last_session": progress.last_session
    }


@app.post("/api/v1/sessions/{session_id}/complete")
async def complete_session(
    session_id: str,
    duration_minutes: float,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Mark a learning session as complete"""
    # Update user progress
    progress = db.query(UserProgress).filter(
        UserProgress.user_id == user_id
    ).first()
    
    if not progress:
        progress = UserProgress(user_id=user_id)
        db.add(progress)
    
    progress.total_sessions += 1
    progress.total_practice_minutes += duration_minutes
    progress.last_session = datetime.utcnow()
    
    # Update streak
    if progress.last_session:
        days_since_last = (datetime.utcnow() - progress.last_session).days
        if days_since_last <= 1:
            progress.streak_days += 1
        else:
            progress.streak_days = 1
    else:
        progress.streak_days = 1
    
    db.commit()
    
    return {"status": "success", "streak_days": progress.streak_days}


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


# Helper functions
async def update_user_progress(user_id: str, language: str, score: float, db: Session):
    """Update user progress in background"""
    try:
        progress = db.query(UserProgress).filter(
            UserProgress.user_id == user_id
        ).first()
        
        if not progress:
            progress = UserProgress(
                user_id=user_id,
                language=language,
                average_score=score
            )
            db.add(progress)
        else:
            # Update rolling average
            total_score = progress.average_score * progress.total_sessions
            progress.total_sessions += 1
            progress.average_score = (total_score + score) / progress.total_sessions
        
        progress.last_session = datetime.utcnow()
        db.commit()
        
        # Update active users metric
        active_users.inc()
        
    except Exception as e:
        logger.error(f"Error updating user progress: {e}")


def calculate_level(practice_minutes: float) -> int:
    """Calculate user level based on practice time"""
    thresholds = [0, 60, 300, 600, 1200, 2400, 4800]  # Minutes
    for i, threshold in enumerate(thresholds):
        if practice_minutes < threshold:
            return i
    return len(thresholds)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return {
        "error": {
            "code": exc.status_code,
            "message": exc.detail
        },
        "request_id": str(uuid.uuid4())
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": {
            "code": 500,
            "message": "Internal server error"
        },
        "request_id": str(uuid.uuid4())
    }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("BABEL-BEATS API starting up...")
    
    # Create directories
    os.makedirs("generated", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Test database connection
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("BABEL-BEATS API shutting down...")
    
    # Reset active users
    active_users.set(0)


if __name__ == "__main__":
    uvicorn.run(
        "enhanced_main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )