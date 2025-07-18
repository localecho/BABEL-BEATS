#!/usr/bin/env python3
"""
Local testing script for BABEL-BEATS
Test the application locally without external dependencies
"""

import asyncio
import numpy as np
import base64
import json
from datetime import datetime

# Generate test audio data
def generate_test_audio(duration=2.0, sample_rate=16000):
    """Generate synthetic audio for testing"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a simple sine wave with some noise
    frequency = 440  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    # Add some noise
    audio += 0.1 * np.random.randn(len(audio))
    return audio.astype(np.float32)

# Test language processor
def test_basic_analysis():
    """Test basic language analysis"""
    print("\n🧪 Testing Basic Language Analysis...")
    
    try:
        from backend.language_processor import LanguageProcessor
        
        processor = LanguageProcessor()
        audio = generate_test_audio()
        
        # Convert to base64
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        # Test analysis
        result = processor.analyze(audio_base64, "en-US", "Hello world")
        
        print("✅ Basic Analysis Results:")
        print(f"   - Rhythm consistency: {result.rhythm['rhythm_consistency']:.2f}")
        print(f"   - Overall score: {result.overall_score:.2f}")
        print(f"   - Recommendations: {result.recommendations[:2]}")
        
        return True
    except Exception as e:
        print(f"❌ Basic analysis failed: {e}")
        return False

# Test advanced features
async def test_advanced_features():
    """Test advanced language processing features"""
    print("\n🧪 Testing Advanced Features...")
    
    try:
        from backend.advanced_language_processor import AdvancedLanguageProcessor
        
        processor = AdvancedLanguageProcessor()
        await processor.warmup()
        
        audio = generate_test_audio()
        
        # Test comprehensive analysis
        print("   Testing Whisper ASR integration...")
        features = await processor.analyze_comprehensive(
            audio,
            language="en",
            reference_text="Hello world"
        )
        
        print("✅ Advanced Analysis Results:")
        print(f"   - Transcription: {features.transcription}")
        print(f"   - Pronunciation score: {features.pronunciation_score:.2f}")
        print(f"   - Processing time: {features.processing_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Advanced analysis failed: {e}")
        print("   Note: This requires Whisper model to be downloaded")
        return False

# Test real-time processing
async def test_realtime_processing():
    """Test real-time processing capabilities"""
    print("\n🧪 Testing Real-time Processing...")
    
    try:
        from backend.realtime_processor import RealtimeLanguageProcessor, ProcessingMode
        
        processor = RealtimeLanguageProcessor(
            mode=ProcessingMode.LOW_LATENCY,
            enable_gpu=False  # Use CPU for testing
        )
        
        # Benchmark latency
        stats = await processor.benchmark_latency(duration_seconds=2)
        
        print("✅ Real-time Processing Results:")
        print(f"   - Mean latency: {stats['mean']:.2f}ms")
        print(f"   - 95th percentile: {stats['p95']:.2f}ms")
        print(f"   - Frames below 100ms: {stats['below_100ms']:.1f}%")
        
        processor.cleanup()
        return True
    except Exception as e:
        print(f"❌ Real-time processing failed: {e}")
        return False

# Test prosody analysis
def test_prosody_analysis():
    """Test prosody and rhythm analysis"""
    print("\n🧪 Testing Prosody Analysis...")
    
    try:
        from backend.prosody_rhythm_models import ProsodyRhythmAnalyzer
        
        analyzer = ProsodyRhythmAnalyzer(device="cpu")
        audio = generate_test_audio()
        
        # Analyze prosody
        prosody = analyzer.analyze_prosody(audio, 16000)
        rhythm = analyzer.analyze_rhythm(audio, 16000)
        
        print("✅ Prosody Analysis Results:")
        print(f"   - Pitch mean: {prosody.pitch_mean:.1f} Hz")
        print(f"   - Speaking style: {prosody.speaking_style}")
        print(f"   - Tempo: {rhythm.tempo:.1f} BPM")
        print(f"   - Rhythm class: {rhythm.rhythm_class}")
        
        return True
    except Exception as e:
        print(f"❌ Prosody analysis failed: {e}")
        return False

# Test API endpoints
async def test_api_endpoints():
    """Test FastAPI endpoints locally"""
    print("\n🧪 Testing API Endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        print("✅ Root endpoint: OK")
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        health = response.json()
        print(f"✅ Health check: {health['status']}")
        
        # Test analysis endpoint
        audio = generate_test_audio(0.5)
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        response = client.post("/analyze", json={
            "audio_base64": audio_base64,
            "language": "en-US",
            "text": "Test"
        })
        
        if response.status_code == 200:
            print("✅ Analysis endpoint: OK")
        else:
            print(f"⚠️  Analysis endpoint returned: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ API testing failed: {e}")
        return False

# Test music generation
def test_music_generation():
    """Test music generation module"""
    print("\n🧪 Testing Music Generation...")
    
    try:
        from backend.music_generator import MusicGenerator
        
        generator = MusicGenerator()
        
        # Test with mock features
        features = {
            "rhythm": {"tempo": 120, "beat_strength": 0.8},
            "tone": {"pitch_mean": 220, "pitch_range": 50},
            "pronunciation": {"phoneme_accuracy": 0.85}
        }
        
        print("✅ Music generator initialized")
        print("   Note: Actual generation requires more setup")
        
        return True
    except Exception as e:
        print(f"❌ Music generation failed: {e}")
        return False

# Main test runner
async def run_all_tests():
    """Run all local tests"""
    print("🎵 BABEL-BEATS Local Testing Suite")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(test_basic_analysis())
    results.append(await test_advanced_features())
    results.append(await test_realtime_processing())
    results.append(test_prosody_analysis())
    results.append(await test_api_endpoints())
    results.append(test_music_generation())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("   ✅ All tests passed!")
    else:
        print(f"   ⚠️  {total - passed} tests failed")
    
    return passed == total

# Echo test for audio
def echo_test():
    """Simple echo test to verify audio processing"""
    print("\n🎤 Audio Echo Test")
    print("=" * 50)
    
    # Generate test audio
    print("Generating test audio...")
    audio = generate_test_audio(duration=1.0)
    
    # Process through basic pipeline
    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"Sample rate: 16000 Hz")
    
    # Simulate echo
    delay_samples = int(0.2 * 16000)  # 200ms delay
    echo = np.zeros(len(audio) + delay_samples)
    echo[:len(audio)] = audio
    echo[delay_samples:] += audio * 0.5  # 50% echo
    
    print(f"Echo added with 200ms delay")
    print("✅ Echo test complete")

if __name__ == "__main__":
    print("\n🚀 Starting BABEL-BEATS Local Tests...\n")
    
    # Run echo test first
    echo_test()
    
    # Run all tests
    asyncio.run(run_all_tests())
    
    print("\n💡 To start the full application:")
    print("   python main.py")
    print("\n📖 See README.md for more information")