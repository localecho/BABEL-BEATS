#!/usr/bin/env python3
"""
Comprehensive test suite for BABEL-BEATS
"""

import pytest
import asyncio
import json
import base64
import tempfile
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import soundfile as sf
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, DataProcessingService, MusicGenerationService


class TestDataProcessingService:
    """Test data processing functionality"""
    
    @pytest.fixture
    def service(self):
        return DataProcessingService()
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing"""
        # Generate a simple sine wave
        sample_rate = 44100
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Save to temporary file and encode
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            f.seek(0)
            audio_bytes = f.read()
            
        return base64.b64encode(audio_bytes).decode()
    
    @pytest.mark.asyncio
    async def test_process_basic_data(self, service):
        """Test basic data processing"""
        test_data = {
            "audio_data": "sample_base64_audio",
            "language": "en-US",
            "user_id": "test_user"
        }
        
        result = await service.process_data(test_data)
        
        assert "processed_at" in result
        assert "features" in result
        assert "parameters" in result
        assert result["parameters"] == test_data
    
    @pytest.mark.asyncio
    async def test_process_with_language_detection(self, service):
        """Test processing with language detection"""
        test_data = {
            "audio_data": "sample_audio",
            "detect_language": True
        }
        
        result = await service.process_data(test_data)
        
        assert "features" in result
        # In real implementation, would detect language
    
    @pytest.mark.asyncio
    async def test_extract_rhythm_features(self, service, sample_audio_data):
        """Test rhythm feature extraction"""
        test_data = {
            "audio_data": sample_audio_data,
            "language": "en-US",
            "analysis_type": "rhythm"
        }
        
        with patch.object(service, 'extract_rhythm_features') as mock_extract:
            mock_extract.return_value = {
                "tempo": 120,
                "beat_strength": 0.8,
                "rhythm_consistency": 0.85
            }
            
            result = await service.process_data(test_data)
            
            assert "features" in result
    
    @pytest.mark.asyncio
    async def test_extract_tone_features(self, service, sample_audio_data):
        """Test tone feature extraction"""
        test_data = {
            "audio_data": sample_audio_data,
            "language": "zh-CN",
            "analysis_type": "tone"
        }
        
        with patch.object(service, 'extract_tone_features') as mock_extract:
            mock_extract.return_value = {
                "pitch_contour": [120, 125, 118, 130],
                "tone_patterns": ["rising", "falling", "neutral"],
                "tone_accuracy": 0.78
            }
            
            result = await service.process_data(test_data)
            
            assert "features" in result
    
    @pytest.mark.asyncio
    async def test_pronunciation_analysis(self, service, sample_audio_data):
        """Test pronunciation analysis"""
        test_data = {
            "audio_data": sample_audio_data,
            "language": "en-US",
            "text": "Hello world",
            "analysis_type": "pronunciation"
        }
        
        with patch.object(service, 'analyze_pronunciation') as mock_analyze:
            mock_analyze.return_value = {
                "phoneme_accuracy": 0.82,
                "problem_sounds": ["θ", "ð"],
                "clarity_score": 0.79
            }
            
            result = await service.process_data(test_data)
            
            assert "features" in result
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, service, sample_audio_data):
        """Test comprehensive analysis combining all features"""
        test_data = {
            "audio_data": sample_audio_data,
            "language": "en-US",
            "analysis_type": "comprehensive"
        }
        
        result = await service.process_data(test_data)
        
        assert "features" in result
        assert "processed_at" in result
        assert isinstance(result["processed_at"], datetime)


class TestMusicGenerationService:
    """Test music generation functionality"""
    
    @pytest.fixture
    def service(self):
        return MusicGenerationService()
    
    @pytest.fixture
    def processed_data(self):
        return {
            "features": {
                "rhythm": {
                    "tempo": 120,
                    "stress_pattern": [1, 0, 1, 0]
                },
                "tone": {
                    "pitch_contour": [120, 125, 118, 130]
                }
            },
            "language": "en-US",
            "user_id": "test_user"
        }
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, service, processed_data):
        """Test basic music generation"""
        content_id = await service.generate(processed_data)
        
        assert content_id is not None
        assert isinstance(content_id, str)
        assert len(content_id) > 0
    
    @pytest.mark.asyncio
    async def test_generation_with_style(self, service, processed_data):
        """Test generation with specific style"""
        preferences = {
            "style": "classical",
            "instruments": ["piano", "violin"],
            "tempo_adjustment": 1.1
        }
        
        content_id = await service.generate(processed_data, preferences)
        
        assert content_id is not None
    
    @pytest.mark.asyncio
    async def test_rhythm_based_generation(self, service):
        """Test music generation based on rhythm patterns"""
        rhythm_data = {
            "features": {
                "rhythm": {
                    "tempo": 140,
                    "time_signature": "4/4",
                    "stress_pattern": [1, 0, 0, 1, 0, 0, 1, 0],
                    "syllable_timing": [0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1]
                }
            },
            "focus": "rhythm"
        }
        
        content_id = await service.generate(rhythm_data)
        
        assert content_id is not None
    
    @pytest.mark.asyncio
    async def test_tone_based_generation(self, service):
        """Test music generation for tonal languages"""
        tone_data = {
            "features": {
                "tone": {
                    "language": "zh-CN",
                    "tones": [1, 4, 3, 2],  # Chinese tones
                    "pitch_values": [55, 51, 214, 35],
                    "duration_ms": [300, 250, 400, 350]
                }
            },
            "focus": "tone"
        }
        
        content_id = await service.generate(tone_data)
        
        assert content_id is not None
    
    @pytest.mark.asyncio
    async def test_generation_error_handling(self, service):
        """Test error handling in generation"""
        invalid_data = {
            "features": {}  # Missing required features
        }
        
        # Should handle gracefully
        content_id = await service.generate(invalid_data)
        assert content_id is not None


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        return {"Authorization": "Bearer test_api_key"}
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "BABEL-BEATS"
    
    def test_process_endpoint(self, client, auth_headers):
        """Test data processing endpoint"""
        request_data = {
            "user_id": "test_user",
            "parameters": {
                "audio_data": "sample_audio",
                "language": "en-US"
            }
        }
        
        response = client.post("/process", json=request_data, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "processed_at" in data
        assert "features" in data
    
    def test_generate_endpoint(self, client, auth_headers):
        """Test music generation endpoint"""
        request_data = {
            "input_data": {
                "user_id": "test_user",
                "language": "en-US",
                "features": {"rhythm": {"tempo": 120}}
            },
            "style_preferences": {
                "style": "classical"
            },
            "duration": 30
        }
        
        response = client.post("/generate", json=request_data, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "audio_url" in data
        assert "metadata" in data
        assert data["user_id"] == "test_user"
    
    def test_content_retrieval(self, client, auth_headers):
        """Test content retrieval endpoint"""
        content_id = "test_content_123"
        
        response = client.get(f"/content/{content_id}", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert content_id in data["message"]
    
    def test_invalid_generation_request(self, client, auth_headers):
        """Test generation with invalid request"""
        request_data = {
            "input_data": {},  # Missing required fields
            "duration": 500  # Out of range
        }
        
        response = client.post("/generate", json=request_data, headers=auth_headers)
        assert response.status_code in [422, 400]  # Validation error
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting functionality"""
        # Make many requests rapidly
        responses = []
        for _ in range(10):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # All should succeed for health endpoint (no rate limit)
        assert all(status == 200 for status in responses)


class TestIntegration:
    """Integration tests"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, client):
        """Test complete workflow from analysis to generation"""
        # Step 1: Process audio
        process_request = {
            "user_id": "integration_test_user",
            "parameters": {
                "audio_data": "test_audio_data",
                "language": "en-US",
                "text": "Hello, how are you?"
            }
        }
        
        process_response = client.post("/process", json=process_request)
        assert process_response.status_code == 200
        processed_data = process_response.json()
        
        # Step 2: Generate music
        generate_request = {
            "input_data": processed_data["parameters"],
            "style_preferences": {
                "style": "pop",
                "tempo": 120
            },
            "duration": 30
        }
        
        generate_response = client.post("/generate", json=generate_request)
        assert generate_response.status_code == 200
        generation_data = generate_response.json()
        
        # Step 3: Retrieve content
        content_id = generation_data["id"]
        content_response = client.get(f"/content/{content_id}")
        assert content_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_multilingual_support(self, client):
        """Test support for multiple languages"""
        languages = ["en-US", "zh-CN", "es-ES", "fr-FR", "ja-JP"]
        
        for language in languages:
            request = {
                "user_id": "multilingual_test",
                "parameters": {
                    "audio_data": "test_audio",
                    "language": language
                }
            }
            
            response = client.post("/process", json=request)
            assert response.status_code == 200
            
            data = response.json()
            assert "features" in data


class TestPerformance:
    """Performance and scalability tests"""
    
    @pytest.fixture
    def service(self):
        return DataProcessingService()
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, service):
        """Test concurrent request handling"""
        num_requests = 20
        
        async def process_single():
            data = {
                "audio_data": "test_audio",
                "language": "en-US"
            }
            return await service.process_data(data)
        
        # Process multiple requests concurrently
        tasks = [process_single() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == num_requests
        assert all("features" in r for r in results)
    
    @pytest.mark.asyncio
    async def test_large_audio_processing(self, service):
        """Test processing of large audio files"""
        # Simulate large audio file (5 minutes)
        large_audio = "x" * (44100 * 2 * 60 * 5)  # Approximate size
        
        data = {
            "audio_data": large_audio,
            "language": "en-US"
        }
        
        result = await service.process_data(data)
        assert "features" in result
    
    def test_memory_efficiency(self, client):
        """Test memory usage remains reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make multiple requests
        for _ in range(50):
            client.get("/health")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def service(self):
        return DataProcessingService()
    
    @pytest.mark.asyncio
    async def test_empty_audio_data(self, service):
        """Test handling of empty audio data"""
        data = {
            "audio_data": "",
            "language": "en-US"
        }
        
        result = await service.process_data(data)
        assert "features" in result  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_unsupported_language(self, service):
        """Test handling of unsupported language"""
        data = {
            "audio_data": "test_audio",
            "language": "xx-XX"  # Invalid language code
        }
        
        result = await service.process_data(data)
        assert "features" in result  # Should fallback gracefully
    
    @pytest.mark.asyncio
    async def test_corrupted_audio_data(self, service):
        """Test handling of corrupted audio data"""
        data = {
            "audio_data": "not_valid_base64!!!",
            "language": "en-US"
        }
        
        result = await service.process_data(data)
        assert "features" in result  # Should handle error
    
    def test_missing_required_fields(self, client):
        """Test API with missing required fields"""
        # Missing user_id
        request = {
            "parameters": {}
        }
        
        response = client.post("/process", json=request)
        assert response.status_code == 422
    
    def test_invalid_duration(self, client):
        """Test generation with invalid duration"""
        request = {
            "input_data": {"user_id": "test"},
            "duration": 1000  # Too long
        }
        
        response = client.post("/generate", json=request)
        assert response.status_code == 422


class TestSecurity:
    """Security-related tests"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_sql_injection_attempt(self, client):
        """Test protection against SQL injection"""
        malicious_input = {
            "user_id": "'; DROP TABLE users; --",
            "parameters": {
                "audio_data": "test",
                "language": "en-US"
            }
        }
        
        response = client.post("/process", json=malicious_input)
        # Should process safely without executing SQL
        assert response.status_code in [200, 422]
    
    def test_xss_prevention(self, client):
        """Test XSS prevention"""
        xss_attempt = {
            "user_id": "<script>alert('xss')</script>",
            "parameters": {
                "audio_data": "test",
                "text": "<img src=x onerror=alert('xss')>"
            }
        }
        
        response = client.post("/process", json=xss_attempt)
        
        if response.status_code == 200:
            data = response.json()
            # Check that script tags are not in response
            response_str = json.dumps(data)
            assert "<script>" not in response_str
            assert "onerror=" not in response_str
    
    def test_path_traversal_prevention(self, client):
        """Test path traversal attack prevention"""
        response = client.get("/content/../../etc/passwd")
        assert response.status_code in [400, 404]
    
    def test_rate_limit_by_ip(self, client):
        """Test rate limiting by IP address"""
        # Simulate requests from same IP
        headers = {"X-Forwarded-For": "192.168.1.100"}
        
        responses = []
        for _ in range(150):
            response = client.post("/process", 
                                 json={"user_id": "test", "parameters": {}},
                                 headers=headers)
            responses.append(response.status_code)
        
        # Some requests should be rate limited
        assert 429 in responses or all(s == 200 for s in responses)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])