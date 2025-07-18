# BABEL-BEATS API Documentation

## Overview

BABEL-BEATS is an innovative language learning platform that leverages AI-generated musical patterns to enhance pronunciation, rhythm, and tonal language acquisition. This API enables developers to integrate language-music synthesis capabilities into their applications.

## Base URL

```
https://api.babel-beats.ai/v1
```

For local development:
```
http://localhost:8000
```

## Authentication

All API requests require authentication using API keys.

```http
Authorization: Bearer YOUR_API_KEY
```

## Rate Limiting

- **Free tier**: 100 requests per hour
- **Pro tier**: 1,000 requests per hour
- **Enterprise**: Unlimited (contact sales)

## Endpoints

### 1. Language Analysis

#### Analyze Speech Pattern

```http
POST /api/v1/analyze
```

Analyzes speech input to extract rhythm, tone, and pronunciation features.

**Request Body:**
```json
{
  "audio_data": "base64_encoded_audio",
  "language": "en-US",
  "analysis_type": "comprehensive",
  "user_id": "user123",
  "metadata": {
    "session_id": "sess_abc123",
    "skill_level": "beginner"
  }
}
```

**Parameters:**
- `audio_data` (required): Base64 encoded audio file (WAV, MP3, or M4A)
- `language` (required): Language code (e.g., "en-US", "zh-CN", "es-ES")
- `analysis_type` (optional): Type of analysis ["basic", "comprehensive", "rhythm_only", "tone_only"]
- `user_id` (optional): User identifier for personalization
- `metadata` (optional): Additional context information

**Response:**
```json
{
  "analysis_id": "ana_123456",
  "timestamp": "2024-01-15T10:30:00Z",
  "language": "en-US",
  "features": {
    "rhythm": {
      "tempo": 120,
      "stress_pattern": [1, 0, 1, 0],
      "syllable_timing": [0.2, 0.15, 0.25, 0.18],
      "consistency_score": 0.85
    },
    "tone": {
      "pitch_contour": [120, 125, 118, 130],
      "tone_accuracy": 0.78,
      "intonation_pattern": "rising"
    },
    "pronunciation": {
      "phoneme_accuracy": 0.82,
      "problem_sounds": ["θ", "ð", "r"],
      "clarity_score": 0.79
    }
  },
  "recommendations": [
    "Focus on 'th' sounds",
    "Practice rising intonation patterns",
    "Work on consistent rhythm"
  ]
}
```

### 2. Music Generation

#### Generate Language-Based Music

```http
POST /api/v1/generate
```

Creates musical patterns based on language features.

**Request Body:**
```json
{
  "analysis_id": "ana_123456",
  "music_style": "classical",
  "duration": 30,
  "complexity": "medium",
  "focus_areas": ["rhythm", "tone"],
  "personalization": {
    "tempo_preference": 120,
    "instrument_preferences": ["piano", "strings"]
  }
}
```

**Parameters:**
- `analysis_id` (required): ID from previous analysis
- `music_style` (optional): Style of music ["classical", "pop", "jazz", "traditional", "electronic"]
- `duration` (optional): Duration in seconds (10-300)
- `complexity` (optional): Complexity level ["simple", "medium", "advanced"]
- `focus_areas` (optional): Array of focus areas ["rhythm", "tone", "pronunciation", "flow"]
- `personalization` (optional): User preferences

**Response:**
```json
{
  "generation_id": "gen_789012",
  "audio_url": "https://cdn.babel-beats.ai/audio/gen_789012.mp3",
  "sheet_music_url": "https://cdn.babel-beats.ai/sheets/gen_789012.pdf",
  "metadata": {
    "duration": 30,
    "tempo": 120,
    "key": "C major",
    "time_signature": "4/4",
    "instruments": ["piano", "violin"]
  },
  "learning_elements": {
    "rhythm_patterns": [
      {
        "measure": 1,
        "pattern": "quarter-eighth-eighth-quarter",
        "maps_to": "Hello how are you"
      }
    ],
    "tone_markers": [
      {
        "time": 2.5,
        "pitch_change": "rising",
        "linguistic_element": "question intonation"
      }
    ]
  }
}
```

### 3. Learning Sessions

#### Create Learning Session

```http
POST /api/v1/sessions
```

Creates a structured learning session combining analysis and music generation.

**Request Body:**
```json
{
  "user_id": "user123",
  "language": "en-US",
  "target_language": "zh-CN",
  "session_type": "pronunciation",
  "difficulty": "beginner",
  "duration_minutes": 15,
  "focus_areas": ["tones", "rhythm"]
}
```

**Response:**
```json
{
  "session_id": "sess_345678",
  "exercises": [
    {
      "exercise_id": "ex_001",
      "type": "listen_and_repeat",
      "audio_url": "https://cdn.babel-beats.ai/exercises/ex_001.mp3",
      "instructions": "Listen to the musical pattern and repeat the phrase",
      "target_phrase": "你好吗",
      "phonetic": "nǐ hǎo ma"
    },
    {
      "exercise_id": "ex_002",
      "type": "rhythm_matching",
      "audio_url": "https://cdn.babel-beats.ai/exercises/ex_002.mp3",
      "instructions": "Match your speech rhythm to the musical beat"
    }
  ],
  "estimated_duration": 15,
  "learning_objectives": [
    "Master tones 2 and 3",
    "Improve syllable timing"
  ]
}
```

#### Submit Exercise Results

```http
POST /api/v1/sessions/{session_id}/exercises/{exercise_id}/submit
```

Submits user's attempt for evaluation.

**Request Body:**
```json
{
  "audio_data": "base64_encoded_audio",
  "attempt_number": 1,
  "metadata": {
    "device": "mobile",
    "environment": "quiet"
  }
}
```

**Response:**
```json
{
  "evaluation_id": "eval_901234",
  "score": 85,
  "feedback": {
    "strengths": ["Good tone accuracy", "Consistent rhythm"],
    "improvements": ["Work on the third tone fall-rise pattern"],
    "detailed_analysis": {
      "tone_accuracy": 0.88,
      "rhythm_accuracy": 0.82,
      "overall_fluency": 0.85
    }
  },
  "next_exercise_id": "ex_003",
  "achievement_unlocked": "Tone Master Level 1"
}
```

### 4. Progress Tracking

#### Get User Progress

```http
GET /api/v1/users/{user_id}/progress
```

Retrieves comprehensive progress data for a user.

**Response:**
```json
{
  "user_id": "user123",
  "overall_progress": {
    "level": 3,
    "total_practice_minutes": 450,
    "streak_days": 15,
    "languages": {
      "zh-CN": {
        "proficiency": 0.65,
        "total_sessions": 30,
        "average_score": 82
      }
    }
  },
  "skill_breakdown": {
    "pronunciation": {
      "current_level": 0.78,
      "improvement_rate": 0.02,
      "problem_areas": ["retroflex sounds", "aspirated consonants"]
    },
    "rhythm": {
      "current_level": 0.85,
      "improvement_rate": 0.03,
      "strengths": ["consistent tempo", "stress patterns"]
    },
    "tone": {
      "current_level": 0.72,
      "improvement_rate": 0.04,
      "mastered_patterns": ["tone 1", "tone 4"],
      "learning_patterns": ["tone 2", "tone 3"]
    }
  },
  "achievements": [
    {
      "id": "ach_001",
      "name": "Rhythm Master",
      "earned_date": "2024-01-10"
    }
  ],
  "recommendations": [
    "Practice tone sandhi rules",
    "Focus on sentence-level intonation"
  ]
}
```

### 5. Cultural Music Library

#### Browse Cultural Music

```http
GET /api/v1/library/cultural
```

Access culturally relevant music for language learning.

**Query Parameters:**
- `language`: Target language code
- `genre`: Music genre
- `difficulty`: Difficulty level
- `page`: Page number
- `limit`: Results per page

**Response:**
```json
{
  "items": [
    {
      "id": "cm_001",
      "title": "春天在哪里",
      "english_title": "Where is Spring",
      "language": "zh-CN",
      "genre": "children's song",
      "difficulty": "beginner",
      "audio_url": "https://cdn.babel-beats.ai/cultural/cm_001.mp3",
      "lyrics_url": "https://cdn.babel-beats.ai/lyrics/cm_001.pdf",
      "learning_points": ["tone pairs", "question words"],
      "cultural_notes": "Popular children's song teaching seasons"
    }
  ],
  "pagination": {
    "page": 1,
    "total_pages": 10,
    "total_items": 98
  }
}
```

### 6. Collaboration

#### Share Learning Content

```http
POST /api/v1/share
```

Share generated music or learning sessions with others.

**Request Body:**
```json
{
  "content_type": "generation",
  "content_id": "gen_789012",
  "share_with": ["user456", "user789"],
  "permissions": ["view", "practice"],
  "message": "Check out this pattern for practicing tones!"
}
```

**Response:**
```json
{
  "share_id": "shr_567890",
  "share_url": "https://babel-beats.ai/share/shr_567890",
  "recipients_notified": ["user456", "user789"],
  "expires_at": "2024-02-15T10:30:00Z"
}
```

### 7. Export and Integration

#### Export Learning Data

```http
POST /api/v1/export
```

Export user data in various formats.

**Request Body:**
```json
{
  "user_id": "user123",
  "format": "pdf",
  "content_types": ["progress", "generated_music", "practice_recordings"],
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-01-31"
  }
}
```

**Response:**
```json
{
  "export_id": "exp_123456",
  "status": "processing",
  "estimated_time": 120,
  "webhook_url": "https://yourapp.com/webhook/export_complete"
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_AUDIO_FORMAT",
    "message": "The provided audio format is not supported",
    "details": {
      "supported_formats": ["wav", "mp3", "m4a"],
      "provided_format": "wma"
    }
  },
  "request_id": "req_abc123"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INVALID_AUDIO_FORMAT` | 400 | Unsupported audio format |
| `LANGUAGE_NOT_SUPPORTED` | 400 | Requested language not available |
| `INSUFFICIENT_AUDIO_QUALITY` | 400 | Audio quality too low for analysis |
| `GENERATION_FAILED` | 500 | Music generation process failed |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource doesn't exist |

## Webhooks

### Configure Webhooks

```http
POST /api/v1/webhooks
```

Set up webhooks for asynchronous events.

**Request Body:**
```json
{
  "url": "https://yourapp.com/webhook",
  "events": ["generation.complete", "export.ready", "achievement.earned"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload

```json
{
  "event": "generation.complete",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "generation_id": "gen_789012",
    "user_id": "user123",
    "audio_url": "https://cdn.babel-beats.ai/audio/gen_789012.mp3"
  },
  "signature": "sha256=abcdef123456..."
}
```

## Code Examples

### Python

```python
import requests
import base64

# Initialize client
API_KEY = "your_api_key"
BASE_URL = "https://api.babel-beats.ai/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Analyze speech
with open("speech.wav", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode()

response = requests.post(
    f"{BASE_URL}/analyze",
    headers=headers,
    json={
        "audio_data": audio_data,
        "language": "zh-CN",
        "analysis_type": "comprehensive"
    }
)

analysis = response.json()
print(f"Analysis ID: {analysis['analysis_id']}")

# Generate music based on analysis
response = requests.post(
    f"{BASE_URL}/generate",
    headers=headers,
    json={
        "analysis_id": analysis['analysis_id'],
        "music_style": "classical",
        "duration": 30,
        "focus_areas": ["tone", "rhythm"]
    }
)

generation = response.json()
print(f"Generated music: {generation['audio_url']}")
```

### JavaScript

```javascript
const BABEL_BEATS_API = {
  key: 'your_api_key',
  baseUrl: 'https://api.babel-beats.ai/v1'
};

// Analyze speech
async function analyzeSpeech(audioBlob) {
  const base64Audio = await blobToBase64(audioBlob);
  
  const response = await fetch(`${BABEL_BEATS_API.baseUrl}/analyze`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${BABEL_BEATS_API.key}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      audio_data: base64Audio,
      language: 'en-US',
      analysis_type: 'comprehensive'
    })
  });
  
  return response.json();
}

// Generate music
async function generateMusic(analysisId) {
  const response = await fetch(`${BABEL_BEATS_API.baseUrl}/generate`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${BABEL_BEATS_API.key}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      analysis_id: analysisId,
      music_style: 'pop',
      duration: 30,
      focus_areas: ['pronunciation', 'rhythm']
    })
  });
  
  return response.json();
}
```

## Best Practices

1. **Audio Quality**: Ensure audio recordings are clear with minimal background noise
2. **Language Codes**: Use standard ISO 639-1 language codes
3. **Rate Limiting**: Implement exponential backoff for retries
4. **Caching**: Cache generated content to reduce API calls
5. **Error Handling**: Always handle errors gracefully and provide user feedback
6. **Security**: Never expose API keys in client-side code

## Support

- **Documentation**: https://docs.babel-beats.ai
- **Status Page**: https://status.babel-beats.ai
- **Support Email**: support@babel-beats.ai
- **Developer Forum**: https://forum.babel-beats.ai