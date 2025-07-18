# BABEL-BEATS Gradio Frontend

This is a simple Gradio interface for the BABEL-BEATS language learning application that provides an easy-to-use web interface for testing the advanced speech analysis and music generation features.

## Features

- **🎤 Speech Analysis**: Record audio and get comprehensive pronunciation, rhythm, and tone feedback
- **🔤 Phoneme Analysis**: Detailed phoneme-by-phoneme alignment analysis
- **🎵 Music Generation**: Automatically generate personalized learning music based on speech patterns
- **🌍 Multi-language Support**: Support for 10+ languages including Spanish, French, Japanese, German, Korean, and more
- **📊 Real-time Feedback**: Get instant analysis results with detailed scoring and recommendations

## Quick Start

### 1. Start the Backend Server

First, make sure the BABEL-BEATS backend is running:

```bash
cd BABEL-BEATS
python main.py
```

The backend should be accessible at `http://localhost:8000`

### 2. Launch the Gradio Interface

In a new terminal, run:

```bash
cd BABEL-BEATS
python gradio_frontend.py
```

The Gradio interface will be available at:
- **Local**: `http://localhost:7860`
- **Network**: `http://0.0.0.0:7860` (accessible from other devices on your network)

## Usage Guide

### Speech Analysis Tab

1. **Select Language**: Choose your target language from the dropdown
2. **Record Audio**: Click the microphone button and speak clearly
3. **Add Reference Text** (optional): Enter the text you spoke for better analysis
4. **Analyze**: Click "Analyze Speech" to get comprehensive feedback

You'll receive:
- Pronunciation, rhythm, and tone scores
- Detailed feedback and recommendations
- Technical analysis details
- Automatically generated personalized learning music

### Phoneme Analysis Tab

1. **Record Audio**: Speak the phrase you want to analyze
2. **Enter Reference Text**: This is required for phoneme alignment
3. **Analyze**: Get detailed phoneme-by-phoneme breakdown with timing and confidence scores

## API Integration

The Gradio frontend connects to these BABEL-BEATS API endpoints:

- `POST /analyze/advanced` - Advanced speech analysis
- `POST /phoneme/align` - Phoneme alignment analysis  
- `POST /music/generate` - Personalized music generation
- `GET /health` - Backend health check

## Example Phrases

Try these phrases in different languages:

**Spanish**: "Hola, ¿cómo estás? Me llamo María."

**French**: "Bonjour, comment allez-vous? Je suis ravi de vous rencontrer."

**Japanese**: "こんにちは、元気ですか？私の名前は田中です。"

**German**: "Guten Tag, wie geht es Ihnen? Ich freue mich, Sie kennenzulernen."

## Troubleshooting

### Backend Connection Issues

If you see "Backend Status: Unhealthy", make sure:

1. The backend server is running on `localhost:8000`
2. All backend dependencies are installed
3. No firewall is blocking the connection

### Audio Recording Issues

- Ensure your browser has microphone permissions
- Use a quiet environment for best results
- Speak clearly and at a natural pace

## Technical Details

- **Frontend**: Gradio 5.38.0
- **Backend**: FastAPI with advanced language processing
- **Audio Processing**: librosa, soundfile
- **Languages**: 100+ supported via OpenAI Whisper
- **Real-time Processing**: WebSocket support for <100ms latency

## Development

The Gradio frontend is designed to complement the existing HTML/JavaScript frontend, providing:

- Rapid prototyping and testing interface
- Easy integration with Jupyter notebooks
- Simple deployment for demos and research
- Clean API for programmatic access

Both frontends can run simultaneously, giving you flexibility in how you interact with the BABEL-BEATS system.
