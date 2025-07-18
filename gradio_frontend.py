#!/usr/bin/env python3
"""
BABEL-BEATS Gradio Frontend
Simple interface for language learning through musical patterns
"""

import gradio as gr
import requests
import base64
import json
import io
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

LANGUAGES = {
    "🇨🇳 Mandarin Chinese": "zh-CN",
    "🇪🇸 Spanish": "es-ES", 
    "🇫🇷 French": "fr-FR",
    "🇯🇵 Japanese": "ja-JP",
    "🇩🇪 German": "de-DE",
    "🇰🇷 Korean": "ko-KR",
    "🇮🇹 Italian": "it-IT",
    "🇵🇹 Portuguese": "pt-PT",
    "🇷🇺 Russian": "ru-RU",
    "🇦🇷 Arabic": "ar-SA"
}

def audio_to_base64(audio_data: Tuple[int, np.ndarray]) -> str:
    """Convert audio data to base64 string"""
    try:
        sample_rate, audio_array = audio_data
        
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        buffer.seek(0)
        
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return audio_base64
    except Exception as e:
        logger.error(f"Error converting audio to base64: {e}")
        raise

def check_backend_health() -> Dict[str, Any]:
    """Check if the backend is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}

def analyze_speech(audio_data: Optional[Tuple[int, np.ndarray]], 
                  language: str, 
                  reference_text: str = "") -> Tuple[str, str, str, str]:
    """Analyze speech using the BABEL-BEATS backend"""
    
    if audio_data is None:
        return "❌ No audio recorded", "", "", ""
    
    try:
        health = check_backend_health()
        if health.get("status") != "healthy":
            return f"❌ Backend Error: {health.get('error', 'Unknown error')}", "", "", ""
        
        audio_base64 = audio_to_base64(audio_data)
        
        language_code = LANGUAGES.get(language, "en-US")
        
        request_data = {
            "audio_base64": audio_base64,
            "language": language_code,
            "text": reference_text if reference_text.strip() else None,
            "analysis_mode": "comprehensive"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/analyze/advanced",
            json=request_data,
            timeout=30
        )
        
        if response.status_code != 200:
            return f"❌ API Error: {response.status_code} - {response.text}", "", "", ""
        
        result = response.json()
        features = result.get("features", {})
        
        pronunciation_score = features.get("pronunciation_score", 0) * 100
        rhythm_score = features.get("rhythm_score", 0) * 100
        tone_score = features.get("tone_score", 0) * 100
        
        analysis_summary = f"""

**Language:** {language}
**Processing Time:** {result.get('processing_time_ms', 0):.1f}ms

- **Pronunciation:** {pronunciation_score:.1f}%
- **Rhythm & Flow:** {rhythm_score:.1f}%
- **Tone Accuracy:** {tone_score:.1f}%

- **Transcription:** {features.get('transcription', 'N/A')}
- **Confidence:** {features.get('confidence', 0):.2f}
- **Speech Rate:** {features.get('speech_rate', 0):.1f} words/min
"""
        
        feedback = features.get('feedback', {})
        detailed_feedback = f"""

**Pronunciation:** {feedback.get('pronunciation', 'Good overall pronunciation')}

**Rhythm:** {feedback.get('rhythm', 'Natural speech rhythm detected')}

**Tone:** {feedback.get('tone', 'Appropriate tonal patterns')}

**Recommendations:** {', '.join(feedback.get('recommendations', ['Keep practicing!']))}
"""
        
        technical_info = f"""

**Audio Quality:** {features.get('audio_quality', 'Good')}
**Background Noise:** {features.get('noise_level', 'Low')}
**Voice Activity:** {features.get('voice_activity_ratio', 0.8):.2f}
**Pitch Range:** {features.get('pitch_range', [0, 0])}
**Energy:** {features.get('energy_stats', {})}
"""
        
        music_info = generate_music_suggestion(features, language_code)
        
        return analysis_summary, detailed_feedback, technical_info, music_info
        
    except Exception as e:
        logger.error(f"Error analyzing speech: {e}")
        return f"❌ Error: {str(e)}", "", "", ""

def generate_music_suggestion(features: Dict[str, Any], language_code: str) -> str:
    """Generate personalized music based on analysis"""
    try:
        music_request = {
            "language_features": features,
            "style": "adaptive",
            "duration": 30,
            "mood": "encouraging"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/music/generate",
            json=music_request,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            music_id = result.get("music_id", "")
            
            return f"""

**Music ID:** {music_id}
**Style:** Adaptive to your speech patterns
**Duration:** 30 seconds
**Focus Areas:** Rhythm, Tone, Pronunciation

*Music has been generated based on your speech analysis. Use it to practice speaking in rhythm!*

**Download URL:** {API_BASE_URL}/music/{music_id}
"""
        else:
            return "🎵 Music generation temporarily unavailable"
            
    except Exception as e:
        logger.error(f"Error generating music: {e}")
        return f"🎵 Music generation error: {str(e)}"

def get_phoneme_alignment(audio_data: Optional[Tuple[int, np.ndarray]], 
                         text: str, 
                         language: str) -> str:
    """Get detailed phoneme alignment analysis"""
    
    if audio_data is None or not text.strip():
        return "❌ Please provide both audio and reference text"
    
    try:
        audio_base64 = audio_to_base64(audio_data)
        language_code = LANGUAGES.get(language, "en-US")
        
        request_data = {
            "audio_base64": audio_base64,
            "language": language_code,
            "text": text.strip()
        }
        
        response = requests.post(
            f"{API_BASE_URL}/phoneme/align",
            json=request_data,
            timeout=30
        )
        
        if response.status_code != 200:
            return f"❌ Phoneme alignment error: {response.status_code}"
        
        result = response.json()
        alignment = result.get("alignment", {})
        segments = alignment.get("segments", [])
        
        phoneme_info = f"""

**Total Duration:** {alignment.get('total_duration', 0):.2f}s
**Speech Rate:** {alignment.get('speech_rate', 0):.1f} phonemes/sec
**Alignment Confidence:** {alignment.get('alignment_confidence', 0):.2f}

"""
        
        for i, segment in enumerate(segments[:20]):  # Show first 20 phonemes
            phoneme_info += f"""
**{i+1}.** `{segment.get('phoneme', '')}` ({segment.get('ipa_symbol', '')})
- Time: {segment.get('start_time', 0):.2f}s - {segment.get('end_time', 0):.2f}s
- Duration: {segment.get('duration', 0):.2f}s
- Confidence: {segment.get('confidence', 0):.2f}
- Features: {segment.get('manner', '')} {segment.get('place', '')} {segment.get('voicing', '')}
"""
        
        if len(segments) > 20:
            phoneme_info += f"\n*... and {len(segments) - 20} more phonemes*"
        
        return phoneme_info
        
    except Exception as e:
        logger.error(f"Error in phoneme alignment: {e}")
        return f"❌ Error: {str(e)}"

def create_gradio_interface():
    """Create the main Gradio interface"""
    
    css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .tab-nav {
        background: #f7fafc;
        border-radius: 8px;
    }
    """
    
    with gr.Blocks(css=css, title="BABEL-BEATS - Language Learning Through Music") as interface:
        
        gr.HTML("""
        <div class="header">
            <h1>🎵 BABEL-BEATS</h1>
            <p>Learn Languages Through the Universal Language of Music</p>
            <p><em>Advanced AI-powered speech analysis with personalized music generation</em></p>
        </div>
        """)
        
        with gr.Row():
            status_display = gr.HTML()
            
            def update_status():
                health = check_backend_health()
                if health.get("status") == "healthy":
                    return "🟢 Backend Status: Healthy ✅"
                else:
                    return f"🔴 Backend Status: {health.get('error', 'Unknown error')} ❌"
            
            interface.load(update_status, outputs=status_display)
        
        with gr.Tabs():
            with gr.TabItem("🎤 Speech Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        language_dropdown = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="🇪🇸 Spanish",
                            label="🌍 Target Language"
                        )
                        
                        audio_input = gr.Audio(
                            label="🎙️ Record Your Voice",
                            type="numpy"
                        )
                        
                        reference_text = gr.Textbox(
                            label="📝 Reference Text (Optional)",
                            placeholder="Enter the text you're trying to pronounce..."
                        )
                        
                        analyze_btn = gr.Button("🎯 Analyze Speech", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        analysis_output = gr.Markdown(label="📊 Analysis Results")
                        feedback_output = gr.Markdown(label="💡 Detailed Feedback")
                
                with gr.Row():
                    technical_output = gr.Markdown(label="🔬 Technical Details")
                    music_output = gr.Markdown(label="🎵 Personalized Music")
                
                analyze_btn.click(
                    fn=analyze_speech,
                    inputs=[audio_input, language_dropdown, reference_text],
                    outputs=[analysis_output, feedback_output, technical_output, music_output]
                )
            
            with gr.TabItem("🔤 Phoneme Analysis"):
                with gr.Row():
                    with gr.Column():
                        phoneme_language = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="🇪🇸 Spanish",
                            label="🌍 Language"
                        )
                        
                        phoneme_audio = gr.Audio(
                            label="🎙️ Audio for Phoneme Analysis",
                            type="numpy"
                        )
                        
                        phoneme_text = gr.Textbox(
                            label="📝 Reference Text (Required)",
                            placeholder="Enter the exact text you spoke..."
                        )
                        
                        phoneme_btn = gr.Button("🔤 Analyze Phonemes", variant="primary")
                    
                    with gr.Column():
                        phoneme_output = gr.Markdown(label="🔤 Phoneme Alignment Results")
                
                phoneme_btn.click(
                    fn=get_phoneme_alignment,
                    inputs=[phoneme_audio, phoneme_text, phoneme_language],
                    outputs=phoneme_output
                )
            
            with gr.TabItem("📚 API Information"):
                gr.Markdown("""
                
                This Gradio interface connects to the BABEL-BEATS FastAPI backend running on `localhost:8000`.
                
                
                - **`POST /analyze/advanced`** - Advanced speech analysis with Whisper ASR
                - **`POST /phoneme/align`** - High-precision phoneme alignment  
                - **`POST /pronunciation/assess`** - Compare with native speakers
                - **`POST /music/generate`** - Generate personalized learning music
                - **`WS /realtime`** - Real-time WebSocket processing
                - **`GET /health`** - Backend health check
                
                
                - **100+ Languages** supported via OpenAI Whisper
                - **Real-time feedback** with <100ms latency
                - **Phoneme-level precision** using Montreal Forced Aligner
                - **AI music generation** based on speech patterns
                - **Comprehensive scoring** for pronunciation, rhythm, and tone
                
                
                ```bash
                cd BABEL-BEATS
                python main.py
                ```
                
                The backend should be running on `http://localhost:8000` for this interface to work properly.
                """)
            
            with gr.TabItem("💡 Examples"):
                gr.Markdown("""
                
                1. Select your target language from the dropdown
                2. Click the microphone and record yourself speaking
                3. Optionally enter the reference text you spoke
                4. Click "Analyze Speech" to get comprehensive feedback
                
                1. Record audio of yourself speaking
                2. Enter the exact text you spoke (required)
                3. Get detailed phoneme-by-phoneme breakdown
                
                - After speech analysis, personalized music is automatically generated
                - The music helps you internalize rhythm and tone patterns
                - Use the provided download link to save your learning music
                
                
                **Spanish:** "Hola, ¿cómo estás? Me llamo María."
                
                **French:** "Bonjour, comment allez-vous? Je suis ravi de vous rencontrer."
                
                **Japanese:** "こんにちは、元気ですか？私の名前は田中です。"
                
                **German:** "Guten Tag, wie geht es Ihnen? Ich freue mich, Sie kennenzulernen."
                
                - Speak clearly and at a natural pace
                - Record in a quiet environment
                - Provide reference text when possible
                - Try different languages to see how the analysis adapts
                """)
    
    return interface

def main():
    """Main function to launch the Gradio interface"""
    
    print("🔍 Checking backend status...")
    health = check_backend_health()
    
    if health.get("status") == "healthy":
        print("✅ Backend is healthy and ready!")
        print(f"🎯 Backend info: {health}")
    else:
        print(f"⚠️  Backend warning: {health.get('error')}")
        print("💡 Make sure to start the backend with: python main.py")
    
    print("🚀 Launching BABEL-BEATS Gradio Interface...")
    
    interface = create_gradio_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
