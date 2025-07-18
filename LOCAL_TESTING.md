# 🧪 BABEL-BEATS Local Testing Guide

This guide helps you test BABEL-BEATS locally without setting up the full environment.

## 🎯 Quick Echo Test

The simplest way to see the echo effect in action:

```bash
python simple_echo_demo.py
```

This demonstrates how echo works with:
- Visual waveform display (if matplotlib is installed)
- ASCII art visualization
- Interactive parameter adjustment

## 🔧 Component Testing

To test all BABEL-BEATS components:

```bash
# Make script executable (first time only)
chmod +x run_local_tests.sh

# Run all tests
./run_local_tests.sh
```

Or test individual components:

```bash
# Test basic language analysis
python local_test.py

# Test echo server (requires pyaudio)
python local_echo_server.py

# Simple echo demo (no dependencies)
python simple_echo_demo.py
```

## 🎤 Echo Server Features

The local echo server (`local_echo_server.py`) provides:
- Real-time audio input/output with echo effect
- Adjustable delay (50-1000ms)
- Adjustable decay (0.1-0.9)
- Optional reverb effect
- Pitch shifting
- WebSocket control interface
- Latency monitoring

### Echo Server Commands:
- `delay <ms>` - Set echo delay in milliseconds
- `decay <0-1>` - Set echo decay factor
- `reverb` - Toggle reverb effect
- `pitch <value>` - Set pitch shift factor
- `stats` - Show performance statistics
- `quit` - Exit the server

## 📊 Test Coverage

The local tests cover:
1. **Basic Language Analysis** - Rhythm, tone, pronunciation scoring
2. **Advanced Features** - Whisper ASR integration (if models available)
3. **Real-time Processing** - Latency benchmarking
4. **Prosody Analysis** - Pitch and rhythm detection
5. **API Endpoints** - FastAPI route testing
6. **Music Generation** - Generator initialization

## 🚀 Full Application

To run the complete BABEL-BEATS application:

```bash
# Install all dependencies
pip install -r requirements-advanced.txt

# Start Redis
redis-server

# Run the application
python main.py
```

Then access:
- Web interface: http://localhost:8000
- API docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8765

## 💡 Troubleshooting

### Missing Dependencies
If you get import errors:
```bash
pip install numpy scipy librosa
```

### No Audio Input/Output
The echo server requires PyAudio:
```bash
# macOS
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev
pip install pyaudio

# Windows
pip install pyaudio
```

### Can't Install All Requirements
Start with basic requirements:
```bash
pip install -r requirements.txt
```

## 🎵 Echo Algorithm

The echo effect works by:
1. Capturing audio input
2. Storing it in a circular buffer
3. Mixing delayed audio with current input
4. Applying decay factor to create natural echo

Example with 200ms delay and 0.5 decay:
```
Original: [1.0, 0.5, -0.5, -1.0, 0.0]
With Echo: [1.0, 0.5, 0.5, -0.25, -0.25, -0.5, 0.0]
           └─────┘   └── echo ──┘
```

## 🔊 Testing Without Audio Hardware

Use `simple_echo_demo.py` to:
- Visualize echo effects
- Understand the algorithm
- Experiment with parameters
- No audio hardware needed!

---

Happy testing! 🎵🧪