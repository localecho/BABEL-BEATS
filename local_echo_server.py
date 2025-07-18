#!/usr/bin/env python3
"""
Local Echo Server for BABEL-BEATS
Test real-time audio processing with echo effect
"""

import asyncio
import numpy as np
import pyaudio
import websockets
import json
import time
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalEchoServer:
    """
    Local echo server for testing audio processing
    Provides real-time echo with adjustable delay and effects
    """
    
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        # Audio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Echo buffer (circular buffer for delay)
        self.echo_delay_ms = 200  # Default 200ms delay
        self.echo_decay = 0.5      # Echo volume decay
        self.buffer_size = int(self.echo_delay_ms * self.sample_rate / 1000)
        self.echo_buffer = deque(maxlen=self.buffer_size)
        
        # Initialize buffer with silence
        for _ in range(self.buffer_size):
            self.echo_buffer.append(0.0)
        
        # Effects
        self.reverb_enabled = False
        self.pitch_shift_enabled = False
        self.pitch_shift_factor = 1.0
        
        # Performance tracking
        self.latency_measurements = deque(maxlen=100)
        
    def start_audio_stream(self):
        """Start audio input/output stream"""
        def audio_callback(in_data, frame_count, time_info, status):
            start_time = time.perf_counter()
            
            # Convert input to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Process audio
            processed = self.process_audio(audio_data)
            
            # Track latency
            latency = (time.perf_counter() - start_time) * 1000
            self.latency_measurements.append(latency)
            
            # Convert back to bytes
            out_data = processed.astype(np.float32).tobytes()
            
            return (out_data, pyaudio.paContinue)
        
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=audio_callback
        )
        
        self.stream.start_stream()
        logger.info(f"Audio stream started: {self.sample_rate}Hz, {self.chunk_size} samples/chunk")
    
    def process_audio(self, audio):
        """Process audio with echo and effects"""
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Get delayed sample from buffer
            delayed_sample = self.echo_buffer[0]
            
            # Mix original with echo
            output[i] = audio[i] + delayed_sample * self.echo_decay
            
            # Add current sample to buffer
            self.echo_buffer.append(audio[i])
            
            # Apply additional effects
            if self.reverb_enabled:
                output[i] = self.apply_reverb(output[i], i)
            
            if self.pitch_shift_enabled:
                output[i] = self.apply_pitch_shift(output[i], i)
        
        # Prevent clipping
        output = np.clip(output, -1.0, 1.0)
        
        return output
    
    def apply_reverb(self, sample, index):
        """Simple reverb effect"""
        # Add multiple delayed echoes with decay
        reverb = sample
        delays = [0.05, 0.1, 0.15, 0.2]  # seconds
        decays = [0.7, 0.5, 0.3, 0.1]
        
        for delay, decay in zip(delays, decays):
            delay_samples = int(delay * self.sample_rate)
            if index >= delay_samples:
                reverb += sample * decay
        
        return reverb * 0.7  # Reduce overall volume
    
    def apply_pitch_shift(self, sample, index):
        """Simple pitch shift effect"""
        # This is a very basic implementation
        # Real pitch shifting would use FFT or PSOLA
        return sample * self.pitch_shift_factor
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for control"""
        logger.info("WebSocket client connected")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'set_echo_delay':
                    self.echo_delay_ms = data['value']
                    self.buffer_size = int(self.echo_delay_ms * self.sample_rate / 1000)
                    self.echo_buffer = deque(maxlen=self.buffer_size)
                    logger.info(f"Echo delay set to {self.echo_delay_ms}ms")
                
                elif data['type'] == 'set_echo_decay':
                    self.echo_decay = data['value']
                    logger.info(f"Echo decay set to {self.echo_decay}")
                
                elif data['type'] == 'toggle_reverb':
                    self.reverb_enabled = not self.reverb_enabled
                    logger.info(f"Reverb {'enabled' if self.reverb_enabled else 'disabled'}")
                
                elif data['type'] == 'set_pitch_shift':
                    self.pitch_shift_enabled = True
                    self.pitch_shift_factor = data['value']
                    logger.info(f"Pitch shift set to {self.pitch_shift_factor}")
                
                elif data['type'] == 'get_stats':
                    avg_latency = np.mean(self.latency_measurements) if self.latency_measurements else 0
                    stats = {
                        'type': 'stats',
                        'avg_latency_ms': avg_latency,
                        'echo_delay_ms': self.echo_delay_ms,
                        'echo_decay': self.echo_decay,
                        'reverb_enabled': self.reverb_enabled,
                        'pitch_shift_factor': self.pitch_shift_factor
                    }
                    await websocket.send(json.dumps(stats))
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected")
    
    async def start_websocket_server(self, host='localhost', port=9876):
        """Start WebSocket server for control interface"""
        logger.info(f"Starting WebSocket server on {host}:{port}")
        await websockets.serve(self.websocket_handler, host, port)
    
    def stop(self):
        """Stop audio stream and cleanup"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        logger.info("Audio stream stopped")


# Simple CLI interface
def create_cli_interface():
    """Create command-line interface for echo server"""
    print("""
╔══════════════════════════════════════════╗
║      BABEL-BEATS Local Echo Server       ║
╠══════════════════════════════════════════╣
║  Commands:                               ║
║  - delay <ms>   : Set echo delay        ║
║  - decay <0-1>  : Set echo decay        ║
║  - reverb       : Toggle reverb         ║
║  - pitch <val>  : Set pitch shift       ║
║  - stats        : Show statistics       ║
║  - quit         : Exit                  ║
╚══════════════════════════════════════════╝
    """)


async def run_echo_server():
    """Run the echo server with CLI"""
    server = LocalEchoServer()
    
    # Start audio stream
    server.start_audio_stream()
    
    # Start WebSocket server
    websocket_task = asyncio.create_task(
        server.start_websocket_server()
    )
    
    # CLI interface
    create_cli_interface()
    
    try:
        while True:
            # Get user input
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\n🎤 Echo> "
                )
            except EOFError:
                break
            
            parts = command.strip().split()
            if not parts:
                continue
            
            cmd = parts[0].lower()
            
            if cmd == 'quit':
                break
            
            elif cmd == 'delay' and len(parts) > 1:
                try:
                    delay = int(parts[1])
                    server.echo_delay_ms = delay
                    server.buffer_size = int(delay * server.sample_rate / 1000)
                    server.echo_buffer = deque(maxlen=server.buffer_size)
                    print(f"✅ Echo delay set to {delay}ms")
                except ValueError:
                    print("❌ Invalid delay value")
            
            elif cmd == 'decay' and len(parts) > 1:
                try:
                    decay = float(parts[1])
                    if 0 <= decay <= 1:
                        server.echo_decay = decay
                        print(f"✅ Echo decay set to {decay}")
                    else:
                        print("❌ Decay must be between 0 and 1")
                except ValueError:
                    print("❌ Invalid decay value")
            
            elif cmd == 'reverb':
                server.reverb_enabled = not server.reverb_enabled
                print(f"✅ Reverb {'enabled' if server.reverb_enabled else 'disabled'}")
            
            elif cmd == 'pitch' and len(parts) > 1:
                try:
                    pitch = float(parts[1])
                    server.pitch_shift_factor = pitch
                    server.pitch_shift_enabled = True
                    print(f"✅ Pitch shift set to {pitch}")
                except ValueError:
                    print("❌ Invalid pitch value")
            
            elif cmd == 'stats':
                avg_latency = np.mean(server.latency_measurements) if server.latency_measurements else 0
                print(f"\n📊 Statistics:")
                print(f"   Average latency: {avg_latency:.2f}ms")
                print(f"   Echo delay: {server.echo_delay_ms}ms")
                print(f"   Echo decay: {server.echo_decay}")
                print(f"   Reverb: {'ON' if server.reverb_enabled else 'OFF'}")
                print(f"   Pitch shift: {server.pitch_shift_factor}")
            
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
    
    finally:
        server.stop()
        websocket_task.cancel()


# Test the echo locally without full server
def test_echo_algorithm():
    """Test echo algorithm without audio I/O"""
    print("\n🧪 Testing Echo Algorithm")
    print("=" * 40)
    
    # Generate test signal
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a chirp signal (frequency sweep)
    f0, f1 = 200, 800
    signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
    
    # Apply echo
    delay_ms = 200
    decay = 0.5
    delay_samples = int(delay_ms * sample_rate / 1000)
    
    echo_signal = np.zeros(len(signal) + delay_samples)
    echo_signal[:len(signal)] = signal
    echo_signal[delay_samples:] += signal * decay
    
    print(f"Original signal length: {len(signal)} samples")
    print(f"Echo delay: {delay_ms}ms ({delay_samples} samples)")
    print(f"Echo decay: {decay}")
    print(f"Output length: {len(echo_signal)} samples")
    print("✅ Echo algorithm test complete")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run algorithm test only
        test_echo_algorithm()
    else:
        # Check if pyaudio is available
        try:
            import pyaudio
            print("🎵 Starting BABEL-BEATS Local Echo Server...")
            print("🎤 Speak into your microphone to hear the echo effect!")
            asyncio.run(run_echo_server())
        except ImportError:
            print("⚠️  PyAudio not installed. Running algorithm test instead...")
            print("   To use the echo server, install pyaudio:")
            print("   pip install pyaudio")
            test_echo_algorithm()