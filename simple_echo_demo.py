#!/usr/bin/env python3
"""
Simple Echo Demo for BABEL-BEATS
Demonstrates echo effect without audio I/O dependencies
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time

class SimpleEchoDemo:
    """Visual demonstration of echo effect"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 3.0
        self.echo_delay = 0.3  # 300ms
        self.echo_decay = 0.5
        
    def generate_signal(self, signal_type='speech'):
        """Generate different test signals"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        if signal_type == 'speech':
            # Simulate speech with bursts
            signal = np.zeros_like(t)
            # Word 1: "Hello" (0.2-0.5s)
            mask1 = (t >= 0.2) & (t <= 0.5)
            signal[mask1] = np.sin(2 * np.pi * 150 * t[mask1]) * np.exp(-(t[mask1] - 0.2) * 5)
            
            # Word 2: "World" (0.8-1.1s)
            mask2 = (t >= 0.8) & (t <= 1.1)
            signal[mask2] = np.sin(2 * np.pi * 200 * t[mask2]) * np.exp(-(t[mask2] - 0.8) * 5)
            
            # Word 3: "Echo" (1.5-1.8s)
            mask3 = (t >= 1.5) & (t <= 1.8)
            signal[mask3] = np.sin(2 * np.pi * 180 * t[mask3]) * np.exp(-(t[mask3] - 1.5) * 5)
            
        elif signal_type == 'tone':
            # Pure tone
            signal = np.sin(2 * np.pi * 440 * t) * 0.5
            
        elif signal_type == 'click':
            # Click/impulse
            signal = np.zeros_like(t)
            signal[int(0.5 * self.sample_rate)] = 1.0
            
        return t, signal
    
    def apply_echo(self, signal):
        """Apply echo effect to signal"""
        delay_samples = int(self.echo_delay * self.sample_rate)
        output = np.zeros(len(signal) + delay_samples)
        
        # Original signal
        output[:len(signal)] = signal
        
        # Add echoes
        current_decay = self.echo_decay
        for i in range(3):  # 3 echo repetitions
            start_idx = (i + 1) * delay_samples
            end_idx = start_idx + len(signal)
            if end_idx <= len(output):
                output[start_idx:end_idx] += signal * current_decay
                current_decay *= self.echo_decay
        
        return output
    
    def visualize_echo(self):
        """Create visual demonstration of echo"""
        print("\n🎵 Simple Echo Demonstration")
        print("=" * 40)
        
        # Generate signals
        t, original = self.generate_signal('speech')
        echo_signal = self.apply_echo(original)
        t_echo = np.linspace(0, len(echo_signal) / self.sample_rate, len(echo_signal))
        
        # Create visualization
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use TkAgg backend
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Original signal
            ax1.plot(t, original, 'b-', linewidth=2)
            ax1.set_title('Original Signal (Speech Simulation)', fontsize=14)
            ax1.set_ylabel('Amplitude')
            ax1.set_xlim(0, 3)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-1.2, 1.2)
            
            # Add labels for words
            ax1.text(0.35, 0.8, 'Hello', ha='center', fontsize=12, color='blue')
            ax1.text(0.95, 0.8, 'World', ha='center', fontsize=12, color='blue')
            ax1.text(1.65, 0.8, 'Echo', ha='center', fontsize=12, color='blue')
            
            # Echo signal
            ax2.plot(t_echo, echo_signal, 'r-', linewidth=2, alpha=0.8)
            ax2.set_title(f'Signal with Echo (delay={self.echo_delay*1000:.0f}ms, decay={self.echo_decay})', fontsize=14)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Amplitude')
            ax2.set_xlim(0, 4)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-1.2, 1.2)
            
            # Add echo indicators
            for i in range(3):
                delay_time = (i + 1) * self.echo_delay
                decay = self.echo_decay ** (i + 1)
                
                # Hello echo
                ax2.text(0.35 + delay_time, 0.6 * decay, f'Hello\n({decay:.2f})', 
                        ha='center', fontsize=10, color='red', alpha=0.7)
                
                # World echo
                ax2.text(0.95 + delay_time, 0.6 * decay, f'World\n({decay:.2f})', 
                        ha='center', fontsize=10, color='red', alpha=0.7)
                
                # Echo echo
                if 1.65 + delay_time < 4:
                    ax2.text(1.65 + delay_time, 0.6 * decay, f'Echo\n({decay:.2f})', 
                            ha='center', fontsize=10, color='red', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("⚠️  Matplotlib not available for visualization")
            print("   Install with: pip install matplotlib")
        
        # Text-based visualization
        self.text_visualization(original, echo_signal)
    
    def text_visualization(self, original, echo):
        """ASCII art visualization of echo"""
        print("\n📊 Echo Effect Visualization (Text)")
        print("-" * 60)
        
        # Downsample for display
        display_rate = 50  # samples per line
        original_display = original[::self.sample_rate//display_rate][:150]
        echo_display = echo[::self.sample_rate//display_rate][:200]
        
        print("Original:")
        self.draw_waveform(original_display)
        
        print("\nWith Echo:")
        self.draw_waveform(echo_display)
        
        print("\n📈 Echo Analysis:")
        print(f"   Original duration: {len(original)/self.sample_rate:.2f}s")
        print(f"   Echo delay: {self.echo_delay*1000:.0f}ms")
        print(f"   Echo decay: {self.echo_decay}")
        print(f"   Total duration: {len(echo)/self.sample_rate:.2f}s")
        
    def draw_waveform(self, signal):
        """Draw ASCII waveform"""
        # Normalize signal
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal))
        
        height = 10
        for sample in signal:
            level = int((sample + 1) * height / 2)
            level = max(0, min(height, level))
            
            line = [' '] * (height + 1)
            line[height - level] = '█'
            print(''.join(line), end='')
        print()
    
    def interactive_demo(self):
        """Interactive echo parameter adjustment"""
        print("\n🎛️  Interactive Echo Demo")
        print("=" * 40)
        
        while True:
            print(f"\nCurrent settings:")
            print(f"  Delay: {self.echo_delay*1000:.0f}ms")
            print(f"  Decay: {self.echo_decay:.2f}")
            
            print("\nOptions:")
            print("  1. Change delay (50-1000ms)")
            print("  2. Change decay (0.1-0.9)")
            print("  3. Test with speech")
            print("  4. Test with tone")
            print("  5. Test with click")
            print("  6. Visualize")
            print("  0. Exit")
            
            try:
                choice = input("\nChoice: ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    delay = float(input("New delay (ms): ")) / 1000
                    if 0.05 <= delay <= 1.0:
                        self.echo_delay = delay
                    else:
                        print("❌ Delay must be between 50-1000ms")
                elif choice == '2':
                    decay = float(input("New decay (0.1-0.9): "))
                    if 0.1 <= decay <= 0.9:
                        self.echo_decay = decay
                    else:
                        print("❌ Decay must be between 0.1-0.9")
                elif choice in ['3', '4', '5']:
                    signal_types = {'3': 'speech', '4': 'tone', '5': 'click'}
                    signal_type = signal_types[choice]
                    
                    print(f"\n🎵 Testing with {signal_type}...")
                    t, signal = self.generate_signal(signal_type)
                    echo = self.apply_echo(signal)
                    
                    print(f"✅ Echo applied!")
                    print(f"   Original energy: {np.sum(signal**2):.3f}")
                    print(f"   Echo energy: {np.sum(echo**2):.3f}")
                    print(f"   Energy increase: {np.sum(echo**2)/np.sum(signal**2):.2f}x")
                    
                elif choice == '6':
                    self.visualize_echo()
                    
            except ValueError:
                print("❌ Invalid input")
            except KeyboardInterrupt:
                break
        
        print("\n👋 Thanks for trying the echo demo!")


def main():
    """Run the echo demonstration"""
    demo = SimpleEchoDemo()
    
    print("🎵 BABEL-BEATS Simple Echo Demo")
    print("==============================")
    print("\nThis demo shows how echo effects work")
    print("without requiring audio hardware.\n")
    
    # Run visualization
    demo.visualize_echo()
    
    # Run interactive demo
    try:
        demo.interactive_demo()
    except KeyboardInterrupt:
        print("\n\n✅ Demo completed!")


if __name__ == "__main__":
    main()