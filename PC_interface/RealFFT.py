#!/usr/bin/python3
"""
Real-Time Interferometer Data with Dual FFT (Wavelength Display)
For DOUBLE-PASS interferometer system
Shows optical spectrum in nm, not temporal frequency
"""

import serial
import struct
import threading
import queue
import time
import collections
import sys
import csv
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.signal import windows, detrend
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configuration
START_BYTE = 0xAA
STOP_BYTE = 0x55
PORT = '/dev/ttyUSB0'
BAUD_RATE = 921600  # Data acquisition rate (not related to optical sampling)
ADC_SIZE = 4
ADC_FULLSCALE = 16261

# INTERFEROMETER CONFIGURATION (CRITICAL!)
MIRROR_SAMPLING_NM = 45.8996  # Physical mirror displacement per sample (nm)
IS_DOUBLE_PASS = True  # Set to True for Michelson-type interferometer

# Calculate effective optical sampling
if IS_DOUBLE_PASS:
    OPTICAL_SAMPLING_NM = 0.5 * MIRROR_SAMPLING_NM  # 150.602 nm for double-pass
    config_text = "DOUBLE-PASS Interferometer (Michelson type)"
else:
    OPTICAL_SAMPLING_NM = MIRROR_SAMPLING_NM  # 75.301 nm for single-pass
    config_text = "SINGLE-PASS Interferometer"

# FFT Window Configuration
FFT_WINDOW_SMALL = 512     # Fast update, lower resolution
FFT_WINDOW_LARGE = 10240   # Slow update, higher resolution

# Wavelength display range
WAVELENGTH_MIN = 400  # nm
WAVELENGTH_MAX = 1600  # nm

# Performance settings
USE_MULTITHREAD_FFT = True
BUFFER_SIZE = max(FFT_WINDOW_LARGE * 2, 20480)

# Pre-compute windows for FFT
HANN_WINDOW_SMALL = windows.hann(FFT_WINDOW_SMALL)
HANN_WINDOW_LARGE = windows.hann(FFT_WINDOW_LARGE)

# Thread pool for parallel FFT
if USE_MULTITHREAD_FFT:
    executor = ThreadPoolExecutor(max_workers=2)

# Serial and data structures
ser = serial.Serial(PORT, BAUD_RATE, timeout=1, parity='N')
data_queue = queue.Queue(maxsize=43*1024)

# Circular buffer for efficient data storage
class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.data = np.zeros(size, dtype=np.float32)
        self.index = 0
        self.full = False
        self.lock = threading.Lock()
    
    def append(self, value):
        with self.lock:
            self.data[self.index] = value
            self.index = (self.index + 1) % self.size
            if self.index == 0:
                self.full = True
    
    def get_recent(self, n):
        """Get the most recent n samples"""
        with self.lock:
            if not self.full and self.index < n:
                return None
            
            if self.index >= n:
                return self.data[self.index-n:self.index].copy()
            else:
                return np.concatenate([
                    self.data[self.index-n:],
                    self.data[:self.index]
                ])
    
    def get_all(self):
        """Get all valid data"""
        with self.lock:
            if self.full:
                return np.concatenate([
                    self.data[self.index:],
                    self.data[:self.index]
                ])
            else:
                return self.data[:self.index].copy()
    
    def length(self):
        with self.lock:
            return self.size if self.full else self.index

data_buffer = CircularBuffer(BUFFER_SIZE)
save_buffer = collections.deque(maxlen=10000000)
save_lock = threading.Lock()

DIR = "./GreenRawResults"
filename = "."

# Create figure
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.25, wspace=0.25)

ax1 = fig.add_subplot(gs[0, :])  # Top - interferogram
ax2 = fig.add_subplot(gs[1, 0])  # Bottom left - FFT 512
ax3 = fig.add_subplot(gs[1, 1])  # Bottom right - FFT 10240

# Initialize plots
line1, = ax1.plot([], [], 'b-', linewidth=0.5, label='Interferogram')
point1, = ax1.plot([], [], 'ro', markersize=8)
line2, = ax2.plot([], [], 'g-', linewidth=1)
line3, = ax3.plot([], [], 'b-', linewidth=1)

# Set up static elements
ax1.set_xlabel('Mirror Position (samples)', fontsize=10)
ax1.set_ylabel('Intensity', fontsize=10)
ax1.set_title(f'Real-time Interferogram - {config_text}', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

ax2.set_xlabel('Wavelength (nm)', fontsize=10)
ax2.set_ylabel('Spectral Amplitude', fontsize=10)
ax2.set_title(f'Optical Spectrum - {FFT_WINDOW_SMALL} points (Δx = {OPTICAL_SAMPLING_NM:.1f} nm optical)', 
              fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(WAVELENGTH_MIN, WAVELENGTH_MAX)

ax3.set_xlabel('Wavelength (nm)', fontsize=10)
ax3.set_ylabel('Spectral Amplitude', fontsize=10)
ax3.set_title(f'Optical Spectrum - {FFT_WINDOW_LARGE} points (Δx = {OPTICAL_SAMPLING_NM:.1f} nm optical)', 
              fontsize=10, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(WAVELENGTH_MIN, WAVELENGTH_MAX)

# Peak annotation objects
peak_text2 = ax2.text(0, 0, '', color='red', fontsize=9, rotation=45, visible=False)
peak_text3 = ax3.text(0, 0, '', color='red', fontsize=9, rotation=45, visible=False)
peak_vline2 = ax2.axvline(0, color='r', linestyle='--', alpha=0.5, visible=False)
peak_vline3 = ax3.axvline(0, color='r', linestyle='--', alpha=0.5, visible=False)

# Add reference lines for common wavelengths
reference_wavelengths = {
    532: ('Green laser', 'green'),
    632.8: ('HeNe laser', 'red'),
    650: ('Red diode', 'darkred'),
    785: ('NIR diode', 'brown'),
    1064: ('Nd:YAG', 'purple')
}

for wavelength, (label, color) in reference_wavelengths.items():
    if WAVELENGTH_MIN <= wavelength <= WAVELENGTH_MAX:
        ax2.axvline(wavelength, color=color, linestyle=':', alpha=0.3, linewidth=0.8)
        ax3.axvline(wavelength, color=color, linestyle=':', alpha=0.3, linewidth=0.8)


def compute_optical_fft(data_window, window_size):
    """
    Compute FFT for interferometer data and convert to wavelength domain
    """
    if data_window is None:
        return None, None
    
    # Select pre-computed window
    if window_size == FFT_WINDOW_SMALL:
        window = HANN_WINDOW_SMALL
    else:
        window = HANN_WINDOW_LARGE
    
    # Preprocessing
    data_windowed = data_window - np.mean(data_window)  # Remove DC
    data_windowed = detrend(data_windowed)  # Remove linear trends
    data_windowed *= window  # Apply window
    
    # Compute FFT
    N = len(data_windowed)
    fft_result = np.fft.fft(data_windowed)
    fft_magnitude = 2.0 * np.abs(fft_result[:N//2]) / N
    fft_magnitude[0] = fft_magnitude[0] / 2.0  # Correct DC
    
    # Calculate spatial frequencies
    # For interferometry: spatial frequency in cycles per meter
    dx_m = OPTICAL_SAMPLING_NM * 1e-9  # Convert nm to meters
    freq_spatial_m = np.fft.fftfreq(N, d=dx_m)[:N//2]
    
    # Convert to wavelengths (nm)
    # wavelength = 1 / spatial_frequency
    wavelengths_nm = np.zeros_like(freq_spatial_m)
    valid_idx = freq_spatial_m > 0
    wavelengths_nm[valid_idx] = 1e9 / freq_spatial_m[valid_idx]
    
    # Filter to reasonable wavelength range
    mask = (wavelengths_nm > 100) & (wavelengths_nm < 3000)
    wavelengths_filtered = wavelengths_nm[mask]
    magnitude_filtered = fft_magnitude[mask]
    
    # Sort by wavelength for proper plotting
    if len(wavelengths_filtered) > 0:
        sort_idx = np.argsort(wavelengths_filtered)
        wavelengths_sorted = wavelengths_filtered[sort_idx]
        magnitude_sorted = magnitude_filtered[sort_idx]
        return wavelengths_sorted, magnitude_sorted
    
    return None, None


def compute_ffts_parallel(data_small, data_large):
    """Compute both FFTs in parallel"""
    if USE_MULTITHREAD_FFT:
        future_small = executor.submit(compute_optical_fft, data_small, FFT_WINDOW_SMALL)
        future_large = executor.submit(compute_optical_fft, data_large, FFT_WINDOW_LARGE)
        result_small = future_small.result()
        result_large = future_large.result()
    else:
        result_small = compute_optical_fft(data_small, FFT_WINDOW_SMALL)
        result_large = compute_optical_fft(data_large, FFT_WINDOW_LARGE)
    
    return result_small, result_large


def update_plot(frame):
    """Update interferogram and optical spectrum plots"""
    
    buffer_length = data_buffer.length()
    
    if buffer_length < 100:
        return
    
    # Get interferogram data
    display_data = data_buffer.get_all()
    if display_data is None or len(display_data) == 0:
        return
    
    # Get windows for FFT
    data_small = data_buffer.get_recent(FFT_WINDOW_SMALL) if buffer_length >= FFT_WINDOW_SMALL else None
    data_large = data_buffer.get_recent(FFT_WINDOW_LARGE) if buffer_length >= FFT_WINDOW_LARGE else None
    
    # Update interferogram plot
    line1.set_data(np.arange(len(display_data)), display_data)
    ax1.set_xlim(max(0, len(display_data) - 2000), len(display_data))
    ax1.set_ylim(np.min(display_data) - 0.1, np.max(display_data) + 0.1)
    
    # Mark maximum
    if len(display_data) > 0:
        max_idx = np.argmax(display_data)
        max_val = display_data[max_idx]
        point1.set_data([max_idx], [max_val])
    
    # Compute and display FFTs
    if data_small is not None or data_large is not None:
        results = compute_ffts_parallel(data_small, data_large)
        
        # Update small FFT plot
        if results[0] is not None and results[0][0] is not None:
            wavelengths_small, magnitude_small = results[0]
            line2.set_data(wavelengths_small, magnitude_small)
            
            # Find peaks in visible range
            visible_mask = (wavelengths_small >= WAVELENGTH_MIN) & (wavelengths_small <= WAVELENGTH_MAX)
            if np.any(visible_mask):
                visible_mag = magnitude_small[visible_mask]
                visible_wl = wavelengths_small[visible_mask]
                if len(visible_mag) > 0:
                    peak_idx = np.argmax(visible_mag)
                    peak_wl = visible_wl[peak_idx]
                    peak_mag = visible_mag[peak_idx]
                    
                    peak_vline2.set_xdata([peak_wl, peak_wl])
                    peak_vline2.set_visible(True)
                    peak_text2.set_position((peak_wl, peak_mag))
                    peak_text2.set_text(f'{peak_wl:.1f} nm')
                    peak_text2.set_visible(True)
                    
                    # Update y-axis limits
                    ax2.set_ylim(0, max(visible_mag) * 1.1)
        
        # Update large FFT plot
        if results[1] is not None and results[1][0] is not None:
            wavelengths_large, magnitude_large = results[1]
            line3.set_data(wavelengths_large, magnitude_large)
            
            # Find peaks in visible range
            visible_mask = (wavelengths_large >= WAVELENGTH_MIN) & (wavelengths_large <= WAVELENGTH_MAX)
            if np.any(visible_mask):
                visible_mag = magnitude_large[visible_mask]
                visible_wl = wavelengths_large[visible_mask]
                if len(visible_mag) > 0:
                    peak_idx = np.argmax(visible_mag)
                    peak_wl = visible_wl[peak_idx]
                    peak_mag = visible_mag[peak_idx]
                    
                    peak_vline3.set_xdata([peak_wl, peak_wl])
                    peak_vline3.set_visible(True)
                    peak_text3.set_position((peak_wl, peak_mag))
                    peak_text3.set_text(f'{peak_wl:.1f} nm')
                    peak_text3.set_visible(True)
                    
                    # Update y-axis limits
                    ax3.set_ylim(0, max(visible_mag) * 1.1)
                    
                    # Print spectral resolution
                    if frame % 50 == 0:  # Print every 50 frames
                        spectral_res = OPTICAL_SAMPLING_NM * FFT_WINDOW_LARGE / 1000  # approximate
                        print(f"Peak at {peak_wl:.1f} nm | Spectral resolution: ~{spectral_res:.1f} nm")
    
    fig.canvas.draw_idle()


def read_uart():
    """Read data from UART"""
    while True:
        try:
            available = ser.in_waiting
            if available > 0:
                chunk_size = min(available, 4096)
                data = ser.read(chunk_size)
                data_queue.put(data)
            else:
                time.sleep(0.001)
        except Exception as e:
            print(f"Error in read_uart: {e}")
            time.sleep(0.01)


def read_packet():
    """Process packets from UART"""
    buffer = b''
    start_sequence = bytes([0xAA, 0xAA, 0xAA, 0xAA, 0xAA])
    stop_sequence = bytes([0x55, 0x55, 0x55, 0x55, 0x55])
    
    while True:
        try:
            data = data_queue.get(timeout=0.01)
            buffer += data
            
            while True:
                start_idx = buffer.find(start_sequence)
                if start_idx == -1:
                    buffer = buffer[-len(start_sequence)+1:] if len(buffer) >= len(start_sequence) else buffer
                    break
                
                stop_idx = buffer.find(stop_sequence, start_idx + len(start_sequence))
                if stop_idx == -1:
                    break
                
                packet = buffer[start_idx:stop_idx + len(stop_sequence)]
                
                if len(packet) > 7:
                    float_bytes = packet[7:11]
                    if len(float_bytes) == 4:
                        try:
                            value = struct.unpack('<f', float_bytes)[0]
                            if -0.7 < value < 1.1:
                                data_buffer.append(value)
                                with save_lock:
                                    save_buffer.append(value)
                        except:
                            pass
                
                buffer = buffer[stop_idx + len(stop_sequence):]
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in read_packet: {e}")


def save_csv_func():
    """Save data to CSV periodically"""
    while True:
        with save_lock:
            if save_buffer:
                points = list(save_buffer)
        
        if points:
            save_data_to_csv("Interferometer Data", 
                          f"{config_text} - Mirror sampling: {MIRROR_SAMPLING_NM} nm", 
                          points)
            print(f"[CSV] Saved {len(points)} points")
        
        time.sleep(13)


def get_next_filename():
    files = glob.glob(os.path.join(DIR, "acquisition_*.csv"))
    if not files:
        return os.path.join(DIR, "acquisition_1.csv")
    numbers = [int(f.split("_")[1].split(".")[0]) for f in files]
    return os.path.join(DIR, f"acquisition_{max(numbers) + 1}.csv")


def save_data_to_csv(name, description, values):
    global filename
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name])
        writer.writerow([description])
        for value in values:
            writer.writerow([value])


# Main execution
if __name__ == "__main__":
    filename = get_next_filename()
    
    print(f"\n{'='*60}")
    print(f"REAL-TIME INTERFEROMETER SPECTRUM ANALYZER")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Type: {config_text}")
    print(f"  - Mirror sampling: {MIRROR_SAMPLING_NM} nm")
    print(f"  - Optical sampling: {OPTICAL_SAMPLING_NM} nm")
    print(f"  - FFT Windows: {FFT_WINDOW_SMALL} and {FFT_WINDOW_LARGE} points")
    print(f"  - Wavelength range: {WAVELENGTH_MIN}-{WAVELENGTH_MAX} nm")
    print(f"  - Minimum wavelength (Nyquist): {2*OPTICAL_SAMPLING_NM:.1f} nm")
    print(f"\nData saving to: {filename}")
    print(f"{'='*60}\n")
    
    # Start threads
    thread_packet = threading.Thread(target=read_packet, daemon=True)
    thread_uart = threading.Thread(target=read_uart, daemon=True)
    thread_save = threading.Thread(target=save_csv_func, daemon=True)
    
    thread_save.start()
    thread_uart.start()
    thread_packet.start()
    
    # Animation
    ani = FuncAnimation(fig, update_plot, interval=20, cache_frame_data=False)
    
    # Title
    fig.suptitle('Real-Time Interferometer Data with Optical Spectrum Analysis', 
                 fontsize=14, fontweight='bold')
    
    # Add configuration note
    fig.text(0.5, 0.01, 
             f'{config_text} | Mirror Δx = {MIRROR_SAMPLING_NM} nm | Optical Δx = {OPTICAL_SAMPLING_NM} nm', 
             ha='center', fontsize=10, color='darkblue', weight='bold')
    
    plt.show()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if USE_MULTITHREAD_FFT:
            executor.shutdown(wait=False)
        ser.close()
        print("\nProgram terminated gracefully")
