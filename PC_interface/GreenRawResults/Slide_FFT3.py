#!/usr/bin/python3
"""
Sliding Window FFT Analyzer - Optimized for Multi-core Processing
Pre-computes all FFTs for smooth real-time sliding
"""

import serial
import struct
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import csv
import os
import glob

import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Button, RadioButtons
from scipy.signal.windows import hann, hamming
from scipy.signal import fftconvolve
from numpy.fft import fft 
from scipy.signal import detrend, windows
import numpy as np
from tqdm import tqdm
import time

# Configuration
SAMPLING_INTERVAL_NM = 45.8996  # Sampling interval in nanometers
OVERLAP_PERCENTAGE = 75  # 75% overlap between windows (higher = smoother but more computation)
MAX_WORKERS = 10  # Use 10 of 12 threads for FFT computation

# Initialize plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(top=0.85, right=0.85, bottom=0.08)

# Global variables
selected_rect = None
selected_start = None
selected_end = None
selection_text = None
data_values = []
window_size = None

# FFT Cache - stores all pre-computed FFTs
fft_cache = {
    'x_vals': None,      # Frequency/wavelength axis
    'y_vals': None,      # FFT magnitudes for all windows [n_windows, n_freqs]
    'positions': None,   # Center positions of each window
    'window_size': None,
    'mode': None,        # 'Wavelength (nm)' or 'Wavenumber (cm⁻¹)'
    'computed': False
}

# GUI Elements
filename_ax = plt.axes([0.075, 0.96, 0.15, 0.025])
filename_box = TextBox(filename_ax, 'CSV File:', initial="acquisition_226.csv")

radio_ax = plt.axes([0.2, 0.475, 0.20, 0.045])
domain_selector = RadioButtons(radio_ax, ('Wavelength (nm)', 'Wavenumber (cm⁻¹)'))

fft_ax = plt.axes([0.0075, 0.475, 0.035, 0.035])
fft_button = Button(fft_ax, 'FFT')

reset_ax = plt.axes([0.0525, 0.475, 0.05, 0.035])
reset_button = Button(reset_ax, 'Reset')

load_ax = plt.axes([0.1125, 0.475, 0.08, 0.035])
load_button = Button(load_ax, 'Load CSV')

# Status text
status_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10, color='blue', weight='bold')


def read_csv_values(filename):
    """Reads numeric values from CSV file (skipping metadata rows)."""
    values = []
    
    try:
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            # Skip metadata rows
            next(reader, None)
            next(reader, None)
            
            for row in reader:
                if row:
                    try:
                        values.append(float(row[0]))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return []
    
    return values


def load_csv(event):
    """Load and plot CSV file contents."""
    global data_values, fft_cache
    csv_filename = filename_box.text.strip()
    
    # Reset FFT cache
    fft_cache['computed'] = False
    
    data_values = read_csv_values(csv_filename)
    if not data_values:
        print(f"No data loaded from {csv_filename}")
        status_text.set_text(f"❌ Failed to load {csv_filename}")
        return
    
    ax1.clear()
    ax1.plot(data_values, label='Sensor Data', linewidth=0.5, color='blue', alpha=0.7)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Data from {csv_filename} ({len(data_values)} points)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    status_text.set_text(f"✓ Loaded {len(data_values)} points | Select window size and press FFT")
    fig.canvas.draw_idle()


def compute_single_fft(args):
    """Compute FFT for a single window - optimized for parallel execution."""
    data_segment, idx, mode, sampling_interval = args
    
    N = len(data_segment)
    
    # Preprocessing
    data_proc = data_segment - np.mean(data_segment)
    data_proc = detrend(data_proc)
    window = windows.hann(N)
    windowed_data = data_proc * window
    
    # FFT
    fft_result = np.fft.fft(windowed_data)
    fft_magnitude = 2.0 * np.abs(fft_result[:N//2]) / N
    fft_magnitude[0] = fft_magnitude[0] / 2.0
    
    # Frequency axis
    dx_m = sampling_interval * 1e-9
    freq_spatial_m = np.fft.fftfreq(N, d=dx_m)[:N//2]
    
    # Convert to display units
    if mode == 'Wavelength (nm)':
        valid_idx = freq_spatial_m > 0
        wavelengths_m = np.zeros_like(freq_spatial_m)
        wavelengths_m[valid_idx] = 1.0 / (2.0 * freq_spatial_m[valid_idx])
        x_vals = wavelengths_m * 1e9
        
        # Filter reasonable range
        mask = (x_vals > 100) & (x_vals < 3000)
        x_vals = x_vals[mask]
        y_vals = fft_magnitude[mask]
        
        # Sort
        sort_idx = np.argsort(x_vals)
        x_vals = x_vals[sort_idx]
        y_vals = y_vals[sort_idx]
        
    else:  # Wavenumber
        wavenumbers_cm = 0.5 * freq_spatial_m * 0.01
        mask = (wavenumbers_cm > 0) & (wavenumbers_cm < 10000)
        x_vals = wavenumbers_cm[mask]
        y_vals = fft_magnitude[mask]
    
    return idx, x_vals, y_vals


def compute_all_ffts_parallel(data, win_size, mode, sampling_interval):
    """
    Compute FFTs for all windows in parallel using multiprocessing.
    Returns arrays of x_vals, y_vals, and window positions.
    """
    n_samples = len(data)
    
    # Calculate hop size (stride)
    hop_size = max(1, int(win_size * (1 - OVERLAP_PERCENTAGE / 100)))
    
    # Generate all window positions
    positions = []
    window_data = []
    
    for start_idx in range(0, n_samples - win_size + 1, hop_size):
        end_idx = start_idx + win_size
        center_pos = (start_idx + end_idx) // 2
        positions.append(center_pos)
        window_data.append((data[start_idx:end_idx], len(positions)-1, mode, sampling_interval))
    
    n_windows = len(positions)
    print(f"\n{'='*60}")
    print(f"Computing {n_windows} FFTs in parallel...")
    print(f"Window size: {win_size} | Hop size: {hop_size} | Overlap: {OVERLAP_PERCENTAGE}%")
    print(f"Using {MAX_WORKERS} CPU threads")
    print(f"{'='*60}\n")
    
    status_text.set_text(f"⏳ Computing {n_windows} FFTs in parallel...")
    fig.canvas.draw_idle()
    
    # Parallel FFT computation
    results = {}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compute_single_fft, wd): i for i, wd in enumerate(window_data)}
        
        completed = 0
        for future in as_completed(futures):
            idx, x_vals, y_vals = future.result()
            results[idx] = (x_vals, y_vals)
            completed += 1
            
            # Update progress every 10%
            if completed % max(1, n_windows // 10) == 0:
                progress = (completed / n_windows) * 100
                elapsed = time.time() - start_time
                print(f"Progress: {progress:.1f}% ({completed}/{n_windows}) | Elapsed: {elapsed:.1f}s")
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Computed {n_windows} FFTs in {elapsed_time:.2f}s ({n_windows/elapsed_time:.1f} FFTs/sec)\n")
    
    # Organize results
    # Assume all x_vals are the same (same frequency axis)
    x_axis = results[0][0]
    n_freqs = len(x_axis)
    
    # Stack all y_vals into a 2D array
    y_matrix = np.zeros((n_windows, n_freqs), dtype=np.float32)
    for idx in range(n_windows):
        y_matrix[idx, :] = results[idx][1]
    
    return x_axis, y_matrix, np.array(positions)


def run_fft(event):
    """Pre-compute all FFTs for the selected window size."""
    global data_values, fft_cache, window_size
    
    if not data_values:
        print("No data loaded. Please load a CSV file first.")
        status_text.set_text("❌ No data loaded")
        return
    
    if selected_start is None or selected_end is None:
        print("No selection made. Please select a window size.")
        status_text.set_text("❌ No window selected")
        return
    
    # Calculate window size from selection
    window_size = int(abs(selected_end - selected_start))
    
    if window_size < 4:
        print(f"Window too small ({window_size} points). Need at least 4 points.")
        status_text.set_text(f"❌ Window too small ({window_size} points)")
        return
    
    # Get display mode
    mode = domain_selector.value_selected
    
    # Compute all FFTs in parallel
    x_axis, y_matrix, positions = compute_all_ffts_parallel(
        np.array(data_values), 
        window_size, 
        mode, 
        SAMPLING_INTERVAL_NM
    )
    
    # Store in cache
    fft_cache['x_vals'] = x_axis
    fft_cache['y_vals'] = y_matrix
    fft_cache['positions'] = positions
    fft_cache['window_size'] = window_size
    fft_cache['mode'] = mode
    fft_cache['computed'] = True
    
    status_text.set_text(f"✓ {len(positions)} FFTs ready | Slide window to explore")
    
    # Display first FFT
    update_fft_display(positions[0])
    
    print(f"FFT cache ready with {len(positions)} windows")
    print(f"Memory usage: ~{(y_matrix.nbytes / 1024 / 1024):.1f} MB")


def find_nearest_fft_index(position):
    """Find the index of the nearest pre-computed FFT."""
    if not fft_cache['computed']:
        return None
    
    positions = fft_cache['positions']
    idx = np.argmin(np.abs(positions - position))
    return idx


def update_fft_display(center_position):
    """Update FFT plot based on window center position."""
    if not fft_cache['computed']:
        return
    
    # Find nearest FFT
    idx = find_nearest_fft_index(center_position)
    if idx is None:
        return
    
    x_vals = fft_cache['x_vals']
    y_vals = fft_cache['y_vals'][idx, :]
    mode = fft_cache['mode']
    window_size = fft_cache['window_size']
    actual_position = fft_cache['positions'][idx]
    
    # Get x-axis label
    x_label = mode.split()[0]  # 'Wavelength' or 'Wavenumber'
    
    # Plot
    ax2.clear()
    ax2.plot(x_vals, y_vals, linewidth=0.8, color='green', alpha=0.8)
    ax2.set_xlabel(f"{mode}", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.set_title(f"FFT - Window {idx+1}/{len(fft_cache['positions'])} (Center: {actual_position})", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Set limits
    if len(x_vals) > 0 and len(y_vals) > 0:
        ax2.set_xlim(min(x_vals), max(x_vals))
        ax2.set_ylim(0, max(y_vals) * 1.1)
        
        # Find and mark peak
        peak_idx = np.argmax(y_vals)
        peak_x = x_vals[peak_idx]
        peak_y = y_vals[peak_idx]
        ax2.axvline(peak_x, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        text_x = peak_x * 1.02 if peak_x < np.median(x_vals) else peak_x * 0.98
        ax2.text(text_x, peak_y * 0.95, 
                f'Peak: {peak_x:.1f} {x_label}', 
                fontsize=10, color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    fig.canvas.draw_idle()


def onselect(xmin, xmax):
    """Handle span selection - shows window and updates FFT if cache is ready."""
    global selected_rect, selection_text, selected_start, selected_end
    
    # Store selection boundaries
    selected_start = xmin
    selected_end = xmax
    center_position = (xmin + xmax) / 2
    
    # Remove previous selection visuals
    if selected_rect:
        selected_rect.remove()
        selected_rect = None
    if selection_text:
        selection_text.remove()
        selection_text = None
    
    # Create new selection rectangle
    selected_rect = plt.Rectangle(
        (xmin, ax1.get_ylim()[0]),
        xmax - xmin,
        ax1.get_ylim()[1] - ax1.get_ylim()[0],
        linewidth=1.5,
        edgecolor='red',
        facecolor='red',
        alpha=0.2
    )
    ax1.add_patch(selected_rect)
    
    # Add text showing window info
    num_samples = int(abs(xmax - xmin))
    selection_text = ax1.text(
        xmin + (xmax - xmin) / 2,
        ax1.get_ylim()[1] * 0.98,
        f"Window: {num_samples} samples",
        color='red',
        ha='center',
        fontsize=10,
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
    )
    
    # If FFTs are pre-computed, update display
    if fft_cache['computed'] and fft_cache['window_size'] == num_samples:
        update_fft_display(center_position)
    
    fig.canvas.draw_idle()


def reset(event):
    """Reset the plots and selections."""
    global selected_rect, selected_start, selected_end, selection_text, data_values, fft_cache
    
    selected_start = None
    selected_end = None
    data_values = []
    fft_cache['computed'] = False
    
    if selected_rect:
        selected_rect.remove()
        selected_rect = None
    
    if selection_text:
        selection_text.remove()
        selection_text = None
    
    ax1.clear()
    ax2.clear()
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('Load data to begin', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.set_title('FFT Result - Sliding Window', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    status_text.set_text("Ready")
    fig.canvas.draw_idle()


# Initialize plots
ax1.set_xlabel('Sample Index', fontsize=12)
ax1.set_ylabel('Intensity', fontsize=12)
ax1.set_title('Load data to begin', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Frequency', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title('FFT Result - Sliding Window', fontsize=14)
ax2.grid(True, alpha=0.3)

# Connect event handlers
print("Initializing Sliding Window FFT Analyzer...")
print(f"System: {multiprocessing.cpu_count()} CPU cores detected")
print(f"Configuration: {MAX_WORKERS} threads, {OVERLAP_PERCENTAGE}% overlap")
fft_button.on_clicked(run_fft)
reset_button.on_clicked(reset)
load_button.on_clicked(load_csv)

# Setup span selector
span = SpanSelector(
    ax1, onselect, 'horizontal',
    useblit=True,
    props=dict(facecolor='red', alpha=0.2),
    interactive=True,
    drag_from_anywhere=True
)

status_text.set_text("Ready | Load CSV to begin")

plt.tight_layout()
plt.show()
