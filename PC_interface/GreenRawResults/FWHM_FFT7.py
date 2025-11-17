#!/usr/bin/python3
"""
Sliding Window FFT Analyzer - ULTRA-OPTIMIZED for Real-Time Dragging
Pre-computes all FFTs + uses blitting for instantaneous display updates

HARDWARE OPTIMIZATION:
- Configured for: AMD Ryzen 5 PRO 5675U (6 cores / 12 threads)
- Memory: 16 GB DDR4-2400 dual-channel
- Leverages: AVX2, FMA, L3 cache (16 MB)
- Process usage: 11 workers (92% utilization) - TRUE multiprocessing!
- Overlap: 85% for ultra-smooth real-time FFT display

DISPLAY OPTIMIZATION:
- Uses matplotlib BLITTING for 10-100x faster updates during drag
- Background thread prepares data while main thread renders
- Cached plot elements for minimal redraw overhead
"""

import serial
import struct
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from queue import Queue
from threading import Thread, Lock

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
import time

# Configuration
SAMPLING_INTERVAL_NM = 45.8996  # Sampling interval in nanometers
OVERLAP_PERCENTAGE = 99.95  # 85% overlap for ultra-smooth sliding
MAX_WORKERS = 11  # Use 11 of 12 processes for FFT computation
USE_BACKGROUND_THREAD = True  # Enable background data preparation thread

# Memory and performance optimizations
BUFFER_SIZE_MULTIPLIER = 3
USE_FLOAT32 = True

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

# Blitting cache - stores matplotlib artists for fast updates
blit_cache = {
    'background': None,      # Saved background
    'line': None,           # FFT line artist
    'peak_line': None,      # Peak vertical line artist
    'peak_text': None,      # Peak annotation
    'fwhm_line': None,      # FWHM horizontal line artist
    'fwhm_markers': None,   # FWHM endpoint markers
    'title': None,          # Title text
    'enabled': False        # Whether blitting is active
}

# Background thread communication
update_queue = Queue(maxsize=1)  # Queue for FFT update requests
result_queue = Queue(maxsize=1)  # Queue for prepared data
data_prep_thread = None
thread_lock = Lock()
thread_running = False

# FFT Cache - stores all pre-computed FFTs
fft_cache = {
    'x_vals': None,
    'y_vals': None,
    'positions': None,
    'window_size': None,
    'mode': None,
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

status_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10, color='blue', weight='bold')


def read_csv_values(filename):
    """Reads numeric values from CSV file (skipping metadata rows)."""
    values = []
    
    try:
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
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
    global data_values, fft_cache, blit_cache
    csv_filename = filename_box.text.strip()
    
    # Reset caches
    fft_cache['computed'] = False
    blit_cache['enabled'] = False
    
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
        
        mask = (x_vals > 100) & (x_vals < 3000)
        x_vals = x_vals[mask]
        y_vals = fft_magnitude[mask]
        
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
    """Compute FFTs for all windows in parallel using multiprocessing."""
    n_samples = len(data)
    hop_size = max(1, int(win_size * (1 - OVERLAP_PERCENTAGE / 100)))
    
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
    print(f"Using {MAX_WORKERS} CPU processes (TRUE multiprocessing!)")
    print(f"{'='*60}\n")
    
    status_text.set_text(f"⏳ Computing {n_windows} FFTs in parallel...")
    fig.canvas.draw_idle()
    
    results = {}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compute_single_fft, wd): i for i, wd in enumerate(window_data)}
        
        completed = 0
        for future in as_completed(futures):
            idx, x_vals, y_vals = future.result()
            results[idx] = (x_vals, y_vals)
            completed += 1
            
            if completed % max(1, n_windows // 10) == 0:
                progress = (completed / n_windows) * 100
                elapsed = time.time() - start_time
                print(f"Progress: {progress:.1f}% ({completed}/{n_windows}) | Elapsed: {elapsed:.1f}s")
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Computed {n_windows} FFTs in {elapsed_time:.2f}s ({n_windows/elapsed_time:.1f} FFTs/sec)\n")
    
    x_axis = results[0][0]
    n_freqs = len(x_axis)
    y_matrix = np.zeros((n_windows, n_freqs), dtype=np.float32)
    for idx in range(n_windows):
        y_matrix[idx, :] = results[idx][1]
    
    return x_axis, y_matrix, np.array(positions)


def data_preparation_worker():
    """Background thread that prepares FFT data for plotting."""
    global thread_running
    
    print("Background data preparation thread started")
    
    while thread_running:
        try:
            # Wait for update request (non-blocking with timeout)
            center_position = update_queue.get(timeout=0.1)
            
            if not fft_cache['computed']:
                continue
            
            # Find nearest FFT (fast numpy operation)
            positions = fft_cache['positions']
            idx = np.argmin(np.abs(positions - center_position))
            
            # Prepare data
            x_vals = fft_cache['x_vals']
            y_vals = fft_cache['y_vals'][idx, :]
            mode = fft_cache['mode']
            actual_position = fft_cache['positions'][idx]
            x_label = mode.split()[0]
            
            # Find peak
            peak_idx = np.argmax(y_vals)
            peak_x = x_vals[peak_idx]
            peak_y = y_vals[peak_idx]
            
            # Calculate FWHM
            fwhm, x_left, x_right = calculate_fwhm(x_vals, y_vals, peak_idx)
            
            # Package result
            result = {
                'idx': idx,
                'x_vals': x_vals,
                'y_vals': y_vals,
                'mode': mode,
                'actual_position': actual_position,
                'x_label': x_label,
                'peak_x': peak_x,
                'peak_y': peak_y,
                'fwhm': fwhm,
                'fwhm_left': x_left,
                'fwhm_right': x_right,
                'n_windows': len(fft_cache['positions'])
            }
            
            # Send to main thread (replace old result if queue is full)
            try:
                result_queue.get_nowait()  # Clear old result
            except:
                pass
            result_queue.put(result)
            
        except:
            continue
    
    print("Background data preparation thread stopped")


def start_background_thread():
    """Start the background data preparation thread."""
    global data_prep_thread, thread_running
    
    if USE_BACKGROUND_THREAD and not thread_running:
        thread_running = True
        data_prep_thread = Thread(target=data_preparation_worker, daemon=True)
        data_prep_thread.start()


def stop_background_thread():
    """Stop the background data preparation thread."""
    global thread_running
    thread_running = False


def init_blitting():
    """Initialize blitting for fast FFT plot updates."""
    global blit_cache
    
    if not fft_cache['computed']:
        return
    
    # Clear and setup axes
    ax2.clear()
    ax2.set_xlabel(f"{fft_cache['mode']}", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Create plot artists (empty initially)
    blit_cache['line'], = ax2.plot([], [], linewidth=0.8, color='green', alpha=0.8, animated=True)
    blit_cache['peak_line'] = ax2.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=0.125, animated=True)
    blit_cache['fwhm_line'] = ax2.hlines(0, 0, 1, colors='orange', linestyles='-', linewidth=1.5, alpha=0.7, animated=True)
    blit_cache['fwhm_markers'], = ax2.plot([], [], 'o', color='orange', markersize=4, animated=True)
    blit_cache['peak_text'] = ax2.text(0, 0, '', fontsize=10, color='red',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                        animated=True)
    blit_cache['title'] = ax2.set_title('', fontsize=14, animated=True)
    
    # Draw everything once
    fig.canvas.draw()
    
    # Save background (everything except animated artists)
    blit_cache['background'] = fig.canvas.copy_from_bbox(ax2.bbox)
    blit_cache['enabled'] = True
    
    print("✓ Blitting initialized for ultra-fast updates (with FWHM)")


def update_fft_display_blit(result_data):
    """Update FFT plot using blitting for maximum speed."""
    if not blit_cache['enabled'] or blit_cache['background'] is None:
        return
    
    # Restore background
    fig.canvas.restore_region(blit_cache['background'])
    
    # Update line data
    blit_cache['line'].set_data(result_data['x_vals'], result_data['y_vals'])
    
    # Update axes limits if needed
    ax2.set_xlim(min(result_data['x_vals']), max(result_data['x_vals']))
    ax2.set_ylim(0, max(result_data['y_vals']) * 1.1)
    
    # Update peak line
    blit_cache['peak_line'].set_xdata([result_data['peak_x'], result_data['peak_x']])
    
    # Update FWHM visualization
    if result_data['fwhm'] is not None:
        half_max = result_data['peak_y'] / 2.0
        # Update FWHM horizontal line
        blit_cache['fwhm_line'].set_segments([[(result_data['fwhm_left'], half_max), 
                                                 (result_data['fwhm_right'], half_max)]])
        # Update FWHM markers
        blit_cache['fwhm_markers'].set_data([result_data['fwhm_left'], result_data['fwhm_right']], 
                                             [half_max, half_max])
        
        # Update peak text with FWHM
        text_x = result_data['peak_x'] * 1.02 if result_data['peak_x'] < np.median(result_data['x_vals']) else result_data['peak_x'] * 0.98
        blit_cache['peak_text'].set_position((text_x, result_data['peak_y'] * 0.95))
        blit_cache['peak_text'].set_text(f"Peak: {result_data['peak_x']:.1f} {result_data['x_label']}\nFWHM: {result_data['fwhm']:.1f} {result_data['x_label']}")
    else:
        # Hide FWHM elements if calculation failed
        blit_cache['fwhm_line'].set_segments([[(0, 0), (0, 0)]])
        blit_cache['fwhm_markers'].set_data([], [])
        
        # Update peak text without FWHM
        text_x = result_data['peak_x'] * 1.02 if result_data['peak_x'] < np.median(result_data['x_vals']) else result_data['peak_x'] * 0.98
        blit_cache['peak_text'].set_position((text_x, result_data['peak_y'] * 0.95))
        blit_cache['peak_text'].set_text(f"Peak: {result_data['peak_x']:.1f} {result_data['x_label']}")
    
    # Update title
    title_text = f"FFT - Window {result_data['idx']+1}/{result_data['n_windows']} (Center: {result_data['actual_position']})"
    ax2.set_title(title_text, fontsize=14)
    
    # Redraw animated artists
    ax2.draw_artist(blit_cache['line'])
    ax2.draw_artist(blit_cache['peak_line'])
    ax2.draw_artist(blit_cache['fwhm_line'])
    ax2.draw_artist(blit_cache['fwhm_markers'])
    ax2.draw_artist(blit_cache['peak_text'])
    
    # Blit the updates
    fig.canvas.blit(ax2.bbox)
    fig.canvas.flush_events()


def calculate_fwhm(x_vals, y_vals, peak_idx):
    """
    Calculate Full Width at Half Maximum (FWHM).
    Returns (fwhm, x_left, x_right) or (None, None, None) if calculation fails.
    """
    peak_y = y_vals[peak_idx]
    half_max = peak_y / 2.0
    
    # Find points on left side of peak where signal crosses half maximum
    left_indices = np.where((x_vals < x_vals[peak_idx]) & (y_vals <= half_max))[0]
    if len(left_indices) == 0:
        return None, None, None
    left_idx = left_indices[-1]  # Closest to peak
    
    # Find points on right side of peak where signal crosses half maximum
    right_indices = np.where((x_vals > x_vals[peak_idx]) & (y_vals <= half_max))[0]
    if len(right_indices) == 0:
        return None, None, None
    right_idx = right_indices[0]  # Closest to peak
    
    # Linear interpolation for precise FWHM
    if left_idx + 1 < len(x_vals):
        # Interpolate on left side
        x1, x2 = x_vals[left_idx], x_vals[left_idx + 1]
        y1, y2 = y_vals[left_idx], y_vals[left_idx + 1]
        if y2 != y1:
            x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        else:
            x_left = x_vals[left_idx]
    else:
        x_left = x_vals[left_idx]
    
    if right_idx > 0:
        # Interpolate on right side
        x1, x2 = x_vals[right_idx - 1], x_vals[right_idx]
        y1, y2 = y_vals[right_idx - 1], y_vals[right_idx]
        if y2 != y1:
            x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        else:
            x_right = x_vals[right_idx]
    else:
        x_right = x_vals[right_idx]
    
    fwhm = abs(x_right - x_left)
    return fwhm, x_left, x_right


def update_fft_display_full(center_position):
    """Full FFT display update (no blitting) - fallback method."""
    if not fft_cache['computed']:
        return
    
    positions = fft_cache['positions']
    idx = np.argmin(np.abs(positions - center_position))
    
    x_vals = fft_cache['x_vals']
    y_vals = fft_cache['y_vals'][idx, :]
    mode = fft_cache['mode']
    actual_position = fft_cache['positions'][idx]
    x_label = mode.split()[0]
    
    ax2.clear()
    ax2.plot(x_vals, y_vals, linewidth=0.8, color='green', alpha=0.8)
    ax2.set_xlabel(f"{mode}", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.set_title(f"FFT - Window {idx+1}/{len(fft_cache['positions'])} (Center: {actual_position})", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    if len(x_vals) > 0 and len(y_vals) > 0:
        ax2.set_xlim(min(x_vals), max(x_vals))
        ax2.set_ylim(0, max(y_vals) * 1.1)
        
        # Find and mark peak
        peak_idx = np.argmax(y_vals)
        peak_x = x_vals[peak_idx]
        peak_y = y_vals[peak_idx]
        ax2.axvline(peak_x, color='red', linestyle='--', alpha=0.5, linewidth=0.125)
        
        # Calculate FWHM
        fwhm, x_left, x_right = calculate_fwhm(x_vals, y_vals, peak_idx)
        
        # Display peak information
        text_x = peak_x * 1.02 if peak_x < np.median(x_vals) else peak_x * 0.98
        if fwhm is not None:
            # Draw FWHM visualization
            half_max = peak_y / 2.0
            ax2.hlines(half_max, x_left, x_right, colors='orange', linestyles='-', linewidth=1.5, alpha=0.7, label='FWHM')
            ax2.plot([x_left, x_right], [half_max, half_max], 'o', color='orange', markersize=4)
            
            # Display peak and FWHM
            ax2.text(text_x, peak_y * 0.95, 
                    f'Peak: {peak_x:.1f} nm\nFWHM: {fwhm:.2f} nm', 
                    fontsize=10, color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            # Only display peak (FWHM calculation failed)
            ax2.text(text_x, peak_y * 0.95, 
                    f'Peak: {peak_x:.1f} {x_label}', 
                    fontsize=10, color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    fig.canvas.draw_idle()


def check_result_queue():
    """Check for prepared data from background thread and update display."""
    try:
        result = result_queue.get_nowait()
        update_fft_display_blit(result)
    except:
        pass
    
    # Schedule next check
    if thread_running and blit_cache['enabled']:
        fig.canvas.get_tk_widget().after(10, check_result_queue)


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
    
    window_size = int(abs(selected_end - selected_start))
    
    if window_size < 4:
        print(f"Window too small ({window_size} points). Need at least 4 points.")
        status_text.set_text(f"❌ Window too small ({window_size} points)")
        return
    
    mode = domain_selector.value_selected
    
    # Compute all FFTs in parallel (multiprocessing)
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
    
    # Initialize blitting for fast updates
    init_blitting()
    
    # Start background thread
    start_background_thread()
    
    # Start checking for results
    check_result_queue()
    
    status_text.set_text(f"✓ {len(positions)} FFTs ready | Drag window for real-time updates (BLITTING ENABLED)")
    
    # Display first FFT
    update_fft_display_full(positions[0])
    
    print(f"FFT cache ready with {len(positions)} windows")
    print(f"Memory usage: ~{(y_matrix.nbytes / 1024 / 1024):.1f} MB")
    print("Real-time blitting mode: ACTIVE 🚀")


def onselect(xmin, xmax):
    """Handle span selection completion."""
    global selected_rect, selection_text, selected_start, selected_end
    
    selected_start = xmin
    selected_end = xmax
    
    if selected_rect:
        selected_rect.remove()
        selected_rect = None
    if selection_text:
        selection_text.remove()
        selection_text = None
    
    selected_rect = plt.Rectangle(
        (xmin, ax1.get_ylim()[0]),
        xmax - xmin,
        ax1.get_ylim()[1] - ax1.get_ylim()[0],
        linewidth=0.125,
        edgecolor='red',
        facecolor='red',
        alpha=0.2
    )
    ax1.add_patch(selected_rect)
    
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
    
    fig.canvas.draw_idle()


def on_span_motion(xmin, xmax):
    """Handle real-time updates while dragging - optimized with threading + blitting."""
    global selected_rect, selection_text
    
    if not fft_cache['computed']:
        return
    
    num_samples = int(abs(xmax - xmin))
    
    if fft_cache['window_size'] == num_samples:
        center_position = (xmin + xmax) / 2
        
        # Update visuals
        if selection_text:
            selection_text.set_position((xmin + (xmax - xmin) / 2, ax1.get_ylim()[1] * 0.98))
            selection_text.set_text(f"Window: {num_samples} samples")
        
        if selected_rect:
            selected_rect.set_x(xmin)
            selected_rect.set_width(xmax - xmin)
        
        # Queue FFT update (background thread will prepare data)
        if USE_BACKGROUND_THREAD:
            try:
                update_queue.get_nowait()  # Clear old request
            except:
                pass
            update_queue.put(center_position)
        else:
            # Fallback: direct update with blitting
            idx = np.argmin(np.abs(fft_cache['positions'] - center_position))
            y_vals_current = fft_cache['y_vals'][idx, :]
            peak_idx = np.argmax(y_vals_current)
            fwhm, x_left, x_right = calculate_fwhm(fft_cache['x_vals'], y_vals_current, peak_idx)
            
            result = {
                'idx': idx,
                'x_vals': fft_cache['x_vals'],
                'y_vals': y_vals_current,
                'mode': fft_cache['mode'],
                'actual_position': fft_cache['positions'][idx],
                'x_label': fft_cache['mode'].split()[0],
                'peak_x': fft_cache['x_vals'][peak_idx],
                'peak_y': np.max(y_vals_current),
                'fwhm': fwhm,
                'fwhm_left': x_left,
                'fwhm_right': x_right,
                'n_windows': len(fft_cache['positions'])
            }
            update_fft_display_blit(result)
        
        # Redraw top plot only
        fig.canvas.draw_idle()


def reset(event):
    """Reset the plots and selections."""
    global selected_rect, selected_start, selected_end, selection_text, data_values, fft_cache, blit_cache
    
    stop_background_thread()
    
    selected_start = None
    selected_end = None
    data_values = []
    fft_cache['computed'] = False
    blit_cache['enabled'] = False
    
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
print("Initializing ULTRA-OPTIMIZED Sliding Window FFT Analyzer...")
print(f"System: {multiprocessing.cpu_count()} CPU cores detected")
print(f"Configuration: {MAX_WORKERS} processes, {OVERLAP_PERCENTAGE}% overlap")
print("Multiprocessing: ProcessPoolExecutor (TRUE multi-core parallelism)")
print(f"Display optimization: Blitting + Background thread = MAXIMUM SPEED 🚀")

fft_button.on_clicked(run_fft)
reset_button.on_clicked(reset)
load_button.on_clicked(load_csv)

# Setup span selector with motion callback
span = SpanSelector(
    ax1, onselect, 'horizontal',
    useblit=True,
    props=dict(facecolor='red', alpha=0.2),
    interactive=True,
    drag_from_anywhere=True,
    onmove_callback=on_span_motion
)

status_text.set_text("Ready | Load CSV to begin")

plt.tight_layout()
plt.show()

# Cleanup
stop_background_thread()
