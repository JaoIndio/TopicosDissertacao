#!/usr/bin/python3
"""
Sliding Window FFT Analyzer - ULTRA-OPTIMIZED with FIXED Blitting
Pre-computes all FFTs + uses blitting for instantaneous display updates
WITH MANUAL WINDOW SIZE ENTRY + LARGE DATASET OPTIMIZATIONS

FIXES APPLIED:
- Removed flush_events() that caused hanging
- Added matplotlib performance optimizations for large datasets
- Fixed blit() timing issues
- Added proper error handling for blitting failures
- Optimized line rendering for massive datasets

HARDWARE OPTIMIZATION:
- Configured for: AMD Ryzen 5 PRO 5675U (6 cores / 12 threads)
- Memory: 16 GB DDR4-2400 dual-channel
- Process usage: 11 workers (92% utilization)
- Overlap: 85% for ultra-smooth display
"""

import serial
import struct
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from queue import Queue, Empty
from threading import Thread, Lock

import csv
import os
import glob

import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")

# MATPLOTLIB PERFORMANCE OPTIMIZATIONS FOR LARGE DATASETS
import matplotlib.pyplot as plt
matplotlib.rcParams['path.simplify'] = True  # Simplify paths for faster rendering
matplotlib.rcParams['path.simplify_threshold'] = 1.0  # Aggressive simplification
matplotlib.rcParams['agg.path.chunksize'] = 10000  # Render in chunks for large data
matplotlib.rcParams['figure.max_open_warning'] = 0  # Disable warnings

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, SpanSelector, Button, RadioButtons
from scipy.signal import detrend, windows
import numpy as np
import time

# Configuration
SAMPLING_INTERVAL_NM = 45.8996
OVERLAP_PERCENTAGE = 85
MAX_WORKERS = 11
USE_BACKGROUND_THREAD = True

# Global variables
selected_rect = None
selected_start = None
selected_end = None
selection_text = None
data_values = []
window_size = None
manual_window_center = None

# Blitting cache
blit_cache = {
    'background': None,
    'line': None,
    'peak_line': None,
    'peak_text': None,
    'fwhm_line': None,
    'fwhm_markers': None,
    'enabled': False
}

# Background thread communication
update_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)
data_prep_thread = None
thread_running = False

# FFT Cache
fft_cache = {
    'x_vals': None,
    'y_vals': None,
    'positions': None,
    'window_size': None,
    'mode': None,
    'computed': False
}

# Initialize plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(top=0.85, right=0.85, bottom=0.08)

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

window_size_ax = plt.axes([0.425, 0.475, 0.08, 0.035])
window_size_box = TextBox(window_size_ax, 'Win:', initial="512")

set_window_ax = plt.axes([0.51, 0.475, 0.08, 0.035])
set_window_button = Button(set_window_ax, 'Set Window')

zoom_reset_ax = plt.axes([0.6, 0.475, 0.08, 0.035])
zoom_reset_button = Button(zoom_reset_ax, 'Reset Zoom')

status_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10, color='blue', weight='bold')

processed_region_patch = None


def read_csv_values(filename):
    """Reads numeric values from CSV file."""
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
    """Load and plot CSV file - OPTIMIZED for large datasets."""
    global data_values, fft_cache, blit_cache
    csv_filename = filename_box.text.strip()
    
    fft_cache['computed'] = False
    blit_cache['enabled'] = False
    
    data_values = read_csv_values(csv_filename)
    if not data_values:
        status_text.set_text(f"❌ Failed to load {csv_filename}")
        return
    
    ax1.clear()
    
    # OPTIMIZATION: Downsample if dataset is huge (>100k points)
    if len(data_values) > 100000:
        # Plot every Nth point for initial display
        stride = len(data_values) // 50000
        ax1.plot(range(0, len(data_values), stride), data_values[::stride], 
                linewidth=0.3, color='blue', alpha=0.7, rasterized=True)
        print(f"Downsampled plot for performance: showing every {stride}th point")
    else:
        ax1.plot(data_values, linewidth=0.5, color='blue', alpha=0.7)
    
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Data: {csv_filename} ({len(data_values)} points) | Zoom in to analyze', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    status_text.set_text(f"✓ Loaded {len(data_values)} points | Zoom to region, select window, press FFT")
    fig.canvas.draw()


def set_manual_window(event):
    """Set window size manually from text box."""
    global selected_rect, selection_text, selected_start, selected_end, manual_window_center
    
    if not data_values:
        status_text.set_text("❌ No data loaded")
        return
    
    try:
        manual_size = int(window_size_box.text.strip())
        
        if manual_size < 4 or manual_size > len(data_values):
            status_text.set_text(f"❌ Invalid window size")
            return
        
        if selected_rect:
            selected_rect.remove()
        if selection_text:
            selection_text.remove()
        
        if manual_window_center is None:
            manual_window_center = len(data_values) / 2
        
        selected_start = manual_window_center - manual_size / 2
        selected_end = manual_window_center + manual_size / 2
        
        if selected_start < 0:
            selected_start = 0
            selected_end = manual_size
        elif selected_end > len(data_values):
            selected_end = len(data_values)
            selected_start = selected_end - manual_size
        
        selected_rect = plt.Rectangle(
            (selected_start, ax1.get_ylim()[0]),
            selected_end - selected_start,
            ax1.get_ylim()[1] - ax1.get_ylim()[0],
            linewidth=2,
            edgecolor='blue',
            facecolor='blue',
            alpha=0.2
        )
        ax1.add_patch(selected_rect)
        
        selection_text = ax1.text(
            (selected_start + selected_end) / 2,
            ax1.get_ylim()[1] * 0.98,
            f"Manual: {manual_size} samples",
            color='blue',
            ha='center',
            fontsize=10,
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )
        
        status_text.set_text(f"✓ Window set: {manual_size} samples | Press FFT")
        fig.canvas.draw()
        
    except ValueError:
        status_text.set_text(f"❌ Invalid window size")


def compute_single_fft(args):
    """Compute FFT for single window."""
    data_segment, idx, mode, sampling_interval = args
    
    N = len(data_segment)
    data_proc = data_segment - np.mean(data_segment)
    data_proc = detrend(data_proc)
    window = windows.hann(N)
    windowed_data = data_proc * window
    
    fft_result = np.fft.fft(windowed_data)
    fft_magnitude = 2.0 * np.abs(fft_result[:N//2]) / N
    fft_magnitude[0] = fft_magnitude[0] / 2.0
    
    dx_m = sampling_interval * 1e-9
    freq_spatial_m = np.fft.fftfreq(N, d=dx_m)[:N//2]
    
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
    else:
        wavenumbers_cm = 0.5 * freq_spatial_m * 0.01
        mask = (wavenumbers_cm > 0) & (wavenumbers_cm < 10000)
        x_vals = wavenumbers_cm[mask]
        y_vals = fft_magnitude[mask]
    
    return idx, x_vals, y_vals


def compute_all_ffts_parallel(data, win_size, mode, sampling_interval):
    """Compute all FFTs in parallel using TRUE multiprocessing."""
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
    print(f"\nComputing {n_windows} FFTs | Window: {win_size} | Hop: {hop_size}")
    print(f"Using {MAX_WORKERS} processes (multiprocessing)")
    
    status_text.set_text(f"⏳ Computing {n_windows} FFTs...")
    fig.canvas.draw()
    
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
                print(f"Progress: {progress:.1f}% ({completed}/{n_windows})")
    
    elapsed = time.time() - start_time
    print(f"✓ Done in {elapsed:.2f}s ({n_windows/elapsed:.1f} FFTs/sec)\n")
    
    x_axis = results[0][0]
    n_freqs = len(x_axis)
    y_matrix = np.zeros((n_windows, n_freqs), dtype=np.float32)
    for idx in range(n_windows):
        y_matrix[idx, :] = results[idx][1]
    
    return x_axis, y_matrix, np.array(positions)


def calculate_fwhm(x_vals, y_vals, peak_idx):
    """Calculate Full Width at Half Maximum."""
    peak_y = y_vals[peak_idx]
    half_max = peak_y / 2.0
    
    left_indices = np.where((x_vals < x_vals[peak_idx]) & (y_vals <= half_max))[0]
    if len(left_indices) == 0:
        return None, None, None
    left_idx = left_indices[-1]
    
    right_indices = np.where((x_vals > x_vals[peak_idx]) & (y_vals <= half_max))[0]
    if len(right_indices) == 0:
        return None, None, None
    right_idx = right_indices[0]
    
    x_left = x_vals[left_idx]
    x_right = x_vals[right_idx]
    fwhm = abs(x_right - x_left)
    
    return fwhm, x_left, x_right


def data_preparation_worker():
    """Background thread for preparing FFT data."""
    global thread_running
    
    print("Background thread started")
    
    while thread_running:
        try:
            center_position = update_queue.get(timeout=0.1)
            
            if not fft_cache['computed']:
                continue
            
            positions = fft_cache['positions']
            idx = np.argmin(np.abs(positions - center_position))
            
            x_vals = fft_cache['x_vals']
            y_vals = fft_cache['y_vals'][idx, :]
            mode = fft_cache['mode']
            
            peak_idx = np.argmax(y_vals)
            peak_x = x_vals[peak_idx]
            peak_y = y_vals[peak_idx]
            
            fwhm, x_left, x_right = calculate_fwhm(x_vals, y_vals, peak_idx)
            
            result = {
                'idx': idx,
                'x_vals': x_vals,
                'y_vals': y_vals,
                'mode': mode,
                'actual_position': fft_cache['positions'][idx],
                'x_label': mode.split()[0],
                'peak_x': peak_x,
                'peak_y': peak_y,
                'fwhm': fwhm,
                'fwhm_left': x_left,
                'fwhm_right': x_right,
                'n_windows': len(fft_cache['positions'])
            }
            
            try:
                result_queue.get_nowait()
            except Empty:
                pass
            result_queue.put(result)
            
        except Empty:
            continue
        except Exception as e:
            print(f"Background thread error: {e}")
            continue


def start_background_thread():
    """Start background data prep thread."""
    global data_prep_thread, thread_running
    
    if not thread_running:
        thread_running = True
        data_prep_thread = Thread(target=data_preparation_worker, daemon=True)
        data_prep_thread.start()


def stop_background_thread():
    """Stop background thread."""
    global thread_running
    thread_running = False


def init_blitting():
    """Initialize blitting for fast updates - FIXED VERSION."""
    global blit_cache
    
    if not fft_cache['computed']:
        return
    
    # Clear and setup
    ax2.clear()
    ax2.set_xlabel(f"{fft_cache['mode']}", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Create animated artists
    blit_cache['line'], = ax2.plot([], [], linewidth=0.8, color='green', alpha=0.8, animated=True)
    blit_cache['peak_line'] = ax2.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1, animated=True)
    blit_cache['fwhm_line'] = ax2.hlines(0, 0, 1, colors='orange', linestyles='-', linewidth=1.5, alpha=0.7, animated=True)
    blit_cache['fwhm_markers'], = ax2.plot([], [], 'o', color='orange', markersize=4, animated=True)
    blit_cache['peak_text'] = ax2.text(0, 0, '', fontsize=10, color='red',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                        animated=True)
    
    # Draw once to capture background
    fig.canvas.draw()
    fig.canvas.flush_events()  # OK here, just once
    
    # Small delay to ensure canvas is ready
    time.sleep(0.05)
    
    # Capture background
    try:
        blit_cache['background'] = fig.canvas.copy_from_bbox(ax2.bbox)
        blit_cache['enabled'] = True
        print("✓ Blitting initialized successfully")
    except Exception as e:
        print(f"Blitting init failed: {e}")
        blit_cache['enabled'] = False


def update_fft_display_blit(result_data):
    """Update FFT plot using blitting - FIXED VERSION (No Hang!)."""
    
    if not blit_cache['enabled'] or blit_cache['background'] is None:
        # Fallback to full redraw
        print("1")
        update_fft_display_full(result_data.get('actual_position', 0))
        return
    
    try:
        # Restore background
        print("2")
        fig.canvas.restore_region(blit_cache['background'])
        
        # Update line data
        print("3")
        blit_cache['line'].set_data(result_data['x_vals'], result_data['y_vals'])
        
        # Update peak line
        print("4")
        peak_x = result_data['peak_x']
        peak_y = result_data['peak_y']
        blit_cache['peak_line'].set_xdata([peak_x, peak_x])
        blit_cache['peak_line'].set_ydata([0, peak_y])
        
        # Update FWHM
        print("5")
        if result_data['fwhm'] is not None:
            half_max = peak_y / 2.0
            blit_cache['fwhm_line'].set_segments([[(result_data['fwhm_left'], half_max), 
                                                     (result_data['fwhm_right'], half_max)]])
            blit_cache['fwhm_markers'].set_data([result_data['fwhm_left'], result_data['fwhm_right']], 
                                                 [half_max, half_max])
            
            text_x = peak_x + 100
            blit_cache['peak_text'].set_position((text_x, peak_y * 0.80))
            blit_cache['peak_text'].set_text(
                f"Peak: {peak_x:.1f} {result_data['x_label']}\nFWHM: {result_data['fwhm']:.1f}"
            )
        else:
            # Hide FWHM
            print("6")
            blit_cache['fwhm_line'].set_segments([[(0, 0), (0, 0)]])
            blit_cache['fwhm_markers'].set_data([], [])
            blit_cache['peak_text'].set_position((peak_x + 100, peak_y * 0.80))
            blit_cache['peak_text'].set_text(f"Peak: {peak_x:.1f}")
        
        # Redraw artists
        print("7")
        ax2.draw_artist(blit_cache['line'])
        ax2.draw_artist(blit_cache['peak_line'])
        ax2.draw_artist(blit_cache['fwhm_line'])
        ax2.draw_artist(blit_cache['fwhm_markers'])
        ax2.draw_artist(blit_cache['peak_text'])
        
        # CRITICAL FIX: Just blit, nothing else!
        print("7_1")
        fig.canvas.blit(ax2.bbox)
        print("8")
        fig.canvas.flush_events()
        print("9")
        # NO flush_events() - causes hang!
        # NO draw_idle() - defeats blitting purpose!
        
    except Exception as e:
        print(f"Blit error: {e}, disabling blitting")
        blit_cache['enabled'] = False
        update_fft_display_full(result_data.get('actual_position', 0))


def update_fft_display_full(center_position):
    """Full redraw fallback."""
    if not fft_cache['computed']:
        return
    
    positions = fft_cache['positions']
    idx = np.argmin(np.abs(positions - center_position))
    
    x_vals = fft_cache['x_vals']
    y_vals = fft_cache['y_vals'][idx, :]
    
    ax2.clear()
    ax2.plot(x_vals, y_vals, linewidth=0.8, color='green')
    ax2.set_xlabel(f"{fft_cache['mode']}", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.set_title(f"FFT - Window {idx+1}/{len(positions)}", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    if len(y_vals) > 0:
        peak_idx = np.argmax(y_vals)
        peak_x = x_vals[peak_idx]
        peak_y = y_vals[peak_idx]
        ax2.axvline(peak_x, color='red', linestyle='--', alpha=0.5)
        ax2.text(peak_x + 100, peak_y * 0.95, f'Peak: {peak_x:.1f}', fontsize=10, color='red')
    
    fig.canvas.draw_idle()


def check_result_queue():
    """Check for prepared data and update display."""
    try:
        result = result_queue.get_nowait()
        update_fft_display_blit(result)
    except Empty:
        pass
    
    if thread_running and blit_cache['enabled']:
        fig.canvas.get_tk_widget().after(10, check_result_queue)


def zoom_reset(event):
    """Reset zoom."""
    if data_values:
        ax1.set_xlim(0, len(data_values))
        ax1.set_ylim(min(data_values) - 0.1, max(data_values) + 0.1)
        fig.canvas.draw()


def run_fft(event):
    """Compute FFTs for visible region."""
    global fft_cache, window_size, processed_region_patch
    
    if not data_values or selected_start is None or selected_end is None:
        status_text.set_text("❌ No selection")
        return
    
    window_size = int(abs(selected_end - selected_start))
    
    if window_size < 4:
        status_text.set_text("❌ Window too small")
        return
    
    # Get visible region
    xlim = ax1.get_xlim()
    visible_start = max(0, int(xlim[0]))
    visible_end = min(len(data_values), int(xlim[1]))
    visible_data = data_values[visible_start:visible_end]
    
    if len(visible_data) < window_size:
        status_text.set_text("❌ Zoom out or reduce window size")
        return
    
    # Visual indicator
    if processed_region_patch:
        processed_region_patch.remove()
    processed_region_patch = plt.Rectangle(
        (visible_start, ax1.get_ylim()[0]),
        visible_end - visible_start,
        ax1.get_ylim()[1] - ax1.get_ylim()[0],
        linewidth=2,
        edgecolor='green',
        facecolor='green',
        alpha=0.05
    )
    ax1.add_patch(processed_region_patch)
    
    mode = domain_selector.value_selected
    
    # Compute FFTs
    x_axis, y_matrix, positions_rel = compute_all_ffts_parallel(
        np.array(visible_data), window_size, mode, SAMPLING_INTERVAL_NM
    )
    
    positions_abs = positions_rel + visible_start
    
    # Store in cache
    fft_cache['x_vals'] = x_axis
    fft_cache['y_vals'] = y_matrix
    fft_cache['positions'] = positions_abs
    fft_cache['window_size'] = window_size
    fft_cache['mode'] = mode
    fft_cache['computed'] = True
    
    # Initialize blitting
    init_blitting()
    
    # Start background thread
    start_background_thread()
    check_result_queue()
    
    status_text.set_text(f"✓ {len(positions_abs)} FFTs ready | Drag window")
    
    # Show first FFT
    if len(positions_abs) > 0:
        update_fft_display_full(positions_abs[0])


def onselect(xmin, xmax):
    """Handle span selection."""
    global selected_rect, selection_text, selected_start, selected_end, manual_window_center
    
    selected_start = xmin
    selected_end = xmax
    manual_window_center = (xmin + xmax) / 2
    
    if selected_rect:
        selected_rect.remove()
    if selection_text:
        selection_text.remove()
    
    selected_rect = plt.Rectangle(
        (xmin, ax1.get_ylim()[0]),
        xmax - xmin,
        ax1.get_ylim()[1] - ax1.get_ylim()[0],
        linewidth=2,
        edgecolor='red',
        facecolor='red',
        alpha=0.2
    )
    ax1.add_patch(selected_rect)
    
    num_samples = int(abs(xmax - xmin))
    selection_text = ax1.text(
        (xmin + xmax) / 2,
        ax1.get_ylim()[1] * 0.98,
        f"Window: {num_samples}",
        color='red',
        ha='center',
        fontsize=10,
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
    )
    
    fig.canvas.draw_idle()


def on_span_motion(xmin, xmax):
    """Real-time updates while dragging."""
    if not fft_cache['computed']:
        return
    
    num_samples = int(abs(xmax - xmin))
    
    if fft_cache['window_size'] == num_samples:
        center_position = (xmin + xmax) / 2
        
        # Update visuals
        if selection_text:
            selection_text.set_position(((xmin + xmax) / 2, ax1.get_ylim()[1] * 0.98))
        if selected_rect:
            selected_rect.set_x(xmin)
            selected_rect.set_width(xmax - xmin)
        
        # Queue update
        try:
            update_queue.get_nowait()
        except Empty:
            pass
        update_queue.put(center_position)


def reset(event):
    """Reset everything."""
    global selected_rect, selected_start, selected_end, selection_text, data_values
    global fft_cache, blit_cache, manual_window_center, processed_region_patch
    
    stop_background_thread()
    
    selected_start = None
    selected_end = None
    data_values = []
    manual_window_center = None
    fft_cache['computed'] = False
    blit_cache['enabled'] = False
    
    if selected_rect:
        selected_rect.remove()
        selected_rect = None
    if selection_text:
        selection_text.remove()
        selection_text = None
    if processed_region_patch:
        processed_region_patch.remove()
        processed_region_patch = None
    
    ax1.clear()
    ax2.clear()
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('Load data to begin', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.set_title('FFT Result', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    status_text.set_text("Ready")
    fig.canvas.draw_idle()


# Initialize
ax1.set_xlabel('Sample Index', fontsize=12)
ax1.set_ylabel('Intensity', fontsize=12)
ax1.set_title('Load data | Use zoom tools', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Frequency', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title('FFT Result', fontsize=14)
ax2.grid(True, alpha=0.3)

# Connect events
print("\nInitializing OPTIMIZED FFT Analyzer...")
print(f"CPU cores: {multiprocessing.cpu_count()}")
print(f"Workers: {MAX_WORKERS} processes")
print(f"Overlap: {OVERLAP_PERCENTAGE}%")
print("Matplotlib optimizations: ENABLED")
print("Blitting: FIXED (no hang!)")

fft_button.on_clicked(run_fft)
reset_button.on_clicked(reset)
load_button.on_clicked(load_csv)
set_window_button.on_clicked(set_manual_window)
zoom_reset_button.on_clicked(zoom_reset)

span = SpanSelector(
    ax1, onselect, 'horizontal',
    useblit=True,
    props=dict(facecolor='red', alpha=0.2),
    interactive=True,
    drag_from_anywhere=True,
    onmove_callback=on_span_motion
)

status_text.set_text("Ready | Load CSV")

plt.tight_layout()
plt.show()

stop_background_thread()
