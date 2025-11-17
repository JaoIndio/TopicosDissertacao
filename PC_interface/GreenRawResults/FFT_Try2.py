#!/usr/bin/python3

import serial
import struct
import threading

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
from matplotlib.widgets import Button, RadioButtons, RectangleSelector
from scipy.signal.windows import hann, hamming
from scipy.signal import fftconvolve
#from scipy.fftpack import fft
from numpy.fft import fft 

from scipy.signal import detrend, windows
import numpy as np

# Configuration
SAMPLING_INTERVAL_NM = 45.8996  # Sampling interval in nanometers
# IMPORTANT: This is configured for a SINGLE-PASS interferometer

# Initialize plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(top=0.85, right=0.85)

# Global variables for selection
selected_rect = None
selected_start = None
selected_end = None
selection_text = None
data_values = []

# GUI Elements
filename_ax = plt.axes([0.075, 0.97, 0.15, 0.025])
filename_box = TextBox(filename_ax, 'CSV File:', initial="acquisition_226.csv")

radio_ax = plt.axes([0.2, 0.475, 0.20, 0.045])
domain_selector = RadioButtons(radio_ax, ('Wavelength (nm)', 'Wavenumber (cm⁻¹)'))

fft_ax = plt.axes([0.0075, 0.475, 0.035, 0.035])
fft_button = Button(fft_ax, 'FFT')

reset_ax = plt.axes([0.0525, 0.475, 0.05, 0.035])
reset_button = Button(reset_ax, 'Reset')

load_ax = plt.axes([0.1125, 0.475, 0.08, 0.035])
load_button = Button(load_ax, 'Load CSV')


def read_csv_values(filename):
    """Reads numeric values from CSV file (skipping metadata rows)."""
    values = []
    
    try:
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            # Skip metadata rows
            next(reader, None)  # Skip first row
            next(reader, None)  # Skip second row
            
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
    global data_values
    csv_filename = filename_box.text.strip()
    
    data_values = read_csv_values(csv_filename)
    if not data_values:
        print(f"No data loaded from {csv_filename}")
        return
    
    ax1.clear()
    ax1.plot(data_values, label='Sensor Data', linewidth=0.125)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Data from {csv_filename}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.canvas.draw_idle()


def onselect(xmin, xmax):
    """Handle span selection on the plot."""
    global selected_rect, selection_text, selected_start, selected_end
    
    # Store selection boundaries
    selected_start = xmin
    selected_end = xmax
    
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
        linewidth=0.125,
        edgecolor='red',
        facecolor='none',
        alpha=0.5
    )
    ax1.add_patch(selected_rect)
    
    # Add text showing number of samples
    num_samples = int(abs(xmax - xmin))
    selection_text = ax1.text(
        xmin + (xmax - xmin) / 2,
        ax1.get_ylim()[1] * 0.98,
        f"Samples: {num_samples}",
        color='red',
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    
    fig.canvas.draw_idle()


def run_fft(event):
    """Perform FFT on selected data region for SINGLE-PASS interferometer."""
    global data_values
    
    if not data_values:
        print("No data loaded. Please load a CSV file first.")
        return
    
    if selected_start is None or selected_end is None:
        print("No selection made. Please select a region of data.")
        return
    
    # Extract selected data
    i_start = int(max(0, min(selected_start, selected_end)))
    i_end = int(min(len(data_values), max(selected_start, selected_end)))
    
    if i_end - i_start < 4:
        print(f"Selection too small ({i_end - i_start} points). Need at least 4 points.")
        return
    
    selected_data = np.array(data_values[i_start:i_end])
    
    # Preprocessing for FFT
    # 1. Remove DC component/
    selected_data = selected_data - np.mean(selected_data)
    
    # 2. Detrend to remove linear trends
    selected_data = detrend(selected_data)
    
    # 3. Apply window function to reduce spectral leakage
    window = windows.hann(len(selected_data))
    windowed_data = selected_data * window
    
    # Perform FFT
    N = len(windowed_data)
    fft_result = np.fft.fft(windowed_data)
    
    # Get magnitude spectrum (single-sided)
    # Normalize by N for amplitude spectrum
    # Multiply by 2 for single-sided spectrum (except DC and Nyquist)
    fft_magnitude = 2.0 * np.abs(fft_result[:N//2]) / N
    fft_magnitude[0] = fft_magnitude[0] / 2.0  # Correct DC component
    
    # Calculate frequency axis
    # For interferometry: spatial frequency = 1 / (sampling_interval * N_samples)
    dx_m = SAMPLING_INTERVAL_NM * 1e-9  # Convert nm to meters
    
    # Spatial frequencies in cycles per meter
    freq_spatial_m = np.fft.fftfreq(N, d=dx_m)[:N//2]
    
    # Convert to appropriate units based on selection
    display_mode = domain_selector.value_selected
    
    if display_mode == 'Wavelength (nm)':
        # For SINGLE-PASS interferometer: wavelength = 1 / spatial_frequency
        # NO factor of 2!
        # Avoid division by zero
        valid_idx = freq_spatial_m > 0
        wavelengths_m = np.zeros_like(freq_spatial_m)
        wavelengths_m[valid_idx] = 1.0 /(2.0 * freq_spatial_m[valid_idx])  # 
        wavelengths_nm = wavelengths_m * 1e9
        
        # Filter reasonable wavelength range
        mask = (wavelengths_nm > 100) & (wavelengths_nm < 3000)
        x_vals = wavelengths_nm[mask]
        y_vals = fft_magnitude[mask]
        x_label = "Wavelength (nm)"
        
        # Sort by wavelength for proper plotting
        sort_idx = np.argsort(x_vals)
        x_vals = x_vals[sort_idx]
        y_vals = y_vals[sort_idx]
        
    else:  # Wavenumber (cm⁻¹)
        # For SINGLE-PASS: Wavenumber = spatial_frequency (in m⁻¹) * 0.01 (to convert to cm⁻¹)
        # NO factor of 2 division!
        wavenumbers_cm = 0.5 * freq_spatial_m * 0.01  # m⁻¹ to cm⁻¹ conversion
        
        # Filter reasonable wavenumber range
        mask = (wavenumbers_cm > 0) & (wavenumbers_cm < 10000)
        x_vals = wavenumbers_cm[mask]
        y_vals = fft_magnitude[mask]
        x_label = "Wavenumber (cm⁻¹)"
    
    # Plot FFT result
    ax2.clear()
    ax2.plot(x_vals, y_vals, linewidth=0.125)
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.set_title(f"FFT Result - Double Pass ({N} points, Δx = {SAMPLING_INTERVAL_NM:.2f} nm)", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    if len(x_vals) > 0:
        ax2.set_xlim(min(x_vals), max(x_vals))
        ax2.set_ylim(0, max(y_vals) * 1.1)
    
    # Add some statistics and peak identification
    if len(y_vals) > 0:
        peak_idx = np.argmax(y_vals)
        peak_x = x_vals[peak_idx]
        peak_y = y_vals[peak_idx]
        ax2.axvline(peak_x, color='red', linestyle='--', alpha=0.5)
        
        # Position text to avoid overlap
        text_x = peak_x * 1.02 if peak_x < np.median(x_vals) else peak_x * 0.98
        ax2.text(text_x, peak_y * 0.95, 
                f'Peak: {peak_x:.1f} {x_label.split()[0]}', 
                fontsize=10, color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Print some debug information
    print(f"FFT Analysis (Double-Pass Interferometer):")
    print(f"  Number of points: {N}")
    print(f"  Sampling interval: {SAMPLING_INTERVAL_NM} nm")
    print(f"  Frequency resolution: {freq_spatial_m[1] - freq_spatial_m[0]:.2e} m⁻¹")
    if display_mode == 'Wavenumber (cm⁻¹)' and len(x_vals) > 1:
        print(f"  Wavenumber resolution: {x_vals[1] - x_vals[0]:.2f} cm⁻¹")
    
    fig.canvas.draw_idle()


def reset(event):
    """Reset the plots and selections."""
    global selected_rect, selected_start, selected_end, selection_text, data_values
    
    selected_start = None
    selected_end = None
    data_values = []
    
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
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.set_title('FFT Result - Double Pass Interferometer', fontsize=14)
    fig.canvas.draw_idle()


# Initialize plots
ax1.set_xlabel('Sample Index', fontsize=12)
ax1.set_ylabel('Intensity', fontsize=12)
ax1.set_title('Load data to begin', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Frequency', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title('FFT Result - Double Pass Interferometer', fontsize=14)
ax2.grid(True, alpha=0.3)

# Connect event handlers
print("Connect event handlers")
fft_button.on_clicked(run_fft)
reset_button.on_clicked(reset)
load_button.on_clicked(load_csv)

# Setup span selector for interactive region selection
print("Setup span selector for interactive region selection")
span = SpanSelector(
    ax1, onselect, 'horizontal',
    useblit=True,
    props=dict(facecolor='red', alpha=0.2),
    interactive=True,
    drag_from_anywhere=True
)

plt.tight_layout()
plt.show()
