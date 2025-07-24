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
START_BYTE = 0xAA
STOP_BYTE = 0x55
PORT = '/dev/ttyUSB0'  # Serial port for your USB-to-UART adapter
BAUD_RATE = 921600*1    # 4.5Mbps
ADC_SIZE = 4         # Size of each float in bytes
ADC_FULLSCALE = 16261         # Size of each float in bytes
    
WAVELENGHT_SIZE  = 1024*40
DATA_POINTS  = 10000*10*10
# Initialize serial port
#ser = serial.Serial(PORT, BAUD_RATE, timeout=1, parity='N')
gIndex =0

# Data storage for plotting
data_counts = []
data_values = []
max_values_over_time = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(top=0.85, right=0.85)


selected_region = [None, None]
selected_rect = None
selected_start = None
selected_end = None
selection_text = None  # Store the text object

max_value = 0
max_count = 0
max_values_over_time = []
data = np.array([])

# Text box to enter filename
#filename_ax = plt.axes([0.1, 0.15, 0.3, 0.05])
filename_ax = plt.axes([0.075, 0.97, 0.15, 0.025])
filename_box = TextBox(filename_ax, 'CSV Filename:', initial="acquisition_226.csv")

# Radio buttons to choose domain display
radio_ax = plt.axes([0.2, 0.475, 0.20, 0.045])
domain_selector = RadioButtons(radio_ax, ('Wavelength (nm)', 'Wavenumber (cm-1)'))

# FFT Button
fft_ax = plt.axes([0.0075, 0.475, 0.035, 0.035])
fft_button = Button(fft_ax, 'FFT')

# Reset Button
reset_ax = plt.axes([0.0525, 0.475, 0.05, 0.035])
reset_button = Button(reset_ax, 'Reset')

# Load CSV Button
load_ax = plt.axes([0.1125, 0.475, 0.08, 0.035])
load_button = Button(load_ax, 'Load CSV')


DIR = "./GreenRawResults"
filename = "."


def read_csv_values(filename):
  """Reads the numeric values from the CSV file (skipping metadata rows)."""
  values = []
            
  with open(filename, mode="r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the first row (Name)
    next(reader)  # Skip the second row (Description)
                                                        
    for row in reader:
      if row:  # Ensure it's not an empty row
        try:
          values.append(float(row[0]))  # Convert to float
        except(ValueError, IndexError):
          continue

  return values

def load_csv(event):
  global data_values
  """Loads and plots the CSV file contents on ax1."""
  csv_filename = filename_box.text.strip()

  try:
    data_values = read_csv_values(csv_filename)
  except Exception as e:
    print(f"Failed to load", {csv_filename}, ": ", {e})
    return

  ax1.clear()
  ax1.plot(data_values, label='Sensor Data')
  ax1.set_xlabel('Sample Index (Displacement)', fontsize=20)
  ax1.set_ylabel('Intensity',fontsize=20)
  ax1.set_title(f'Data from {csv_filename}',fontsize=20)
  ax1.legend()
  fig.canvas.draw_idle()
# Mouse interaction handlers
def on_press(event):
  global selected_rect, selected_start, selection_text
  if event.inaxes != ax1:
    return
  
  if selected_rect:
    selected_rect.remove()
    selected_rect = None
  if selection_text:
    selection_text.remove()
    selection_text = None

  selected_start = event.xdata

def on_release(event):
  global selected_rect, selected_end, selected_start, selection_text
  if event.inaxes != ax1 or selected_start is None:
    return

  selected_end = event.xdata
  x0 = min(selected_start, selected_end)
  width = abs(selected_end - selected_start)

  # Remove previous rectangle if exists
  if selected_rect:
    selected_rect.remove()
  if selection_text:
    selection_text.remove()

  selected_rect = plt.Rectangle((x0, ax1.get_ylim()[0]), width, ax1.get_ylim()[1] - ax1.get_ylim()[0],
                             linewidth=1, edgecolor='red', facecolor='none')
  ax1.add_patch(selected_rect)
  #ax1.text(x0 + width / 2, ax1.get_ylim()[1], f"Samples: {int(width)}", color='red', ha='center')
  # Add text
  selection_text = ax1.text(x0 + width/2,
                            ax1.get_ylim()[1],
                            f"Samples: {int(width)}",
                            color='red',
                            ha='center')

  fig.canvas.draw_idle()

def run_fft(event):
  global data_values
  csv_filename = filename_box.text.strip()

  #try:
  #    data = np.loadtxt(csv_filename, delimiter=',')
  #except Exception as e:
  #    print(f"Failed to load {csv_filename}: {e}")
  #    return

  if selected_start is None or selected_end is None:
    print("No selection made.")
    return
  data= data_values

  i_start = int(min(selected_start, selected_end))
  i_end = int(max(selected_start, selected_end))
  selected_data = data[i_start:i_end]

  if len(selected_data) < 2:
    print("Selection too small. i_start", i_start, " i_end: ", i_end)
    print("selected_start", selected_start, " selected_end: ", selected_end)
    print("Total data points:", len(data))
    return

  # Remove DC component
  selected_data -= np.mean(selected_data)
  # Detrend to remove linear trends
  selected_data = detrend(selected_data)
  # Apply Hann window to reduce edge effects
  window = windows.hann(len(selected_data))
  selected_data = selected_data * window

  # FFT
  fft_result = np.fft.fft(selected_data)
  # Magnitude spectrum (normalize by N for amplitude)
  fft_magnitude = np.abs(fft_result) / len(selected_data)
  #fft_vals = np.abs(fft_vals[:len(fft_vals)//2])
  # Only keep positive frequencies (up to Nyquist)
  N = len(selected_data)
  fft_magnitude = fft_magnitude[:N//2]

  N = len(selected_data)
  dx = 75.301e-9  # 75.301 nm in meters

  #freqs = np.fft.fftfreq(N, d=dx)[:N//2]
  delta_x_nm = 75.301  # Sampling interval in nm
  delta_x_mm = delta_x_nm * 1e-6  # Convert to mm

  freqs = np.fft.fftfreq(N)[:N//2]
  valid = (freqs > 0)
  freqs = freqs[valid]
  fft_magnitude = fft_magnitude[valid]
  
  freq_mm = freqs / delta_x_mm
  freq_m = freq_mm * 1000  # Spatial frequency in cycles/m
  # Spectroscopic Wavenumber (cm⁻¹)
  # For single-side interferometer, σ = f / 2 (m⁻¹), then convert to cm⁻¹
  sigma_cm = (freq_m / 2) * 1e-2  # 1 m⁻¹ = 10⁻² cm⁻¹


  display_mode = domain_selector.value_selected
  fft_mag = fft_magnitude
  x_vals = sigma_cm
  if display_mode == 'Wavelength (nm)':
    #wavelengths = 1 / freqs  # in meters
    #x_vals = wavelengths * 1e9  # convert to nm
    x_label = "Wavelength (nm)"
    # Wavelength (nm)
# For double-pass interferometer, λ = 2 / f (m), then to nm
    wavelength_nm = (2 / freq_m) * 1e9  # λ_m * 10⁹ = λ_nm
    wavelength_nm = wavelength_nm[::-1]  # Reverse for increasing wavelength
    x_vals = wavelength_nm
    fft_magnitude_wavelength = fft_magnitude[::-1]
    fft_mag = fft_magnitude_wavelength


  elif display_mode == 'Wavenumber (cm⁻¹)':
    #wavenumbers = freqs / 100  # convert 1/m to 1/cm
    #x_vals = wavenumbers
    x_vals = sigma_cm 
    x_label = "Wavenumber (cm⁻¹)"
    fft_mag = fft_magnitude
  else:
    #x_vals = freqs
    #x_label = "Frequency (1/m)"
    x_vals = sigma_cm 
    x_label = "Wavenumber (cm⁻¹)"
    fft_mag = fft_magnitude

  ax2.clear()
  ax2.plot(x_vals, fft_mag)
  ax2.set_xlabel(x_label, fontsize=20)
  ax2.set_ylabel("Amplitude", fontsize=20)
  ax2.set_title("FFT Result", fontsize=20)
  ax2.tick_params(axis='y', labelsize=20)
  ax2.tick_params(axis='x', labelsize=20)

  if display_mode == 'Wavelength (nm)':
    ax2.set_xlim(100, 1500)
  else:
    ax2.set_xlim(min(x_vals), max(x_vals))
  
  ax2.set_ylim(0, max(fft_mag)*1.3)
  fig.canvas.draw_idle()


def reset(event):
  global selected_rect, selected_start, selected_end
  selected_start = None
  selected_end = None
  
  if selected_rect:
    selected_rect.remove()
    selected_rect = None
  
  ax1.clear()
  ax2.clear()
  fig.canvas.draw_idle()

def onselect(xmin, xmax):
  global selected_rect, selection_text
  # Remove previous rectangle and text if they exist
  if selected_rect:
    selected_rect.remove()
    selected_rect = None
  if selection_text:
    selection_text.remove()
    selection_text = None
  
  # Create new rectangle
  selected_rect = plt.Rectangle((xmin, ax1.get_ylim()[0]),
                                xmax - xmin,
                                ax1.get_ylim()[1] - ax1.get_ylim()[0],
                                linewidth=1.5,
                                edgecolor='red',
                                facecolor='none')
  ax1.add_patch(selected_rect)
  # Add text
  num_samples = int(abs(xmax - xmin))
  selection_text = ax1.text(xmin + (xmax-xmin)/2,
                            ax1.get_ylim()[1],
                            f"Samples: {num_samples}",
                            color='red',
                            ha='center')
  
  fig.canvas.draw_idle()

##
## ------------------------------------------------------------------------------------
##
ax1.cla()  # Clear the plot
#data_values = read_csv_values("./acquisition_227.csv")
#ax1.plot(data_values, label='Sensor Data')
ax1.set_xlabel('Time', fontsize = 20)
ax1.set_ylabel('Voltage', fontsize=20)
ax1.set_title('Real-time UART Data', fontsize=20)
ax1.legend()

fft_line, = ax2.plot([], [], label='FFT')
ax2.set_title('FFT',fontsize=20)
ax2.set_xlabel('Frequency (1/m)',fontsize=20)
ax2.set_ylabel('Amplitude',fontsize=20)
ax2.legend()

fft_button.on_clicked(run_fft)
reset_button.on_clicked(reset)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
load_button.on_clicked(load_csv)

ax1.tick_params(axis='y', labelsize=20)
ax1.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='x', labelsize=20)


span = SpanSelector(ax1, onselect, 'horizontal',
                    useblit=True,
                    props=dict(facecolor='none', edgecolor='red', linewidth=1.5),
                    interactive=True,
                    drag_from_anywhere=True)

plt.tight_layout()
plt.show()
