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

#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))

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
        values.append(float(row[0]))  # Convert to float

  return values


ax1.cla()  # Clear the plot
data_values = read_csv_values("./acquisition_201.csv")
ax1.plot(data_values, label='Sensor Data')
ax1.set_xlabel('Time')
ax1.set_ylabel('Voltage')
ax1.set_title('Real-time UART Data')
ax1.legend()

plt.tight_layout()
plt.show()
