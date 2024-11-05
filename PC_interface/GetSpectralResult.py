#!/usr/bin/python3

import serial
import struct
import threading

import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
    
# Configuration
START_BYTE = 0xAA
STOP_BYTE = 0x55
PORT = '/dev/ttyUSB0'  # Serial port for your USB-to-UART adapter
BAUD_RATE = 115200
FLOAT_SIZE = 4         # Size of each float in bytes
    
WAVELENGHT_SIZE  = 620
# Initialize serial port
ser = serial.Serial(PORT, BAUD_RATE, timeout=1, parity='N')

# Data storage for plotting
data_counts = []
data_values = []

def read_packet():
  while True:
    # Look for the Start Byte
    start_sequence = bytes([0xAA, 0xAA, 0xAA, 0xAA, 0xAA])
    stop_sequence = bytes([0x55, 0x55, 0x55, 0x55, 0x55])
    while True:
      start_bytes = ser.read(5)
      if start_bytes==start_sequence:
        print("\t[read Pkg] Start Byte detected")
        break
    
    packet = bytearray()
    # Read data until we reach the Stop Byte
    while True:
      byte = ser.read(1)
      if len(byte) == 0:
        continue
      
      packet.append(byte[0])
      # Check if the last 4 bytes match the stop sequence
      if len(packet) >= 5 and packet[-5:] == stop_sequence:
        print("Stop sequence detected")
        return packet[:-5]  # Return packet excluding the stop sequence

def plot_data(count, value, index):
  """Replace data in lists for plotting with new values."""
  global data_counts, data_values
  data_counts[index] = count+380  # Replace data_counts with the new count
  data_values[index] = value  # Replace data_values with the new value
    #    data_values.pop(0)

def update_plot(frame):
    """Update the plot with new data."""
    plt.cla()  # Clear the plot
    
    #print("values", data_counts)
    plt.plot(data_counts, data_values, label='Sensor Data')
    plt.xlabel('WaveLength')
    plt.ylabel('Value')
    plt.title('Real-time UART Data')
    #plt.ylim(-0.05,0.05)  # Adjust these limits based on your actual data range
    plt.legend()
    plt.tight_layout()


def decode_packet(packet):
  floats = []
  i = 0
  while i < len(packet):
    # Each packet has a count byte followed by a 4-byte float
    count_bytes = packet[i:i+2]
    count = (count_bytes[0] << 8) | count_bytes[1]
    i += 2

    # Extract 4 bytes for the float
    float_bytes = packet[i:i + FLOAT_SIZE]
    if len(float_bytes) < FLOAT_SIZE:
      print("[Decode] Incomplete float data received. Len: ", len(float_bytes))
      print("[Decode] Incomplete float data received. Pkg Len: ", len(packet))
      break

    # Convert bytes to float
    float_value = struct.unpack('<f', bytes(float_bytes))[0]  # '<f' for little-endian float
    floats.append((count, float_value))
    i += FLOAT_SIZE

  return floats

def data_thread():
  """Thread to handle UART data reading and decoding."""
  while True:
    # Read and decode the packet
    packet = read_packet()
    if packet:
      decoded_data = decode_packet(packet)
      #print("Decoded Data:", decoded_data)

      # Append data for each (count, value) pair
      index =0
      for count, value in decoded_data:
        #print("count ", count)
        value*=-1
        plot_data(count, value, index)
        index+=1

# Start the data thread
data_counts = [0]*WAVELENGHT_SIZE  # Counts from 1 to 200
data_values = [0]*WAVELENGHT_SIZE  # Counts from 1 to 200

thread = threading.Thread(target=data_thread, daemon=True)
thread.start()

# Set up the matplotlib figure and animation
fig = plt.figure()
ani = FuncAnimation(fig, update_plot, interval=125, cache_frame_data=False)  # Update plot every 500ms

# Show plot
plt.show()
