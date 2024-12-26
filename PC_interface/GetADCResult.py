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
BAUD_RATE = 921600*1    # 4.5Mbps
ADC_SIZE = 2         # Size of each float in bytes
    
WAVELENGHT_SIZE  = 1024*40
# Initialize serial port
ser = serial.Serial(PORT, BAUD_RATE, timeout=1, parity='N')

# Data storage for plotting
data_counts = []
data_values = []
max_values_over_time = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

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
      #print("\t[read Pkg] Nothing")
    
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
  
  #if index<WAVELENGHT_SIZE:
  #if value > 3.3:
  #  print("High Value: ", value)


  data_counts[index] = count  # Replace data_counts with the new count
  data_values[index] = value  # Replace data_values with the new value
    #    data_values.pop(0)

def update_plot(frame):
  """Update the plot with new data."""
  ax1.cla()  # Clear the plot
  ax2.cla()  # Clear the plot
    
  # Find the maximum value and its index
  max_index = data_values.index(max(data_values))
  max_value = data_values[max_index]
  max_count = data_counts[max_index]
  #print("\t\t[Update Plot]  Len: ", max_count)
  #print("\t\t[Update Plot]  Max Index: ", max_index)

  # Plot a red point at the maximum value
  ax1.plot(max_count, max_value, 'ro')  # 'ro' means red color, circle marker

  # Annotate the maximum value
    #arrowprops=dict(facecolor='red', shrink=0.005),\
  ax1.annotate(f'{max_count:.1f}' ,\
    xy=(max_count, max_value), \
    xytext=(max_count, max_value + 0.001), \
    fontsize=10, color='red')

  #print("values", data_counts)
  ax1.plot(data_counts, data_values, label='Sensor Data')
  ax1.set_xlabel('Time')
  ax1.set_ylabel('Voltage')
  ax1.set_title('Real-time UART Data')
  ax1.set_ylim(-0.002,4.0)  # Adjust these limits based on your actual data range
  ax1.legend()
  
  # Update max_values_over_time for tracking
  max_values_over_time.append(max_value)
         
  # Limit the list to the most recent 120 ms window
  # Assuming an update every 125 ms, keep only the last 10 values
  if len(max_values_over_time) > 10:
    max_values_over_time.pop(0)
                                    
  # Plot the maximum value trend over time
  ax2.plot(max_values_over_time, 'r-', label='Max Value over Time')
  ax2.set_xlabel('Time (approx. 120ms per point)')  
  ax2.set_ylabel('Max Value')
  ax2.set_title('Maximum Value Over Time')
  ax2.legend()
  ax2.set_ylim(min(max_values_over_time) - 0.01, max(max_values_over_time) + 0.01)

  plt.tight_layout()


def decode_packet(packet):
  floats = []
  i = 0
  count =0
  while i < len(packet):
    # Each packet has a count byte followed by a 4-byte float
    #count_bytes = packet[i:i+2]
    #count = (count_bytes[0] << 8) | count_bytes[1]
    #i += 2
    count+=1

    # Extract 4 bytes for the float
    float_bytes = packet[i:i + ADC_SIZE]
    if len(float_bytes)>1:
      adc_value =(float_bytes[0] << 8) | float_bytes[1] 
    else:
      adc_value = 4095/2
    if len(float_bytes) < ADC_SIZE:
      adc_value = 4095/2
      print("[Decode] Incomplete float data received. Len: ", len(float_bytes))
      print("[Decode] Incomplete float data received. Pkg Len: ", len(packet))
      break

    # Convert bytes to float
    #float_value = struct.unpack('<f', bytes(float_bytes))[0]  # '<f' for little-endian float
    if adc_value>4095:
      adc_value = (float_bytes[1] << 8) | float_bytes[0]
      #print("[Decode] adc Value ", adc_value)
      #print("[Decode] UART[0] ", float_bytes[0], "UART[1] ", float_bytes[1])

    float_value = adc_value*3.3/4095
    floats.append((count, float_value))
    i += ADC_SIZE

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
        #value*=-1

# Futuramente Implemetar uma janela móvel em plot_data
# o tamanho da janela eh modulavel e impacta a FFT
        plot_data(count, value, index)
        index+=1

# Start the data thread
data_counts = [0]*WAVELENGHT_SIZE  # Counts from 1 to 200
data_values = [0]*WAVELENGHT_SIZE  # Counts from 1 to 200

thread = threading.Thread(target=data_thread, daemon=True)
thread.start()

# Set up the matplotlib figure and animation
#fig = plt.figure()
ani = FuncAnimation(fig, update_plot, interval=50, cache_frame_data=False)  # Update plot every 500ms

# Show plot
plt.show()
