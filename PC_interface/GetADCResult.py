#!/usr/bin/python3

# python3 ./GetADCResult.py <- To run
import serial
import struct
import threading
import queue
import time
import sys

import csv
import os
import glob

import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
 
from queue import Empty

# Configuration
START_BYTE = 0xAA
STOP_BYTE = 0x55
PORT = '/dev/ttyUSB0'  # Serial port for your USB-to-UART adapter
BAUD_RATE = 921600*1    # 4.5Mbps
ADC_SIZE = 2         # Size of each float in bytes
ADC_FULLSCALE = 4095         # Size of each float in bytes
    
WAVELENGHT_SIZE  = 1024*40*1024
# Initialize serial port
ser = serial.Serial(PORT, BAUD_RATE, timeout=0.5, parity='N')

# Data storage for plotting
data_counts = []
global_DataCounts=0
data_values = []

max_values_over_time = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

data_queue = queue.Queue(maxsize=16*1024*1024)
ADC_DataQueue = queue.Queue(maxsize=16*1024*1024)

DIR = "./ADC_DMA_RawResults"
filename = "."


def get_next_filename():
  """Finds the next available ADC_Burst_X.csv filename, ensuring a new file each time."""
  
  print("\nHEY\n")
  files = glob.glob(os.path.join(DIR, "ADC_Burst_Aq_*.csv"))  # Get all existing files
    
  if not files:
    return os.path.join(DIR, "ADC_Burst_Aq_1.csv")  # Start with 1 if no files exist

  # Extract numbers from filenames
  numbers = [int(f.split("_")[5].split(".")[0]) for f in files]
  next_number = max(numbers) + 1

  print("\nData wil be save at {filename}\n")
  return os.path.join(DIR, f"ADC_Burst_Aq_{next_number}.csv")


def save_data_to_csv(name, description, values):
  """Saves the given data to a new CSV file each time, ensuring overwriting does not happen."""
  #filename = get_next_filename()
  global filename

  with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
        
    # Write metadata
    writer.writerow([name])  # First row: Name
    writer.writerow([description])  # Second row: Description
    
    # Write values
    for value in values:
      writer.writerow([value])


def save_csv_func():
  # drain_queue
  #global data_counts, global_DataCounts, data_values
  while True:
    while True:
      items = [] 
      try:
        items.append(ADC_DataQueue.get_nowait())  # non-blocking
        #global_DataCounts+=1
        #data_values[global_DataCounts] = items[global_DataCounts]
        #data_counts[global_DataCounts] = global_DataCounts
      except Empty:
        break
                        
    save_data_to_csv("ADC DMA Burst. values overtime", 
                  "NEMA deslocando-se Xmm Luzes Off, PWM de 20Hz _ \
                  _ Pos. Ini. FM: No fim do Furo 1 _ \
                  _ Pos. Ini. MM: Precao Maxima no Fim de Curso no Fim de Curso no Fim de Curso _ ",
                  items)

    print("\t\t\t\t\t[Save csv Func] csv saved")
    time.sleep(13)
             
             
def read_packet():
  buffer = b''
  start_sequence = bytes([0xAA, 0xAA, 0xAA, 0xAA, 0xAA])
  stop_sequence = bytes([0x55, 0x55, 0x55, 0x55, 0x55])
  print("\t\t[data_thread] 1")
  global data_counts, global_DataCounts, data_values
  try:
    while True:
      # Look for the Start Byte
      try:
        data = data_queue.get(timeout=0.02)
        buffer += data
        while True:
          try:
            #print("\t\t[data_thread] 3")
            start_bytes = buffer.find(start_sequence)
            if start_bytes == -1:
              buffer = buffer[max(0, len(buffer) - len(start_sequence) + 1):]
              break

            stop_bytes = buffer.find(stop_sequence, start_bytes + len(stop_sequence))
            if stop_bytes == -1:
              break  # Wait for more data
        
            packet = buffer[start_bytes:stop_bytes + len(start_sequence)]
            data_point = decode_packet(packet)
            #print(packet)

            for count, value in data_point:

              data_values[global_DataCounts] = value
              data_counts[global_DataCounts] = global_DataCounts
              global_DataCounts+=1
              
              #if value < 1.1 and value > -0.7:
              ADC_DataQueue.put(value)
            
            buffer = buffer[stop_bytes + len(stop_sequence):]
          
          except Exception as e:
            print(f"Error in process_data: {e}")
            continue
        
      except queue.Empty:
        #print("\t\t[data_thread] | Exception| Bytes waiting in buffer:", ser.in_waiting)
        continue
      except Exception as e:
        print(f"Error in process_data: {e}")
        continue
  except Exception as e:
    print(f"Error in process_data: {e}")

'''
      #start_bytes = ser.read(5)
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
'''
def plot_data(count, value, index):
  """Replace data in lists for plotting with new values."""
  global data_counts, data_values
  
  #if index<WAVELENGHT_SIZE:
  #if value > 3.3:
  #  print("High Value: ", value)


  data_counts[index] = count  # Replace data_counts with the new count
  data_values[index] = value  # Replace data_values with the new value
  #ADC_DataQueue.put(value)
    #    data_values.pop(0)

def update_plot(frame):
  global data_counts, global_DataCounts, data_values

  #print(f"")
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

    # Extract 4 bytes for the float
    float_bytes = packet[i:i + ADC_SIZE]
    if len(float_bytes)>1:
      #print("[Decode] UART[0] ", float_bytes[0], "UART[1] ", float_bytes[1])
      adc_value =(float_bytes[1] << 8) | float_bytes[0] 
    else:
      adc_value = 0 #ADC_FULLSCALE/2
    if len(float_bytes) < ADC_SIZE:
      adc_value = 0 #ADC_FULLSCALE/2
      print("\t\t\t[Decode] Incomplete float data received. Len: ", len(float_bytes))
      print("\t\t\t[Decode] Incomplete float data received. Pkg Len: ", len(packet))
      break

    # Convert bytes to float
    #float_value = struct.unpack('<f', bytes(float_bytes))[0]  # '<f' for little-endian float
    if adc_value>ADC_FULLSCALE:
      adc_value = 0 #(float_bytes[1] << 8) | float_bytes[0]
      #print("[Decode] adc Value ", adc_value)
      #print("[Decode] UART[0] ", float_bytes[0], "UART[1] ", float_bytes[1])

    float_value = adc_value*3.3/4095
    #float_value = adc_value
    if(float_bytes[1] < 16): # Do contrário tem algo de errado
      #print("[Decode] UART[0] ", float_bytes[0], "UART[1] ", float_bytes[1])
      #print("[Decode] adc Value ", adc_value)
      count+=1
      floats.append((count, float_value))

    i += ADC_SIZE
  
  #print("[Decode] adc Value ", adc_value*3.3/4095)
  #print("[Decode] adc Raw ", adc_value)
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

def read_uart():

  print(f"Read_uart thread")
  while True:
    try:
      #with serial_queue_lock:
      if ser.in_waiting > 0:  # Check if data is available
        data = ser.read(ser.in_waiting)  # Read all available bytes
        data_queue.put(data)

      #print("[read_uart] Bytes waiting in buffer:", ser.in_waiting)
      time.sleep(0.010)  # Short sleep to avoid high CPU usage
    except Exception as e:
      print(f"Error in process_data: {e}")
      continue 



filename = get_next_filename()
print("\nData wil be save at",filename ,"\n")

# Start the data thread
data_counts = [0]*WAVELENGHT_SIZE  # Counts from 1 to 200
data_values = [0]*WAVELENGHT_SIZE  # Counts from 1 to 200


read_thread     = threading.Thread(target=read_uart, daemon=True)
read_thread.start()

thread = threading.Thread(target=read_packet, daemon=True)
thread.start()

save_csv_thread = threading.Thread(target=save_csv_func, daemon=True)
save_csv_thread.start()

# Set up the matplotlib figure and animation
#fig = plt.figure()
ani = FuncAnimation(fig, update_plot, interval=50, cache_frame_data=False)  # Update plot every 500ms

# Show plot
plt.show()
