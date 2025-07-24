#!/usr/bin/python3

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
ser = serial.Serial(PORT, BAUD_RATE, timeout=1, parity='N')
data_queue = queue.Queue(maxsize=43*1024)
data_deque = collections.deque(maxlen=1000000)
data_lock = threading.Lock()
serial_queue_lock = threading.Lock()
# Increase read buffer size
#ser.set_buffer_size(rx_size=65536*5, tx_size=65536*5)  # Example: 64 KB buffer

gIndex =0

# Data storage for plotting
data_counts = []
data_values = []
max_values_over_time = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

DIR = "./GreenRawResults"
filename = "."

def save_csv_func():
  # drain_queue

  while True:
    #get potins
    with data_lock:
      points = [data_points for data_points in data_deque]
    #save points
    save_data_to_csv("AS7341 a 2.7kHz Luz vermelha F7. values overtimae", 
                  "NEMA deslocando-se 5mm Luzes Off, PWM de 20Hz _ \
                  _ Pos. Ini. FM: No fim do Furo 1 _ \
                  _ Pos. Ini. MM: Precao Maxima no Fim de Curso no Fim de Curso no Fim de Curso _ ",
                  points)
    print("\t\t\t\t\t[Save csv Func] csv saved")
    time.sleep(13)


def read_queue():
  buffer = b''
  start_sequence = bytes([0xAA, 0xAA, 0xAA, 0xAA, 0xAA])
  stop_sequence = bytes([0x55, 0x55, 0x55, 0x55, 0x55])
  print("\t\t[data_thread] 1")
    
    # Look for the Start Byte
  try:
    #print("\t\t[data_thread] Bytes waiting in buffer:", ser.in_waiting)
    #with serial_queue_lock:
    #  data = data_queue.get(timeout=0.010)
    data = data_queue.get(timeout=0.010)
    print("\t\t[data_thread] 2")

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
      
      except Exception as e:
        print(f"Error in process_data: {e}")
        continue
    
    #print("\t\t[data_thread] 4")
    packet = buffer[start_bytes:stop_bytes + len(start_sequence)]
    data_point = decode_packet(packet)
    

    for count, value in data_point:
      if value < 1.1 and value > -0.01:
        with data_lock:
          #data_deque.append(data_point) 
          data_deque.append(value) 
    buffer = buffer[stop_bytes + len(stop_sequence):]
    #print("\t\t[data_thread] 5")
  
  
  except Exception as e:
    print(f"Error in process_data: try 1 {e}")

def read_uart():
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

def get_next_filename():
  """Finds the next available acquisition_X.csv filename, ensuring a new file each time."""
  
  print("\nHEY\n")
  files = glob.glob(os.path.join(DIR, "acquisition_*.csv"))  # Get all existing files
    
  if not files:
    return os.path.join(DIR, "acquisition_1.csv")  # Start with 1 if no files exist

  # Extract numbers from filenames
  numbers = [int(f.split("_")[1].split(".")[0]) for f in files]
  next_number = max(numbers) + 1

  print("\nData wil be save at {filename}\n")
  return os.path.join(DIR, f"acquisition_{next_number}.csv")

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

  #print(f"\nData saved in {filename}\n")

def read_packet():
  buffer = b''
  start_sequence = bytes([0xAA, 0xAA, 0xAA, 0xAA, 0xAA])
  stop_sequence = bytes([0x55, 0x55, 0x55, 0x55, 0x55])
  print("\t\t[data_thread] 1")
  try:
    while True:
      # Look for the Start Byte
      #time.sleep(0.040)  # Short sleep to avoid high CPU usage
      try:
        #print("\t\t[data_thread] Bytes waiting in buffer:", ser.in_waiting)
        #with serial_queue_lock:
        #  data = data_queue.get(timeout=0.010)

        #print("\t\t[data_thread] buffer len ",len(buffer))
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

            for count, value in data_point:
              if value < 1.1 and value > -0.7:
                with data_lock:
                  data_deque.append(value) 
            
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

def plot_data(count, value, index):
  """Replace data in lists for plotting with new values."""
  global data_counts, data_values
  

  data_values.append(value)
  #data_counts[index] = count  # Replace data_counts with the new count
  #data_values[index] = value  # Replace data_values with the new value
  if len(data_values) > DATA_POINTS:
    data_values.pop(0)
    #data_counts.pop(0)

def update_plot(frame):
  """Update the plot with new data."""

  #global data_counts, data_values
  #read_queue()
  #print("\t\t\t\t[Update Plot] Bytes waiting in buffer:", ser.in_waiting)
  #total_size = sum(sys.getsizeof(data_queue.queue[i]) for i in range(data_queue.qsize()))
  #print("\t\t\t\t[Update Plot] Total bytes stored in the queue:", total_size)
  
  with data_lock:
    try:
      #print(data_deque)
      #print("-----------------------------------------------------------")
      #print("-----------------------------------------------------------")
      #points = [value for data_points in data_deque for value in data_points[0][0]]
      points = [data_points for data_points in data_deque]
      #for data_points in data_deque:
      #  print("data points")
      #  print(data_points)
      #  print("-----")
      #  print(data_points[0][1])
      #  print("-----")
        #print(data_points[1])
      #  print("-----")
      #  for value in data_points[1:]:
      #    print("value")
      #    print(value)

      #all_data_arrays = [data_points[1] for data_points in data_deque]
    except Exception as e:
      print(f"[Update Plot] Error in process_data: {e}")

  
  #points = [item for sublist in all_data_arrays for item in sublist]

 
  #print(points)

  ax1.cla()  # Clear the plot
  ax2.cla()  # Clear the plot
    
  # Find the maximum value and its index
  if points:
    max_index = points.index(max(points))
    max_value = points[max_index]
    max_count = max_index
  else:
    max_index = 0
    max_value = -0.15
    max_count = 0
  #print("\t\t[Update Plot]  Len: ", max_count)
  #print("\t\t[Update Plot]  Max Index: ", max_index)

  # Plot a red point at the maximum value
  ax1.plot(max_count, max_value, 'ro')  # 'ro' means red color, circle marker

  # Annotate the maximum value
    #arrowprops=dict(facecolor='red', shrink=0.005),\
  ax1.annotate(f'{max_count:.1f}' ,
    xy=(max_count, max_value), 
    xytext=(max_count, max_value + 0.001), 
    fontsize=10, color='red')

  #print("values", data_counts)
  #ax1.plot(data_counts, points, label='Sensor Data')
  ax1.plot(points, label='Sensor Data')
  ax1.set_xlabel('Time')
  ax1.set_ylabel('Voltage')
  ax1.set_title('Real-time UART Data')
  ax1.legend()
  
  # Update max_values_over_time for tracking
  max_values_over_time.append(max_value)
         
  # Limit the list to the most recent 120 ms window
  # Assuming an update every 125 ms, keep only the last 10 values
  if len(max_values_over_time) > 10000:
    max_values_over_time.pop(0)
                                    
  # Plot the maximum value trend over time
  #ax2.plot(max_values_over_time, 'r-', label='Max Value over Time')
  ax2.set_xlabel('Time (approx. 120ms per point)')  
  ax2.set_ylabel('Max Value')
  ax2.set_title('Maximum Value Over Time')
  #ax2.legend()
  #ax2.set_ylim(min(max_values_over_time) - 0.01, max(max_values_over_time) + 0.01)
  #ax1.set_ylim(-0.002,0.003)  # Adjust these limits based on your actual data range

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
    float_bytes = packet[i+5+2 : i+5+2+(ADC_SIZE)]
    #print("[Decode] \t\tPacket: ", packet.hex())
    #print("[Decode] \t\tFloat Bytes: ", float_bytes.hex())
    if len(float_bytes)>3:
      #print("[Decode] UART[0] ", float_bytes[0], "UART[1] ", float_bytes[1])
      #print("[Decode] UART[2] ", float_bytes[2], "UART[3] ", float_bytes[3])
      #print("unpakging\n")
      adc_value = struct.unpack('<f', bytes(float_bytes))[0]
      break
    else:
      adc_value = 0
    if len(float_bytes) < ADC_SIZE:
      adc_value = 0
      print("[Decode] Incomplete float data received. Len: ", len(float_bytes))
      print("[Decode] Incomplete float data received. Pkg Len: ", len(packet))
      print("[Decode] Incomplete float data received. Pkg: ", packet.hex())
      print("[Decode] Incomplete float data received. Float Bytes: ", float_bytes.hex())
      break

    # Convert bytes to float
    if adc_value>ADC_FULLSCALE:
      adc_value = 1.8
      #print("[Decode] adc Value ", adc_value)
      #print("[Decode] UART[0] ", float_bytes[0], "UART[1] ", float_bytes[1])

    #float_value = adc_value*3.3/4095
    i += ADC_SIZE
  
  float_value = adc_value
  floats.append((count, float_value))
  #print("[Decode] adc Value ", float_value)
  return floats

#def data_thread():
  #global gIndex
#  """Thread to handle UART data reading and decoding."""

#  print("\t\t[data_thread] 1 1")
#  while True:
    # Read and decode the packet
#    print("\t\t[data_thread] 2")
#    packet = read_packet()


filename = get_next_filename()
print("\nData wil be save at",filename ,"\n")
thread          = threading.Thread(target=read_packet, daemon=True)
read_thread     = threading.Thread(target=read_uart, daemon=True)
save_csv_thread = threading.Thread(target=save_csv_func, daemon=True)

save_csv_thread.start()
read_thread.start()
thread.start()

# Set up the matplotlib figure and animation
#fig = plt.figure()
ani = FuncAnimation(fig, update_plot, interval=23, cache_frame_data=False)  # Update plot every 500ms

# Show plot
plt.show()

# Keep main thread alive
try:
  while True:
    time.sleep(1)

except KeyboardInterrupt:
    ser.close()
    print("Program terminated")


