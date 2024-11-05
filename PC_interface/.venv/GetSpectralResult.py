    
import serial
import struct
import threading

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
    
# Configuration
START_BYTE = 0xAA
STOP_BYTE = 0x55
PORT = '/dev/ttyUSB0'  # Serial port for your USB-to-UART adapter
BAUD_RATE = 115200
FLOAT_SIZE = 4         # Size of each float in bytes
    
# Initialize serial port
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

def read_packet():
  while True:
   # Look for the Start Byte
   byte = ser.read(1)
   if len(byte) == 0:
    continue  # Timeout or no data
   if byte[0] == START_BYTE:
    print("Start Byte detected")
    packet = []

    # Read data until we reach the Stop Byte
    while True:
      byte = ser.read(1)
      if len(byte) == 0:
        print("Timeout waiting for Stop Byte")
        return None

      # Check if this is the Stop Byte
      if byte[0] == STOP_BYTE:
        print("Stop Byte detected")
        return packet  # End of packet

      # Append each byte to packet
      packet.append(byte[0])

def plot_data(count, value):
    """Append data to lists for plotting."""
    data_counts.append(count)
    data_values.append(value)
    if len(data_counts) > 50:  # Limit the plot to the last 50 points
        data_counts.pop(0)
        data_values.pop(0)

def update_plot(frame):
    """Update the plot with new data."""
    plt.cla()  # Clear the plot
    plt.plot(data_counts, data_values, label='Sensor Data')
    plt.xlabel('Count')
    plt.ylabel('Value')
    plt.title('Real-time UART Data')
    plt.legend()
    plt.tight_layout()


def decode_packet(packet):
  floats = []
  i = 0
  while i < len(packet):
    # Each packet has a count byte followed by a 4-byte float
    count = packet[i]
    i += 1

    # Extract 4 bytes for the float
    float_bytes = packet[i:i + FLOAT_SIZE]
    if len(float_bytes) < FLOAT_SIZE:
      print("Incomplete float data received")
      break

    # Convert bytes to float
    float_value = struct.unpack('<f', bytes(float_bytes))[0]  # '<f' for little-endian float
    floats.append((count, float_value))
    i += FLOAT_SIZE

  return floats

def data_thread():
  """Thread to handle UART data reading and decoding."""
  while True:
    byte = ser.read(1)
    if len(byte) > 0 and byte[0] == START_BYTE:
      print("Start Byte detected")

      # Read and decode the packet
      packet = read_packet()
      decoded_data = decode_packet(packet)
      print("Decoded Data:", decoded_data)

      # Append data for each (count, value) pair
      for count, value in decoded_data:
        plot_data(count, value)

# Start the data thread
thread = threading.Thread(target=data_thread, daemon=True)
thread.start()

# Set up the matplotlib figure and animation
fig = plt.figure()
ani = FuncAnimation(fig, update_plot, interval=125)  # Update plot every 500ms

# Show plot
plt.show()
