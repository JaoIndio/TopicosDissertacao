print("Script started Source")
from diffractio.scalar_sources_XY import Scalar_source_XY

print("Mask")
from diffractio.scalar_masks_XY import Scalar_mask_XY
print("Field")
from diffractio.scalar_fields_XY import Scalar_field_XY
print("nm um nm np degrees plt")
from diffractio import mm, um, nm, np, degrees
from scipy.signal import fftconvolve
print("Diffractio importation Done")
from scipy.ndimage import gaussian_filter

from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Define grid
x = np.linspace(-200 * um, 200 * um, 1024)
y = np.linspace(-200 * um, 200 * um, 1024)

# 1. HeNe Laser (coherent)
wavelength_HeNe = 0.6328 * um  # 632.8 nm
beam_HeNe = Scalar_source_XY(x, y, wavelength_HeNe)
beam_HeNe.gauss_beam(A=1,r0=(0*um,0*um), w0=300*um, z0=0, theta=0 * degrees)

# 2. Red Diode Laser (coherent)
wavelength_diode = 0.650 * um  # 650 nm
beam_diode = Scalar_source_XY(x, y, wavelength_diode)
beam_diode.gauss_beam(A=1, r0=(0*um,0*um),w0=100*um, z0=0, theta=0 * degrees)

# 3. Red LED (incoherent)
diameter = 50 * um      # LED emitting area diameter
radius = diameter / 2

# Discretize LED into point sources
num_points = 12  # Number of point sources (adjust for accuracy vs. computation time)
theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
r = np.linspace(0, radius, num_points // 2)
R, Theta = np.meshgrid(r, theta)
X_sources = (R * np.cos(Theta)).flatten()
Y_sources = (R * np.sin(Theta)).flatten()
intensity_per_source = 1 / len(X_sources)  # Uniform intensity

print("Led Simulation 1")
count =0
I_total_LED = np.zeros((len(y), len(x)))
for i, (x_s, y_s) in enumerate(zip(X_sources, Y_sources)):
  # Create point source with random phase
  point = Scalar_source_XY(x, y, wavelength_HeNe)
  point.gauss_beam(A=1, w0=1*um, r0=(x_s, y_s), z0=0, theta=0)
  point.u *= np.exp(1j * np.random.uniform(0, 2*np.pi))  # Critical: random phase
    
  # Propagate and add intensity (not field)
  I_total_LED += np.abs(point.RS(z=100*um).u)**2  # Propagate to detector plane
    
  # Progress
  print(f"Progress: {100*(i+1)/len(X_sources):.1f}%", end='\r')

# Example propagation for lasers (optional)
HeNeDraw = beam_HeNe.RS(z=70*um)
HeNeDraw.draw(kind='intensity', normalize=True)

DiodeDraw = beam_diode.RS(z=70*um)
DiodeDraw.draw(kind='intensity', normalize=True)

# I_HeNe, I_diode, and I_total_LED contain the intensities at z = 1 m
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(I_total_LED), extent=[x.min()/um, x.max()/um, y.min()/um, y.max()/um], cmap='inferno', origin='lower')
plt.colorbar(label="LED 1 Intensity (a.u.)")
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")


print("Final Plot")
plt.title("Interference Pattern at Detector")
plt.show()


