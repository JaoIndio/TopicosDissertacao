#from diffractio import nm, plt, np
#import inspect

print("Script started Source")
from diffractio.scalar_sources_XY import Scalar_source_XY

print("Mask")
from diffractio.scalar_masks_XY import Scalar_mask_XY
print("Field")
from diffractio.scalar_fields_XY import Scalar_field_XY
print("nm um nm np degrees plt")
from diffractio import mm, um, nm, np, degrees
print("Diffractio importation Done")
from scipy.ndimage import gaussian_filter


import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Non-interactive backend
import matplotlib.pyplot as plt
#import os
#os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Disable Qt GUI


print("Numpy and Matplot importation Done")

# 1. Install Diffractio (if not already installed)
# Open your terminal and run: pip install diffractio

# 2. Setup the Simulation

# Define simulation parameters

print("Def 1")
wavelength = 633*nm  # Wavelength of light (micrometers)
simulation_width = 240*um # Width of simulation area (micrometers)
num_points = 1024*1  # Number of points in simulation
propagation_distance = 240*um  # Distance to screen (micrometers)

degress = np.pi/180

# Create x-axis

print("X,Y")
x = np.linspace(-simulation_width / 2, simulation_width / 2, num_points)
y = np.linspace(-simulation_width / 2, simulation_width / 2, num_points)

print("Def 2")
# Component positions and angles
# as Dimensoes da simulacao sao 1k vezes menores que as dimensoes reaias
z_source = 0*um
z_bs = 140 * um  # Beamsplitter Z position
z_m1 = 160 * um  # Mirror 1 Z position
z_m2 = 160 * um  # Mirror 2 Z position
z_detector = 140*um

theta_m1 = 0   # Mirror 1 tilt angle
theta_m2 = 0   # Mirror 2 tilt angle
R_mirror = 12 * um  # Mirror radius (finite size)
x_m1, y_m1 = 0 * um, 0 * um  # Mirror 1 center
x_m2, y_m2 = 0 * um, 0 * um  # Mirror 2 center

# Create light source (plane wave)

print("Scalar Source")
u0 = Scalar_source_XY(x, y, wavelength)
u0.gauss_beam(A=1, w0=10*nm, r0=(0*um, 0*um), z0=0, theta=0) # nao sei qual o valor real
#u0.draw(kind='intensity')

# Introduz Incoerencia
print("Incoherent Source")
sigma_phi = np.pi/2  # Standard deviation of phase (controls incoherence level)
print("Incoherent Source 1")
phi_white = np.random.normal(0, sigma_phi, size=u0.u.shape)  # White noise
print("Incoherent Source 2")
correlation_length = 1*nm  # Correlation length in grid points (e.g., ~3.9 um)
print("Incoherent Source 3")
phi_smooth = gaussian_filter(phi_white, sigma=correlation_length)  # Smooth the phase
print("Incoherent Source 4")
phase_mask = np.exp(1j * phi_smooth)  # Convert to complex phase factor

# Step 4: Apply the phase mask to the source
print("Incoherent Source 5")
u0.u *=phase_mask  # Modify the field to introduce incoherence
#u0.draw(kind='intensity')

#test_coherence(u0, u_incoherent)
print("concave Mirror Param")
focal_length = 50 * um  # Desired focal length
R = 2 * focal_length   # Radius of curvature (R = 2f for mirrors)
aperture_radius = 12 * um  # Physical size of the mirror
concave = Scalar_mask_XY(x, y, wavelength)


print("concave Mirror Init")
# Method 2: Built-in function (equivalent)
concave.lens(r0=(0, 0), radius=(aperture_radius, aperture_radius), \
                        focal=(focal_length, focal_length), angle=0)

# Add a circular aperture to limit the concave size
#concave.circle(r0=(0, 0), radius=aperture_radius, angle=0)

# Step 2: Propagate Light Source to Concave
print("concave Mirror Optical Input")
# Method 2: Built-in function (equivalent)
#u_concav = Scalar_field_XY(x, y, wavelength)
concav_refl = u0.RS(z=focal_length)
u_concav = concav_refl*concave
print("concave Mirror Optical Propagation/Output")
print("From Concav to BM")

# Step 2: Propagate to beamsplitter
u_bs = u_concav.RS(z=z_bs)

print("Scalar Field")
#Step 3: Split at beamsplitter (50/50)
t = 1 / np.sqrt(2)  # Transmission coefficient
r = 1j / np.sqrt(2)  # Reflection coefficient (90-degree phase shift)
u_trans = Scalar_field_XY(x, y, wavelength)
u_refl = Scalar_field_XY(x, y, wavelength)
u_trans.u = u_bs.u * t
u_refl.u = u_bs.u * r

# Step 4: Propagate to Mirror 1 and apply mask
d1 = z_m1 - z_bs  # Distance from beamsplitter to Mirror 1
u_at_m1 = u_trans.RS(z=d1)

print("Mirrors Masks")
# Define Mirror 1 mask (circular mirror with possible tilt)
mask_m1 = Scalar_mask_XY(x, y, wavelength)
mask_m1.circle(r0=(x_m1, y_m1), radius=R_mirror, angle=0)
u_reflected_m1 = Scalar_field_XY(x, y, wavelength)
u_reflected_m1.u = -u_at_m1.u * mask_m1.u  # Reflection within mirror area

# Apply tilt if theta_m1 != 0

#print("Tilt Mirror1")
#if theta_m1 != 0:
#  k = 2 * np.pi / wavelength
#  kx1 = k * np.sin(theta_m1)
#  phase_tilt = np.exp(1j * 2 * kx1 * u_reflected_m1.X)  # Factor of 2 for round trip
#  u_reflected_m1.u *= phase_tilt

print("Mirror1 Prop and Refl")
# Propagate back to beamsplitter
u_trans_return = u_reflected_m1.RS(z=d1)

# Step 5: Propagate to Mirror 2 and apply mask
d2 = z_m2 - z_bs + 0*um # Distance from beamsplitter to Mirror 2
u_at_m2 = u_refl.RS(z=d2)

# Define Mirror 2 mask

mask_m2 = Scalar_mask_XY(x, y, wavelength)
mask_m2.circle(r0=(x_m2, y_m2), radius=R_mirror, angle=0)
u_reflected_m2 = Scalar_field_XY(x, y, wavelength)
u_reflected_m2.u = -u_at_m2.u * mask_m2.u

# Apply tilt if theta_m2 != 0
#print("Tilt Mirror1")
#if theta_m2 != 0:
#  k = 2 * np.pi / wavelength
#  kx2 = k * np.sin(theta_m2)
#  phase_tilt = np.exp(1j * 2 * kx2 * u_reflected_m2.X)
#  u_reflected_m2.u *= phase_tilt

# Propagate back to beamsplitter
u_refl_return = u_reflected_m2.RS(z=d2)


print("Detector")
print("Transaction")
# Step 6: Recombine at beamsplitter towards detector
u_detector_trans = r * u_trans_return  # Reflected part of transmitted beam
print("Reflection")
u_detector_refl = t * u_refl_return    # Transmitted part of reflected beam
print("Detector Field")
u_detector = Scalar_field_XY(x, y, wavelength)
print("Detector combination")
u_detector.u = u_detector_trans.u + u_detector_refl.u

# Step 7: Visualize the interference pattern

print("figure")
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(u_detector.u)**2, extent=[x.min()/um, x.max()/um, y.min()/um, y.max()/um], cmap='inferno', origin='lower')
plt.colorbar(label="Intensity (a.u.)")
plt.xlabel("X (um)")
plt.ylabel("Y (um)")

#print("Detector Draw")
#u_detector.draw(kind='intensity', has_colorbar=True)
#u_detector.draw(kind='amplitude', has_colorbar=True)

# Define X extent for visualization (arbitrary units)
#x_extent = 20
#
## Create figure and axis
#fig, ax = plt.subplots()
#
## Plot source as a point
#ax.plot(z_source, 0, 'o', label='Source', color='blue')
#
## Plot beamsplitter as a vertical line
#ax.plot([z_bs-20, z_bs], [-x_extent/8, x_extent/8], 'k-', label='Beamsplitter')
#
## Plot Mirror 1 as a red line segment
#ax.plot([z_m1, z_m1], [-x_extent/4, x_extent/4], 'r-', label='Mirror 1 (Stationary)')
#
## Plot Mirror 2 as a green line segment
#ax.plot( [40, 60], [5, 5], 'g-', label='Mirror 2 (Moving)')
#
## Plot detector as a square point
#ax.plot(z_detector, -5, 's', label='Detector', color='purple')
#
## Set labels and title
#ax.set_xlabel('Z position (um)')
#ax.set_ylabel('X position (arbitrary units)')
#ax.set_title('Schematic of Michelson Interferometer Components')
#
## Add legend
#ax.legend()



print("Final Plot")
plt.title("Interference Pattern at Detector")
plt.show()

while True:
  a=1

