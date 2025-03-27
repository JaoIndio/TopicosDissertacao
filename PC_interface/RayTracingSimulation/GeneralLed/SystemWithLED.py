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

from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Non-interactive backend
import matplotlib.pyplot as plt

wavelength = 633*nm  # Wavelength of light (micrometers)
simulation_width = 60*um # Width of simulation area (micrometers)
num_points = 128*1  # Number of points in simulation
propagation_distance = 60*um  # Distance to screen (micrometers)

degress = np.pi/180

# Create x-axis

print("X,Y")
x = np.linspace(-simulation_width / 2, simulation_width / 2, num_points)
y = np.linspace(-simulation_width / 2, simulation_width / 2, num_points)

print("Def 2")
# Component positions and angles
# as Dimensoes da simulacao sao 1k vezes menores que as dimensoes reaias
z_source = 0*um
#z_bs = 140 * um  # Beamsplitter Z position /2milVezes
z_bs = 30 * um  # Beamsplitter Z position
#z_m1 = 150 * um  # Mirror 1 Z position
z_m1 = 40 * um  # Mirror 1 Z position
#z_m2_initial = 150 * um  # Mirror 2 Z position
z_m2_initial = 40 * um  # Mirror 2 Z position
#z_detector = 140*um
z_detector = 30*um

theta_m1 = 0   # Mirror 1 tilt angle
theta_m2 = 0   # Mirror 2 tilt angle
R_mirror = 12 * um  # Mirror radius (finite size)
x_m1, y_m1 = 0 * um, 0 * um  # Mirror 1 center
x_m2, y_m2 = 0 * um, 0 * um  # Mirror 2 center

delta_z = 40 * um  # Total distance to move Mirror 2
step_z = 0.1 * um  # Step size
num_steps = int(delta_z / step_z)  # Number of steps


diameter = 50 * um      # LED emitting area diameter
radius = diameter / 2
# Discretize LED into point sources
LED_num_points = 24  # Number of point sources (adjust for accuracy vs. computation time)
theta = np.linspace(0, 2 * np.pi, LED_num_points, endpoint=False)
r = np.linspace(0, radius, LED_num_points // 2)
R, Theta = np.meshgrid(r, theta)
X_sources = (R * np.cos(Theta)).flatten()
Y_sources = (R * np.sin(Theta)).flatten()
intensity_per_source = 1 / len(X_sources)  # Uniform intensity

# Function to propagate through your optical system
def propagate_through_system(u0, i):
  """
  Propagate a point source field through mirrors, concave mirrors, and beamsplitters.
  Customize this based on your system configuration.
  """
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
  concav_refl = u0.RS(z=focal_length)
  u_concav = concav_refl*concave
  u_concav_dbg = u_concav.RS(z=focal_length+1*um)
  #u_concav_dbg.draw(kind='intensity')
  print("concave Mirror Optical Propagation/Output")
  print("From Concav to BM")
  
  # Step 2: Propagate to beamsplitter
  u_bs = u_concav.RS(z=z_bs)
  #u_bs.draw(kind='intensity')
  u_bs = u0.RS(z=z_bs)
  
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
  u_trans_return = u_reflected_m1.RS(z=d1)
  
  mask_m2 = Scalar_mask_XY(x, y, wavelength)
  mask_m2.circle(r0=(x_m2, y_m2), radius=R_mirror, angle=0)
  u_reflected_m2 = Scalar_field_XY(x, y, wavelength)
  u_detector = Scalar_field_XY(x, y, wavelength)

  z_m2 = z_m2_initial + i*step_z
  u_at_m2 = u_refl.RS(z=z_m2-z_bs)
  u_reflected_m2.u = -u_at_m2.u * mask_m2.u
  u_refl_return = u_reflected_m2.RS(z=z_m2-z_bs)
  u_detector_trans = r * u_trans_return  # Reflected part of transmitted beam
  u_detector_refl = t * u_refl_return    # Transmitted part of reflected beam
  u_detector.u = u_detector_trans.u + u_detector_refl.u

  return u_detector, z_m2


# Total intensity at observation plane
I_total = np.zeros((len(y), len(x)))

# Loop over point sources
count = 0
print(f"Espelho 2 Pos >  ", count/len(X_sources)*100,"%", end='\r')

print(f"Simulando Propagacao da Luz do Led...  ", count/len(X_sources)*100,"%", end='\r')

plt.figure(figsize=(8, 6))
plt.ion()  # Enable interactive mode for dynamic updates
I_total_LED = np.zeros((len(y), len(x)))
for i in range(num_steps):
  for x_s, y_s in zip(X_sources, Y_sources):
    # Define point source as a narrow Gaussian beam
    print(f"\nConcluido >  ", count/len(X_sources)*100,"%", end='\n')
    count +=1
    u0 = Scalar_source_XY(x, y, wavelength)
    u0.gauss_beam(A=1, w0=1 * um, r0=(x_s, y_s), z0=0, theta=0)
    u0.u *= np.exp(1j * np.random.uniform(0, 2*np.pi))  # Critical: random phase
      
    # Propagate through the system
    u_final, z_m2 = propagate_through_system(u0, i)
      
    I_total_LED += np.abs(u_final.RS(z=30*um).u)**2  # Propagate to detector plane
  

  count=0
  print(f"\nPropagacao do LED feita com sucesso\n")
  print(f"\nPlot >  \n")
  plt.clf()
  
  # Plot interference pattern
  plt.imshow(np.abs(I_total_LED), extent=[x.min()/um, x.max()/um, y.min()/um, y.max()/um], 
             cmap='inferno', origin='lower')
  plt.colorbar(label="Intensity (a.u.)")
  plt.xlabel("X (um)")
  plt.ylabel("Y (um)")
  plt.title(f"Interference Pattern at Detector, z_m2 = {z_m2/um:.1f} um")
  
  # Update display
  plt.draw()
  plt.pause(0.1)  # Pause briefly to animate
  #plt.show()

print(f"\nPropagacao do LED feita com sucesso\n")
