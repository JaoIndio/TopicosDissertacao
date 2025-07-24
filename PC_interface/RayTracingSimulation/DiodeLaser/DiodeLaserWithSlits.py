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

from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("Numpy and Matplot importation Done")

# 1. Install Diffractio (if not already installed)
# Open your terminal and run: pip install diffractio

# 2. Setup the Simulation

# Define simulation parameters

print("Def 1")
wavelength = 633*nm  # Wavelength of light (micrometers)
simulation_width = 20*um # Width of simulation area (micrometers)
num_points = 1024*1  # Number of points in simulation

degress = np.pi/180

# Create x-axis
print("X,Y")
x = np.linspace(-simulation_width / 2, simulation_width / 2, num_points)
y = np.linspace(-simulation_width / 2, simulation_width / 2, num_points)
dx = (x[1] - x[0]) #* 1e-6  # Grid spacing in meters (x in micrometers)
dy = (y[1] - y[0]) #* 1e-6  # Grid spacing in meters

print("Def 2")
# Component positions and angles
# as Dimensoes da simulacao sao 1k vezes menores que as dimensoes reaias
z_source = 0*um
focal_length = 5 * um  # Desired focal length
aperture_radius_param = 15*um

slit_width_param = 2.5*um
slit_prop_param  = 5*um


#z_bs = 140 * um  # Beamsplitter Z position
z_bs = (14+focal_length) * um  # Beamsplitter Z position
#z_m1 = 150 * um  # Mirror 1 Z position
z_m1 = (15+focal_length) * um  # Mirror 1 Z position
#z_m2_initial = 150 * um  # Mirror 2 Z position
z_m2_initial = (14.8+focal_length) * um  # Mirror 2 Z position
#z_detector = 140*um
z_detector = (14+focal_length)*um

theta_m1 = 0   # Mirror 1 tilt angle
theta_m2 = 0   # Mirror 2 tilt angle
R_mirror = 12.7/2 * um  # Mirror radius (finite size)
x_m1, y_m1 = 0 * um, 0 * um  # Mirror 1 center
x_m2, y_m2 = 0 * um, 0 * um  # Mirror 2 center

delta_z = 40 * um  # Total distance to move Mirror 2
step_z = 0.05 * um  # Step size
num_steps = int(delta_z / step_z)  # Number of steps
# Create light source (plane wave)

print("Scalar Source")
P_source = 0.001  # 1 mW
area = np.pi*((slit_width_param/2)**2)  # 4 × 10⁻⁶ m²
I_source = P_source / area

u0 = Scalar_source_XY(x, y, wavelength)
u0.gauss_beam(A=1, w0=6*um, r0=(0*um, 0*um), z0=0, theta=0) # nao sei qual o valor real

# Step 4: Apply the phase mask to the source
print("Incoherent Source 5")
u0.u *=np.exp(1j * np.random.uniform(0, np.pi/4, size=u0.u.shape)) #  # Modify the field to introduce incoherence

print("concave Mirror 1 Param")
R = 2 * focal_length   # Radius of curvature (R = 2f for mirrors)
aperture_radius = aperture_radius_param  # Physical size of the mirror
concave = Scalar_mask_XY(x, y, wavelength)

print("concave Mirror 1 Init")
# Method 2: Built-in function (equivalent)
#concave.lens(r0=(0, 0), radius=(aperture_radius, aperture_radius), \
#                       focal=(focal_length, focal_length), angle=0)
concave.lens(r0=(0, 0), radius=(aperture_radius),
                       focal=(focal_length), angle=0)

# Add a circular aperture to limit the concave size
#concave.circle(r0=(0, 0), radius=aperture_radius, angle=0)

# Step 2: Propagate Light Source to Concave
print("concave Mirror 1 Optical Input")
# Method 2: Built-in function (equivalent)
#concav_refl = u0.RS(z=2*focal_length)
#u_concav = concav_refl*concave
#u_concav_dbg = u_concav.RS(z=2*focal_length+1*um)
#u_concav_dbg.draw(kind='intensity')
print("concave Mirror 1 Optical Propagation/Output")
print("concave Mirror 1 ---> Slit ")
# Propagate to the image plane (another 2f = 20 µm, total 40 µm from source)
#u_at_slit = u_concav.RS(z=2*focal_length)
#u_at_slit = u0.RS(z=1*focal_length)

# Create a mask for the slit
slit_mask   = Scalar_mask_XY(x, y, wavelength)
slit_width  = slit_width_param  # 10 µm
slit_mask.circle(r0=(0, 0), radius=slit_width / 2)
#slit_mask.u = np.where(np.abs(slit_mask.X) < slit_width / 2, 1, 0)

# Apply the mask to the source
u_at_slit = u0.RS(z=slit_prop_param)
u_at_slit.u *= slit_mask.u
# Propagate the light using FFT

print("concave Mirror 2 Param")
#focal_length2 = 50 * um  # Desired focal length
#R = 2 * focal_length2   # Radius of curvature (R = 2f for mirrors)
#aperture2_radius = 1*12 * um  # Physical size of the mirror
#concave2 = Scalar_mask_XY(x, y, wavelength)

print("concave Mirror 2 Init")
# Method 2: Built-in function (equivalent)
#concave2.lens(r0=(0, 0), radius=(aperture2_radius, aperture2_radius), \
#                        focal=(focal_length2, focal_length2), angle=0)

# Add a circular aperture to limit the concave size
#concave.circle(r0=(0, 0), radius=aperture_radius, angle=0)

# Step 2: Propagate Light Source to Concave
print("concave Mirror 2 Optical Input")
# Method 2: Built-in function (equivalent)
#concav_refl2 = slit_masked.RS(z=focal_length2)
#u_concav2 = concav_refl2*concave2
#u_concav_dbg2 = u_concav2.RS(z=focal_length2+1*um)
#u_concav_dbg2.draw(kind='intensity')
u2 = u_at_slit.RS(z=focal_length) #, new_field=True) #(Laser+Propagacao)
u_concav2 = u2 *concave

print("From Concav to BM")

# Step 2: Propagate to beamsplitter
u_bs = u_concav2.RS(z=z_bs)
#u_bs.draw(kind='intensity')

#print("Scalar Field")
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

#print("Mirrors Masks")
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

# Initialize figure
# **Set Up Plotting**
#plt.figure(figsize=(8, 6))
plt.ion()  # Enable interactive mode for dynamic updates

fig, ax = plt.subplots(figsize=(16, 12))
I_total_LED_physic = np.zeros((len(y), len(x))) 
im = ax.imshow(I_total_LED_physic, 
           extent=[x.min()/um, x.max()/um, y.min()/um, y.max()/um], 
           cmap='inferno', origin='lower', vmax=4*1e-6, vmin=0.5*1e-7)

#plt.colorbar(label="PSF (a.u.)")
cbar = fig.colorbar(im, ax=ax, pad=0.1)
cbar.set_label(label="Intensidade (W/m²)",fontsize=20)
cbar.ax.tick_params(labelsize=20)
cbar.ax.yaxis.offsetText.set_fontsize(20)

ax.set_xlabel("X (\u03bcm)",fontsize=20)
ax.set_ylabel("Y (\u03bcm)",fontsize=20)

#Axis tick labels with larger font
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

for i in range(num_steps):
  z_m2 = z_m2_initial + i*step_z
  print("percentage:", (i/num_steps)*100)
  # Apply tilt if theta_m1 != 0
  
  #print("Mirror1 Prop and Refl")
  # Propagate back to beamsplitter
  
  # Step 5: Propagate to Mirror 2 and apply mask
  u_at_m2 = u_refl.RS(z=z_m2-z_bs)
  
  # Define Mirror 2 mask
  u_reflected_m2.u = -u_at_m2.u * mask_m2.u
  
  # Propagate back to beamsplitter
  u_refl_return = u_reflected_m2.RS(z=z_m2-z_bs)
  
  #print("Detector")
  #print("Transaction")
  # Step 6: Recombine at beamsplitter towards detector
  u_detector_trans = r * u_trans_return  # Reflected part of transmitted beam
  #print("Reflection")
  u_detector_refl = t * u_refl_return    # Transmitted part of reflected beam
  #print("Detector Field")
  #print("Detector combination")
  u_detector.u = u_detector_trans.u + u_detector_refl.u
  
  # Clear previous plot
  #plt.clf()
  
  # Plot interference pattern
  #plt.imshow(np.abs(u_detector.u)**2, extent=[x.min()/um, x.max()/um, y.min()/um, y.max()/um], 

  print(f"\nI_source: ", I_source)
  print(f"\nMAX do abs u0: ", np.max(np.abs(u0.u)**2))
  print(f"\nI_source/MAX ", I_source/np.max(np.abs(u0.u)**2))
  I_total_LED=np.abs(u_detector.RS(z=z_detector).u)**2
  I_total_LED_physic = I_total_LED*I_source/np.max(np.abs(u0.u)**2)
  P_total = np.sum(I_total_LED_physic) * dx * dy
  #plt.imshow(np.abs(u_detector.RS(z=z_detector).u)**2, 
  #plt.imshow(I_total_LED_physic, 
  #           extent=[x.min()/um, x.max()/um, y.min()/um, y.max()/um], 
  #           cmap='inferno', origin='lower', vmax=4*1e-6, vmin=0.5*1e-7) 
  #, vmax=0.2125, vmin=0.0005)
  #plt.colorbar(label="Intensity (a.u.) ")
  #plt.colorbar(label="PSF (W/m²) ")
  #plt.xlabel("X (um)")
  #plt.ylabel("Y (um)")
  im.set_data(np.abs(I_total_LED_physic))
  ax.set_title(f"Potência Total {P_total*1000:.2f} mW  |  EM  {z_m2/um:.2f} µm", fontsize=20)
  
  # Update display
  plt.draw()
  plt.pause(0.1)  # Pause briefly to animate


print("Final Plot")
plt.ioff()
plt.show()

def pause_animation(event):
  ani.event_source.stop()

def resume_animation(event):
  ani.event_source.start()

  ax_pause = plt.axes([0.7, 0.9, 0.1, 0.05])
  ax_resume = plt.axes([0.81, 0.9, 0.1, 0.05])
  btn_pause = Button(ax_pause, 'Pause')
  btn_resume = Button(ax_resume, 'Resume')
  btn_pause.on_clicked(pause_animation)
  btn_resume.on_clicked(resume_animation)

while True:
  a=1

