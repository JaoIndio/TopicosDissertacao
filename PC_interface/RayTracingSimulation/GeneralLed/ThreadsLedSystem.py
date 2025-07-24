print("Script started Source")
# At the start of your script:
try:
  import cupy as cp
  from diffractio import set_backend
  set_backend('cupy')  # Switch to GPU backend
except ImportError:
  print("CuPy not available, using NumPy")
  import numpy as np

# Modify your main loop:
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

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


import matplotlib
matplotlib.use('TkAgg')  # Non-interactive backend
import matplotlib.pyplot as plt

wavelength           = 633*nm # Wavelength of light (micrometers)
simulation_width     = 35*um  # Width of simulation area (micrometers)
num_points           = 256 #128*1  # Number of points in simulation
#propagation_distance = 10*um  # Distance to screen (micrometers)

degress = np.pi/180

# Create x-axis
x = np.linspace(-simulation_width / 2, simulation_width / 2, num_points)
y = np.linspace(-simulation_width / 2, simulation_width / 2, num_points)

dx = (x[1] - x[0])# * 1e-6  # Grid spacing in meters (x in micrometers)
dy = (y[1] - y[0])# * 1e-6  # Grid spacing in meters

#Parametros Espelho Concavo
focal_length_param    = 5*um#10*um
aperture_radius_param = 12.7/2*um#12 *um

# Component positions and angles
# as Dimensoes da simulacao sao 1k vezes menores que as dimensoes reaias
z_source = 0*um
#z_bs = 140 * um  # Beamsplitter Z position /2milVezes
z_bs = (14+focal_length_param) * um  # Beamsplitter Z position
#z_m1 = 150 * um  # Mirror 1 Z position
z_m1 = (15+focal_length_param) * um  # Mirror 1 Z position
#z_m2_initial = 150 * um  # Mirror 2 Z position
z_m2_initial = (14.8+focal_length_param) * um  # Mirror 2 Z position
#z_detector = 140*um
z_detector = (14+focal_length_param)*um

slit_width_param = 1*um
slit_prop_param  = 5*um

theta_m1 = 0   # Mirror 1 tilt angle
theta_m2 = 0   # Mirror 2 tilt angle
R_mirror = 12.7/2 * um  # Mirror radius (finite size)
x_m1, y_m1 = 0 * um, 0 * um  # Mirror 1 center
x_m2, y_m2 = 0 * um, 0 * um  # Mirror 2 center

delta_z = 40 * um  # Total distance to move Mirror 2
step_z = 0.1 * um  # Step size
num_steps = int(delta_z / step_z)  # Number of steps

ItotalLED_lock = Lock()
I_total_LED          = np.zeros((len(y), len(x)))
I_total_LED_physical = np.zeros((len(y), len(x))) 


LED_diameter = 40 * um      # LED emitting area diameter
radius = LED_diameter / 2
# Discretize LED into point sources
LED_num_points = 32  # Number of point sources (adjust for accuracy vs. computation time)

P_source = 0.001  # 1 mW
area = np.pi*((LED_diameter/2)**2)  # 4 × 10⁻⁶ m²
I_source = P_source / area

#--------------------------------------------------------
#Tentativa 1
#theta = np.linspace(0, 2 * np.pi, LED_num_points, endpoint=False)
#r = np.linspace(0, radius, LED_num_points // 2) # '//' r = np.linspace(0, radius, LED_num_points // 2). It divides and truncates the result down to the nearest integer
##It creates a grid of coordinates in polar space for all possible combinations of r and theta. This is useful to describe points in a circular area (e.g., the LED emitting surface).
#R, Theta = np.meshgrid(r, theta) #takes two 1D arrays and produces coordinate matrices from them.
#
##This converts polar coordinates (R, Theta) into Cartesian coordinates:
##  converts the 2D array into a 1D array.
#X_sources = (R * np.cos(Theta)).flatten() 
#Y_sources = (R * np.sin(Theta)).flatten()

#assign each point source an equal share of the total intensity.
# Model the LED as a set of uniformly distributed point emitters.   
# Ensures that their combined intensity is normalized to 1.
# Discretize the circular LED emitting surface into a set of evenly spaced point sources.
# Convert these points from polar to Cartesian coordinates.
# Assign a uniform intensity to each point.
#intensity_per_source = 1 / len(X_sources)  # Uniform intensity

#--------------------------------------------------------
#Tentativa 2
X_sources = np.random.uniform(-LED_diameter, LED_diameter, LED_num_points)
Y_sources = np.random.uniform(-LED_diameter, LED_diameter, LED_num_points)
mask = np.sqrt(X_sources**2 + Y_sources**2) <= LED_diameter
X_sources = X_sources[mask]
Y_sources = Y_sources[mask]
LED_num_points = len(X_sources)  # Update number of valid points

# Precompute weights for Lambertian emission
weights = []
for x_s, y_s in zip(X_sources, Y_sources):
  r = np.sqrt(x_s**2 + y_s**2)
  if r == 0:
    w = 1  # On-axis intensity
  else:
    theta = np.arctan(r / z_detector)
    w = np.cos(theta)  # Lambertian profile
  weights.append(w)
weights = np.array(weights)
norm = np.sum(weights)  # Normalization factor
P_per_source = P_source * weights / norm  # Power per point source (W)


# Function to propagate through your optical system
def propagate_through_system(source_params):
  """
  Propagate a point source field through mirrors, concave mirrors, and beamsplitters.
  Customize this based on your system configuration.
  """
  x_s, y_s, P_i = source_params
  global I_total_LED
  global I_total_LED_physical

  # radial distance from the origin (LED center) to the specific point source (x_s, y_s)."
  """
  In other words, each point source is located at coordinates (x_s, y_s), 
    and r is its distance from the optical axis (usually aligned with z).

  r = distance from axis = how far off-axis the point source i
  """
  r=np.sqrt(x_s**2 + y_s**2)
  n_led = 3.4  # Typical LED semiconductor refractive index
  if r==0:
    intensity= 1
  else:
    """
    the source is off-axis, meaning the light travels at an angle theta relative to the z-axis
    Using trigonometry:
    Imagine a right triangle:
      Opposite side = r (radial distance from center)
      Adjacent side = z_detector (distance along z-axis to detector plane)
      Angle between z-axis and the line connecting source to detector = theta
  
    So:
    theta = arctangent(opposite / adjacent) = arctan(r / z_detector)

    """
    theta=np.arctan(r/z_detector)

# cosine projection law
    intensity=np.cos(theta) #*(n_led / (2 * np.pi))  # Normalized
    
  # Define point source as a narrow Gaussian beam
  """
This creates a scalar electromagnetic field (u0) defined over the spatial grid (x, y) at a specific wavelength.
Scalar_source_XY is a Diffractio class for defining monochromatic scalar fields on a 2D grid.
  """
  u0 = Scalar_source_XY(x, y, wavelength)

  """
This initializes the scalar field u0 to be a Gaussian beam.
  """
  #intensity = I_source*intensity/LED_num_points
  #print(intensity)
  w0 = 1 * um  # Gaussian waist
  # Power of Gaussian beam: P = (|A|^2 * pi * w0^2) / 2
  A = np.sqrt(2 * P_i / (np.pi * w0**2))  # Amplitude in field units
  u0.gauss_beam(A=A, w0=1 * um, r0=(x_s, y_s), z0=0, theta=0)
  #u0.gauss_beam(A=1, w0=1 * um, r0=(x_s, y_s), z0=0, theta=0)

  """
  This applies a random phase shift to every point in the field u0.u.
  enerates a random phase between 0 and 2pi for each grid point
  converts this random phase into a complex exponential:
  represents a unit vector in the complex plane at angle phi.
    
    This models a random phase front, typical in
      Simulating spatial incoherence
      Modeling phase noise
  """
  u0.u *= np.exp(1j * np.random.uniform(0, 2*np.pi, size=u0.u.shape))  # Critical: random phase


  #print("concave Mirror Param")
  focal_length = focal_length_param #10 * um  # Desired focal length
  R = 2 * focal_length   # Radius of curvature (R = 2f for mirrors)
  aperture_radius = aperture_radius_param #12 * um  # Physical size of the mirror

  concave = Scalar_mask_XY(x, y, wavelength)
  
  
  #print("concave Mirror Init")
  # Method 2: Built-in function (equivalent)
  concave.lens(r0=(0, 0), radius=(aperture_radius, aperture_radius), \
                          focal=(focal_length, focal_length), angle=0)
  
  # Add a circular aperture to limit the concave size
  #concave.circle(r0=(0, 0), radius=aperture_radius, angle=0)
  
  # Step 2: Propagate Light Source to Concave
  #print("concave Mirror Optical Input")
  # Method 2: Built-in function (equivalent)
  #concav_refl = u0.RS(z=1*focal_length)
  #u_concav = concav_refl*concave
  #u_concav_dbg = u_concav.RS(z=focal_length+1*um)

  #u_concav_dbg.draw(kind='intensity')
  #print("concave Mirror Optical Propagation/Output")
  #print("concave Mirror 1 ---> Slit ")
  # Propagate to the image plane (another 2f = 20 µm, total 40 µm from source)
  #u_at_slit = u_concav.RS(z=2*focal_length)
  #print("From Concav to BM")
  # Create a mask for the slit
  slit_mask   = Scalar_mask_XY(x, y, wavelength)
  slit_width  = slit_width_param #5*um  # 10 µm
  slit_mask.circle(r0=(0, 0), radius=slit_width / 2) #slit(x0=0, size=slit_width)
  #slit_mask.u = np.where(np.abs(slit_mask.X) < slit_width / 2, 1, 0)
  #u0.u *= slit_mask.u

  #slit_masked = u_at_slit * slit_mask
  
  #print("concave Mirror 2 Param")
  #focal_length2 = 10 * um  # Desired focal length
  #R = 2 * focal_length2   # Radius of curvature (R = 2f for mirrors)
  #aperture2_radius = 12 * um  # Physical size of the mirror
  #concave2 = Scalar_mask_XY(x, y, wavelength)
  
  #print("concave Mirror 2 Init")
  # Method 2: Built-in function (equivalent)
  #concave2.lens(r0=(0, 0), radius=(aperture2_radius, aperture2_radius), \
  #                        focal=(focal_length2, focal_length2), angle=0)
  
  # Add a circular aperture to limit the concave size
  #concave.circle(r0=(0, 0), radius=aperture_radius, angle=0)
  
  # Step 2: Propagate Light Source to Concave
  #print("concave Mirror 2 Optical Input")
  # Method 2: Built-in function (equivalent)
  #concav_refl2 = slit_masked.RS(z=focal_length2)
  u_1 = u0.RS(z=slit_prop_param) #, new_field=True)
  u_1.u *= slit_mask.u
  
  u1 = u_1.RS(z=focal_length) #, new_field=True)
  #colimatted =u1*concave
  #u2= colimatted.RS(z=focal_length) #+ slit_prop_param)# , 
                    #new_field=True)
  #u2.u *= slit_mask.u
  #u_concav2 = u2*concave
  u_concav2 = u1*concave


  # Step 2: Propagate to beamsplitter
  #print("From Concav 2 to BM")
  u_bs = u_concav2.RS(z=z_bs)
  
  # print("Scalar Field")
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

  z_m2 = z_m2_initial + i*step_z
  u_at_m2 = u_refl.RS(z=z_m2-z_bs)
  u_reflected_m2.u = -u_at_m2.u * mask_m2.u
  u_refl_return = u_reflected_m2.RS(z=z_m2-z_bs)
  u_detector_trans = r * u_trans_return  # Reflected part of transmitted beam
  u_detector_refl = t * u_refl_return    # Transmitted part of reflected beam
  u_detector.u = u_detector_trans.u + u_detector_refl.u
  
  # Propagate through the system
  #if u_detector.quality > 0.9:  # Diffractio's built-in check
  #  u_detector.filter_highpass(threshold=0.1)
    

# temporal incoherence
#https://www.rp-photonics.com/optical_intensity.html
# Isso daqui já é a PSF, porém adimencional
  with ItotalLED_lock:
    I_total_LED+=np.abs((u_detector.RS(z=z_detector).u)**2)
    #I_total_LED_physical = I_total_LED * (I_source / np.max(np.abs(u0.u)**2))
    I_total_LED_physical = I_total_LED
  u_detector.normalize()
  #print("quality = {}".format(u_detector.quality))
  #print(f"\nI_source/MAX ", I_source/np.max(np.abs(u0.u)**2))
  return np.abs(u_detector.RS(z=10*nm).u)**2


# Loop over point sources
#plt.figure(figsize=(16, 12))
plt.ion()  # Enable interactive mode for dynamic updates
fig, ax = plt.subplots(figsize=(16, 12))

im = ax.imshow(np.abs(I_total_LED_physical), 
           extent=[x.min()/um, x.max()/um, y.min()/um, y.max()/um], 
           cmap='inferno', origin='lower',vmax=1*1e-10, vmin=1*1e-12)
           #cmap='inferno', origin='lower')
#plt.colorbar(label="PSF (a.u.)")
cbar = fig.colorbar(im, ax=ax, pad=0.1)
cbar.set_label(label="Intensidade (W/m²)",fontsize=18)
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.offsetText.set_fontsize(16)

ax.set_xlabel("X (\u03bcm)",fontsize=16)
ax.set_ylabel("Y (\u03bcm)",fontsize=16)

#Axis tick labels with larger font
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

#ax.set_title(f"Potência  {P_total*1000*1000*1000:.2f} nW  |  EM  {z_m2/um:.1f} \u03bcm",fontsize=20)
for i in range(num_steps):
  
  # Avalisa-se a propagação do LED, simulando multiplos feixes interagindo em
  # cada posição Z+step 
  I_total_LED          = np.zeros((len(y), len(x)))
  I_total_LED_physical = np.zeros((len(y), len(x)))
  I_source = P_source / area  # W/m²
  source_params = [(x_s,y_s, P_i) for x_s, y_s, P_i in zip(X_sources, Y_sources, P_per_source)]

  print(f"\nPropagacao do LED multhread\n")
  with ThreadPoolExecutor(max_workers=6) as executor:
    list(executor.map(propagate_through_system, source_params))
   
  z_m2 = z_m2_initial + i*step_z
  
  count=0
  print(f"\nPropagacao do LED feita com sucesso\n")
  print(f"Plot >  ")
  #plt.clf()
  #ax.clear()
  P_total = np.sum(I_total_LED_physical) * dx * dy

  print(f"P_total >  ", P_total)
  # Plot interference pattern
  # Update the plot instead of recreating it
  im.set_data(np.abs(I_total_LED_physical))
  #im.set_extent([x.min()/um, x.max()/um, y.min()/um, y.max()/um])  # Optional, if your axes change

  ax.set_title(f"Potência  {P_total*1e9:.2f} nW  |  EM  {z_m2/um:.1f} µm", fontsize=20)

  #plt.title(f"Interference Pattern at Detector, z_m2 = {z_m2/um:.1f} um")
  
  # Update display
  plt.draw()
  plt.pause(0.1)  # Pause briefly to animate
    

print(f"\nPropagacao do LED feita com sucesso\n")
