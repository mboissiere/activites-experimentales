import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
h = 6.62607015e-34  # Planck constant in J*s
c = 299792458  # Speed of light in m/s
kB = 1.380649e-23  # Boltzmann constant in J/K

min_wavelength = 1e-9  # Minimum wavelength in meters
max_wavelength = 10e-6  # Maximum wavelength in meters
num_wavelengths = 1000  # Number of wavelengths for the theoretical sample

T = 1425.25  # Temperature in K
# TODO : Read a CSV that has temperature for every current

# Read the file into a DataFrame
df = pd.read_csv('data/Courant701mA.txt', sep=';', names=['wavelength (nm)', 'luminous_intensity (arb. unit)'])

# Convert wavelength to meters
df['wavelength (m)'] = df['wavelength (nm)'] * 1e-9  # Assuming wavelength is in nanometers, convert to meters

# Compute theoretical luminous intensity using Planck's law
df['theoretical_luminous_intensity'] = (2 * h * c**2) / (df['wavelength (m)']**5 * (np.exp((h * c) / (df['wavelength (m)'] * kB * T)) - 1))

# Compute the difference between theoretical and measured luminous intensity
df['difference'] = df['theoretical_luminous_intensity'] - df['luminous_intensity (arb. unit)']

# Add a sample of wavelengths
sample_wavelengths = np.linspace(min_wavelength, max_wavelength, num=num_wavelengths)
sample_theoretical_luminous_intensity = (2 * h * c**2) / (sample_wavelengths**5 * (np.exp((h * c) / (sample_wavelengths * kB * T)) - 1))

# Create a figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot the theoretical luminous intensity over the sample wavelengths
axs[0].plot(sample_wavelengths, sample_theoretical_luminous_intensity, label='Theoretical')
axs[0].set_xlabel('Wavelength (m)')
axs[0].set_ylabel('Luminous Intensity (arb. unit)')
axs[0].set_title('Theoretical Luminous Intensity')
axs[0].legend()

# Plot the measured luminous intensity
axs[1].plot(df['wavelength (m)'], df['luminous_intensity (arb. unit)'], label='Measured')
axs[1].set_xlabel('Wavelength (m)')
axs[1].set_ylabel('Luminous Intensity (arb. unit)')
axs[1].set_title('Measured Luminous Intensity')
axs[1].legend()

# Plot the difference between theoretical and measured luminous intensity
axs[2].plot(df['wavelength (m)'], df['difference'], label='Difference')
axs[2].set_xlabel('Wavelength (m)')
axs[2].set_ylabel('Difference (arb. unit)')
axs[2].set_title('Difference between Theoretical and Measured Luminous Intensity')
axs[2].legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Print the DataFrame
print(df)
