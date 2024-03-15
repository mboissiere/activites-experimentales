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
additional_wavelengths = np.linspace(min_wavelength, max_wavelength, num=num_wavelengths)
additional_theoretical_luminous_intensity = (2 * h * c**2) / (additional_wavelengths**5 * (np.exp((h * c) / (additional_wavelengths * kB * T)) - 1))

# Append the additional wavelengths and theoretical luminous intensity to the DataFrame
additional_df = pd.DataFrame({'wavelength (m)': additional_wavelengths, 'theoretical_luminous_intensity': additional_theoretical_luminous_intensity})
df = pd.concat([df, additional_df])


# Plot the theoretical luminous intensity
plt.figure()
plt.plot(df['wavelength (m)'], df['theoretical_luminous_intensity'], label='Theoretical')
plt.xlabel('Wavelength (m)')
plt.ylabel('Luminous Intensity (arb. unit)')
plt.title('Theoretical Luminous Intensity')
plt.legend()

# Plot the measured luminous intensity
plt.figure()
plt.plot(df['wavelength (m)'], df['luminous_intensity (arb. unit)'], label='Measured')
plt.xlabel('Wavelength (m)')
plt.ylabel('Luminous Intensity (arb. unit)')
plt.title('Measured Luminous Intensity')
plt.legend()

# Plot the difference between theoretical and measured luminous intensity
plt.figure()
plt.plot(df['wavelength (m)'], df['difference'], label='Difference')
plt.xlabel('Wavelength (m)')
plt.ylabel('Difference (arb. unit)')
plt.title('Difference between Theoretical and Measured Luminous Intensity')
plt.legend()

# Show all the plots
plt.show()

# Print the DataFrame
print(df)