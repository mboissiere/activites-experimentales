import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

# Create a DataFrame for the sample wavelengths
df_sample = pd.DataFrame({
    'wavelength (m)': np.linspace(min_wavelength, max_wavelength, num=num_wavelengths)
})

# Compute theoretical luminous intensity for the sample wavelengths
df_sample['theoretical_luminous_intensity'] = (2 * h * c**2) / (df_sample['wavelength (m)']**5 * (np.exp((h * c) / (df_sample['wavelength (m)'] * kB * T)) - 1))

# Define a function that applies a scaling factor and an offset
def fit_func(wavelength, scale, offset):
    theoretical_intensity = (2 * h * c**2) / (wavelength**5 * (np.exp((h * c) / (wavelength * kB * T)) - 1))
    return scale * theoretical_intensity + offset

# Use curve_fit to find the optimal scaling factor and offset
popt, pcov = curve_fit(fit_func, df['wavelength (m)'], df['luminous_intensity (arb. unit)'])

# Extract the scale and offset from popt
scale = popt[0]
offset = popt[1]

# Print the scale and offset
print("Optimal scaling factor:", scale)
print("Optimal offset:", offset)

# Create a DataFrame for the offsetted wavelengths
df_offsetted = pd.DataFrame({
    'wavelength (m)': df['wavelength (m)'] + offset
})

# Compute theoretical luminous intensity for the offsetted wavelengths
df_offsetted['theoretical_luminous_intensity'] = (2 * h * c**2) / (df_offsetted['wavelength (m)']**5 * (np.exp((h * c) / (df_offsetted['wavelength (m)'] * kB * T)) - 1))

# Concatenate the sample, measured and offsetted wavelengths
df = pd.concat([df_sample, df, df_offsetted], ignore_index=True)

# Apply the scaling factor and offset to the measured luminous intensity
df['fitted_luminous_intensity'] = fit_func(df['wavelength (m)'], *popt)

# Create a figure with subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 12))

# Plot the theoretical luminous intensity over the concatenated wavelengths
axs[0].plot(df['wavelength (m)'], df['theoretical_luminous_intensity'], label='Theoretical')

# Plot the fitted measured luminous intensity over the concatenated wavelengths
axs[0].plot(df['wavelength (m)'], df['fitted_luminous_intensity'], label='Fitted Measured')

axs[0].set_xlabel('Wavelength (m)')
axs[0].set_ylabel('Luminous Intensity (arb. unit)')
axs[0].set_title('Theoretical and Fitted Measured Luminous Intensity')
axs[0].legend()

# Plot the difference between theoretical and fitted measured luminous intensity
axs[1].plot(df['wavelength (m)'], df['difference'], label='Difference')
axs[1].set_xlabel('Wavelength (m)')
axs[1].set_ylabel('Difference (arb. unit)')
axs[1].set_title('Difference between Theoretical and Fitted Measured Luminous Intensity')
axs[1].legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Print the DataFrame
print(df)
