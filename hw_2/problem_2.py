import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('font', family='Arial')
matplotlib.use('GTK')

# Load data
mjd_g, google = np.loadtxt('hw_2_data/google_data.txt', unpack=True, skiprows=1)
mjd_n, ny_temp = np.loadtxt('hw_2_data/ny_temps.txt', unpack=True, skiprows=1)
mjd_y, yahoo = np.loadtxt('hw_2_data/yahoo_data.txt', unpack=True, skiprows=1)

# Set up axes
fig, ax1 = plt.subplots(figsize=(12, 9))
ax2 = plt.twinx(ax1)
ax1.minorticks_on()
ax2.minorticks_on()

# Make title
ti = ax1.set_title('New York Temperature, Google, and Yahoo!', fontsize=30,
                   family='Times New Roman', weight='bold')
ti.set_position([0.5, 1.03])

# Label axes
ax1.set_xlabel('Date (MJD)', fontsize=18)
ax1.set_ylabel('Value (Dollars)', fontsize=18)
ax2.set_ylabel('Temperature ($^\circ$F)', fontsize=18)

# Set axis bounds
ax1.set_xlim(min(mjd_n)-200, max(mjd_g)+200)
ax1.set_ylim(-20, 770)
ax2.set_ylim(-150, 100)

# Fix ticks
ax1.tick_params(which='major', width=2, length=8, labelsize=14, pad=8, top='off')
ax1.tick_params(which='minor', length=5, top='off')
ax2.tick_params(which='major', width=2, length=8, labelsize=14, pad=8, top='off')
ax2.tick_params(which='minor', length=5, top='off')

# Make border thicker
for s in ax1.spines.values():
    s.set_linewidth(2.5)

# Plot data
lny = ax1.plot(mjd_y, yahoo, '-', color='purple', label='Yahoo! Stock Value',
               linewidth=2)
lng = ax1.plot(mjd_g, google, 'b-', label='Google Stock Value', linewidth=2)
lnn = ax2.plot(mjd_n, ny_temp, 'r--', label='NY Mon. High Temp', linewidth=2)

# Make the legend
lines = lny+lng+lnn
labels = [l.get_label() for l in lines]
leg = ax2.legend(lines, labels, loc='center left', prop={'size': 14})
leg.get_frame().set_linewidth(0)

# Write to file
fig.savefig('./problem_2.png')
