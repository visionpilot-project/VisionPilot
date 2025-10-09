import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

"""
Using the latest CSV log file, deviation and steering is plotted.
"""

files = [f for f in os.listdir('.') if f.endswith('.csv')]
if files:
	latest = max(files, key=os.path.getctime)
	print(latest)
else:
	print('No CSV files found.')
latest_file = max(files, key=os.path.getctime)
df = pd.read_csv(latest_file)

print(f"Data loaded from {latest_file}")

df['time_s'] = df['frame'] / 30

plt.figure(figsize=(10,5))
plt.plot(df['time_s'], df['deviation_m'], label='Deviation (m)', color='red')
plt.plot(df['time_s'], df['steering'], label='Steering', color='blue')
plt.title('Lane Deviation and Steering over Time')
plt.xlabel('Time (s)')
plt.ylabel('Deviation / Steering')
plt.legend()
plt.grid(True)
plt.show()
