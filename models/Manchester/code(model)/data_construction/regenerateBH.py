# Regenerate BH curve given data in diffBH_N87 table

import mysql.connector
import json
import numpy as np
import matplotlib.pyplot as plt

# Connect to the MySQL database

try:
    connection = mysql.connector.connect(
        host="localhost",  # or your server's IP address
        user="mnc_user",
        password="password",
        database="mnc",
    )
except mysql.connector.Error as e:
    print(f"Error connecting to MySQL: {e}")
    exit(1)


cursor = connection.cursor()

# fetch a random row from mnt table where material is N87 and temperature is 25

query = "SELECT bwav, hwav, freq, temperature FROM mnt WHERE material = 'N87' AND temperature = 25 AND id>9000 LIMIT 1;"
cursor.execute(query)

result = cursor.fetchone()

cursor.close()

# load bwav and hwav into two lists, and then plot the bh curve

bwav = json.loads(result[0])
hwav = json.loads(result[1])
freq = result[2]
temperature = result[3]

# construct the time axis

time = np.linspace(0, 1 / freq, len(hwav))

# fetch max and min values for temperature, hfield, bfield and dhdt values from mnt_info table

query = "SELECT temperature_min, temperature_max, hfield_min, hfield_max, bfield_min, bfield_max, dhdt_min, dhdt_max FROM mnt_info WHERE material = 'N87';"
cursor = connection.cursor()

cursor.execute(query)
result = cursor.fetchone()

cursor.close()

temperature_min = result[0]
temperature_max = result[1]
hfield_min = result[2]
hfield_max = result[3]
bfield_min = result[4]
bfield_max = result[5]
dhdt_min = result[6]
dhdt_max = result[7]

hfield_tick = (hfield_max - hfield_min) / 50000
bfield_tick = (bfield_max - bfield_min) / 50000
dhdt_tick = (dhdt_max - dhdt_min) / 50000

# construct a new b-h curve using interpolated dhdb values

b_val = []
h_val = []

# b value starting from 0
b_val.append(-0.0)

steps = 2000

adaptive_ticks = 10

for i in range(steps):
    h_val.append(hwav[i % len(hwav)])

i = 1

# plot and show the old b-h curve and overlay the new b-h curve, don't pause, in the first subplot of the figure

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(hwav, bwav)
ax1.set_xlabel("H (A/m)")
ax1.set_ylabel("B (T)")
(hl,) = ax1.plot([], [])

ax1.set_title(f"N87 BH Curve at {temperature} degC and {freq} Hz")

# plot hwav
ax2.plot(time, hwav)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("H (A/m)")
ax2.set_title(f"N87 H-field waveform (INPUT) at {temperature} degC and {freq} Hz")

# set window height
plt.gcf().set_size_inches(10, 10)
plt.show(block=False)
plt.pause(0.1)

mid_b_val = b_val[0]

while i < len(h_val):
    dhdt_val = (h_val[i] - h_val[i - 1]) / ((1 / freq / 1024))
    mid_h_val = (h_val[i] + h_val[i - 1]) / 2
    # fetch values from bhttN87 table where temperature is 25, hfield is +/- hfield_tick*adaptive_ticks of h_val[i-1], bfield is +/- bfield_tick*adaptive_ticks of b_val[i-1], dhdt is +/- dhdt_tick*adaptive_ticks of dhdt_val

    if dhdt_val > 0:
        query = f"SELECT hfield, bfield, dhdt, dbdh FROM diffBH_N87 WHERE temperature = 25 AND hfield BETWEEN {mid_h_val - hfield_tick*adaptive_ticks} AND {mid_h_val + hfield_tick*adaptive_ticks} AND bfield BETWEEN {mid_b_val - bfield_tick*adaptive_ticks} AND {mid_b_val + bfield_tick*adaptive_ticks} AND dhdt > 0;"
    else:
        query = f"SELECT hfield, bfield, dhdt, dbdh FROM diffBH_N87 WHERE temperature = 25 AND hfield BETWEEN {mid_h_val - hfield_tick*adaptive_ticks} AND {mid_h_val + hfield_tick*adaptive_ticks} AND bfield BETWEEN {mid_b_val - bfield_tick*adaptive_ticks} AND {mid_b_val + bfield_tick*adaptive_ticks} AND dhdt < 0;"
    cursor = connection.cursor()

    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()

    # load all results into memory

    hfield_values = []
    bfield_values = []
    dhdt_values = []
    dbdh_values = []

    print(f"Point: {i}, Number of rows: {cursor.rowcount}, AT: {adaptive_ticks}")

    if cursor.rowcount < 50:
        adaptive_ticks += 1
        continue

    if cursor.rowcount > 200:
        adaptive_ticks -= 1

    for row in result:
        hfield_values.append(row[0])
        bfield_values.append(row[1])
        dhdt_values.append(row[2])
        dbdh_values.append(row[3])

    # find the smallest sum of squared relative error between (hfield, bfield, dhdt) point to (h_val[i-1], b_val[i-1], dhdt_val)
    min_dist = 100000
    min_index = 0

    if mid_h_val==0:
        mid_h_val=1e-16

    if mid_b_val==0:
        mid_b_val=1e-16

    if dhdt_val==0:
        dhdt_val=1e-16

    for j in range(len(hfield_values)):
        dist = (
            ((hfield_values[j] - mid_h_val) / mid_h_val) ** 2
            + ((bfield_values[j] - mid_b_val) / mid_b_val) ** 2
            + ((dhdt_values[j] - dhdt_val) / dhdt_val) ** 2
        )
        if dist < min_dist:
            min_dist = dist
            min_index = j

    dbdh_val = dbdh_values[min_index]

    # calculate the b value
    new_b_val = mid_b_val + dbdh_val * (h_val[i] - h_val[i - 1])
    b_val.append((mid_b_val + new_b_val) / 2)
    mid_b_val = new_b_val

    i += 1
    # update first i data in the plot, and rerange the axis
    hl.set_xdata(h_val[:i])
    hl.set_ydata(b_val[:i])
    ax1.relim()
    ax1.autoscale_view()

    plt.draw()
    plt.pause(0.001)


plt.show(block=True)

# close the connection

connection.close()
