# Load the data from the database, 

import mysql.connector
import json

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

# Construct the SQL query for fetching data from the database, where temperature is 25.

query = "SELECT bwav, hwav, freq, temperature FROM mnt WHERE material = 'N87'"

cursor.execute(query)

result = cursor.fetchall()

# Loop through the results and load bwav and hwav into two lists
counter = 0
counter_points = 0

# record max and minimum values for hfield, bfield, dhdt and dbdt
max_temperature = -999999
min_temperature = 999999
max_hfield = -999999
min_hfield = 999999
max_bfield = -999999
min_bfield = 999999
max_dhdt = -1e12
min_dhdt = 1e12
max_dbdh = -1e12
min_dbdh = 1e12

for row in result:

    # Convert the JSON arrays to Python lists
    bwav = json.loads(row[0])
    hwav = json.loads(row[1])
    freq = row[2]
    temperature = row[3]

    # Loop each two adjacent points in the list

    for i in range(len(bwav) - 1):
        if hwav[i + 1] == hwav[i]:
            continue
        dbdh = (bwav[i + 1] - bwav[i]) / (hwav[i + 1] - hwav[i])
        dhdt = (hwav[i + 1] - hwav[i]) / (
            (1 / freq / 1024)
        )  # 1024 is the number of points in each cycle
        hfield = (hwav[i + 1] + hwav[i]) / 2
        bfield = (bwav[i + 1] + bwav[i]) / 2
        # Load the data into the diffBH_N87 table, using callproc to call "insert_into_diffBH"
        insert_data = (
            "N87",
            temperature,
            hfield,
            bfield,
            dhdt,
            dbdh,
        )
        cursor.callproc("insert_into_diffBH", insert_data)
        connection.commit()
        counter += 1
        # update max min values
        max_temperature = max(max_temperature, temperature)
        min_temperature = min(min_temperature, temperature)
        min_hfield = min(min_hfield, hfield)
        max_hfield = max(max_hfield, hfield)
        min_bfield = min(min_bfield, bfield)
        max_bfield = max(max_bfield, bfield)
        min_dhdt = min(min_dhdt, dhdt)
        max_dhdt = max(max_dhdt, dhdt)
        min_dbdh = min(min_dbdh, dbdh)
        max_dbdh = max(max_dbdh, dbdh)
        # print a progress percentage out of 1023, format to integer, wipe the row and print again
        # print(f"{  int(counter / 1023 * 100)}%  ", end="\r")
    counter = 0
    counter_points += 1
    print(f"Row {counter_points}/{len(result)} processed")
    # Update mnt_info table
    update_query = "UPDATE mnt_info SET temperature_max = %s, temperature_min = %s, hfield_max = %s, hfield_min = %s, bfield_max = %s, bfield_min = %s, dhdt_max = %s, dhdt_min = %s, dbdh_max = %s, dbdh_min = %s WHERE material = 'N87';"
    update_data = (
        max_temperature,
        min_temperature,
        max_hfield,
        min_hfield,
        max_bfield,
        min_bfield,
        max_dhdt,
        min_dhdt,
        max_dbdh,
        min_dbdh,
    )
    cursor.execute(update_query, update_data)
    connection.commit()
    # Close the connection
cursor.close()
connection.close()
