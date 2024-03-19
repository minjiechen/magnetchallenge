# Load data from rawdata/N87/ into the table mnt

import csv
import json
import mysql.connector

material_str = "N87"

counter = 0

# Establish a database connection
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

# Open your CSV files for reading
with open("rawdata/" + material_str + "/B_waveform[T].csv", newline="") as b_file, open(
    "rawdata/" + material_str + "/H_waveform[Am-1].csv", newline=""
) as h_file, open(
    "rawdata/" + material_str + "/Temperature[C].csv", newline=""
) as t_file, open(
    "rawdata/" + material_str + "/Frequency[Hz].csv", newline=""
) as f_file, open(
    "rawdata/" + material_str + "/Volumetric_losses[Wm-3].csv", newline=""
) as pv_file:
    # Create CSV readers
    b_reader = csv.reader(b_file)
    h_reader = csv.reader(h_file)
    t_reader = csv.reader(t_file)
    f_reader = csv.reader(f_file)
    pv_reader = csv.reader(pv_file)

    # Process each set of rows from the files
    for b_row, h_row, t_row, f_row, pv_row in zip(
        b_reader, h_reader, t_reader, f_reader, pv_reader
    ):
        counter = counter + 1
        #############################################################################################################
        # If you previous loading is broken, check the number of entries and start from the broken line number below.
        #############################################################################################################
        if counter > 0:
            # Convert the bwav and hwav fields to JSON arrays
            bwav_json = json.dumps([float(num) for num in b_row])
            hwav_json = json.dumps([float(num) for num in h_row])

            # Get the other values as floats
            temperature = float(
                t_row[0]
            )  # Adjust if your value is not in the first column
            freq = float(f_row[0])  # Adjust if your value is not in the first column
            pv = float(pv_row[0])  # Adjust if your value is not in the first column

            # Construct the SQL query for inserting data into the database, call the stored procedure "insert_into_mnt"

            values = (material_str, freq, bwav_json, hwav_json, temperature, pv, 0, "train")
            #query = "INSERT INTO mnt (bwav, hwav, temperature, freq, pv, material, tag) VALUES (%s, %s, %s, %s, %s, %s, %s);"
            #values = (bwav_json, hwav_json, temperature, freq, pv, material_str, "train")

            try:
                # Execute the query
                cursor.callproc("insert_into_mnt", values)
                print(f"{counter}")
            except mysql.connector.Error as e:
                print(f"Error inserting into MySQL: {e}")
                continue  # Skip this row and continue with the next row

            # Make sure data is committed to the database
            connection.commit()

# Close the connection
cursor.close()
connection.close()
