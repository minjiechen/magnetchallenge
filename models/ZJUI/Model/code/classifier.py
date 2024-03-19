import numpy as np
import json
from scipy.signal import find_peaks


#%% read csv files and convert to json 
"""
Merge data into a json file, with information of 
B_waveform[T],Frequency[Hz],Temperature[C]
"""
def csv_json(data_B,data_F,data_T,filename):
    json_str_start=""
    json_str="" # Initialize the json string

    data_BFT = [data_B,data_F,data_T]
    # Loop through each file in the list of filenames
    for index in range(len(data_BFT)):
        f = data_BFT[index]
        f = f.tolist()
        data_list=list()  # Initialize the list to store data
        if (index==1 or index==2): 
            #if the database has one column, run the code below
            data=[]
            for row in f:
                data.append(row) 
        else:
            #otherwise, run the code below
            data = []
            for row in f:
                row = [float(i) for i in row]
                data_list.append(row)
            data = [row for row in data_list]
        if (index==0): # the beginning of the .json file
            json_str = '{"B_waveform[T]":' + json.dumps(data) + ","
            
        elif (index==1):
            json_str = json_str + '"Frequency[Hz]":' + json.dumps(data) + ","
            
        elif (index==2):
            json_str = json_str + '"Temperature[C]":' + json.dumps(data) + "}"
        
    with open(filename, 'w+') as jsonfile:
        
        jsonfile.write(json_str)  # Write the JSON string to the file

def openfile(file):
    with open(file,'r') as load_f:
        DATA = json.load(load_f)
    return DATA
    

def writefile(file,DATA):
    #data_path = os.path.join(road, file)
    with open(file,'w') as f:
        json.dump(DATA, f)
#%% Determine whether a waveform is sinusoidal.          
def is_sinusoidal(signal, threshold=0.96):
    fft_result = np.fft.fft(signal)
    magnitude_spectrum = np.abs(fft_result)
    peaks, _ = find_peaks(magnitude_spectrum)

    # Check if the dominant frequency is the highest peak
    if len(peaks) > 0:
        dominant_frequency = np.argmax(magnitude_spectrum[peaks])
        dominant_amplitude = magnitude_spectrum[peaks][dominant_frequency]
        total_amplitude = np.sum(magnitude_spectrum[peaks])
        amplitude_ratio = dominant_amplitude / total_amplitude

        return amplitude_ratio >= threshold
    return False
#%% Find dutycycles and deltaB for each segment of piecewise linear waveforms.
def find_corner_points(x_values, y_values):
    corner_points = []
    slopes = []
    last_slope = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
    last_y_value = y_values[0]

    # Calculate slopes and identify corner points
    for i in range(1, len(x_values) - 21):
        prev_slope = (y_values[i] - y_values[i-1]) / (x_values[i] - x_values[i-1])
        next_slope = (y_values[i+1] - y_values[i]) / (x_values[i+1] - x_values[i])
        
        slope_difference = abs(next_slope - prev_slope)
        tolerance=abs(prev_slope)*0.1
        if slope_difference > tolerance and slope_difference > 1e-6:
            if abs(last_slope - prev_slope) > 5e-5:
                corner_points.append((x_values[i], y_values[i]))
                slopes.append(prev_slope)
                last_slope = prev_slope
                last_y_value = y_values[i]
        # Merge nearby corner points
    merged_corner_points = []
    merged_slopes = []
    duty_cycles = []
    delta_Bs = []
    i = 0
    merge_threshold =50
    
    merged_corner_points.append((x_values[0], y_values[0]))
    while i < len(corner_points):
        x1, y1 = corner_points[i]
        merged_point = (x1, y1)
        j = i + 1
        while j < len(corner_points):
            x2, y2 = corner_points[j]
            distance = abs(x2 - x1)
            if distance <= merge_threshold :   
                merged_point = (x2, y2)
                j += 1
            else:
                break
        merged_corner_points.append(merged_point)
        i = j
    
    merged_corner_points.append((x_values[-1], y_values[-1]))
    
    for i in range(1, len(merged_corner_points)):     # Record slope for corner points
        duty_cycle = (merged_corner_points[i][0] - merged_corner_points[i-1][0])/1024
        delta_B = merged_corner_points[i][1] - merged_corner_points[i-1][1]
        
        slope = (merged_corner_points[i][1] - merged_corner_points[i-1][1]) / (merged_corner_points[i][0] - merged_corner_points[i-1][0])
        merged_slopes.append(slope)
        duty_cycles.append(duty_cycle)
        delta_Bs.append(delta_B)


    start_point = (x_values[0], y_values[0])
    end_point = (x_values[-1], y_values[-1])
    
    return start_point, end_point, merged_corner_points, merged_slopes, duty_cycles, delta_Bs
#%%
"""
merge data into a json file, with information including 
'B_waveform[T]','Frequency[Hz]','Temperature[C]'
and labels 'Waveform','Duty_cycle','Slopes','Delta_B'
"""
def classifier(data_B,data_F,data_T,jsonfile,labeledfile):
    
    csv_json(data_B,data_F,data_T,jsonfile)
    
    DATA = openfile(jsonfile)
    B = DATA['B_waveform[T]']
    B = np.array(B)
    # classifier
    sin_idx = []
    for i in range(B.shape[0]):
        B_new = B[i,:]
        if is_sinusoidal(B_new) != False:
            sin_idx.append(i)
    L =['piecewise' for i in range(B.shape[0])]  #classify sine and piecewise
    D =[ [0] for i in range(B.shape[0])]
    for i in sin_idx:
        L[i] = 'sine'
        B_sine = B[i,:]
        delta_B = max(B_sine)-min(B_sine)
        D[i] = [delta_B]
        
        
    #find all piecewise slopes, duty_cycle and delta_B
    Duty = [[0] for i in range(B.shape[0])]
    S = [[0] for i in range(B.shape[0])]
    for i in range(B.shape[0]):
        if L[i] == 'piecewise':
            B_lin = B[i,:]
            x_values = list(range(1024))  # List of x-values with 1024 points
            y_values = B_lin  # List of y-values with 1024 points   #36675,1000

            start_point, end_point, merged_corner_points, slopes, duty_cycles, delta_Bs = find_corner_points(x_values, y_values)
            D[i] = delta_Bs
            Duty[i] = duty_cycles
            S[i] = slopes

            piece = len(S[i])
            if piece==2:
                L[i] = 'tri'
            elif (piece==3 and S[i][0]>0 and S[i][1]<0):
                L[i] = 'tri'
            else:
                L[i] = 'trap'

    #update dict
    DATA['Waveform'] = L
    DATA['Duty_cycle'] = Duty
    DATA['Slopes'] = S
    DATA['Delta_B'] = D

    writefile(labeledfile,DATA)
    