import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model
import joblib
Material = "Material D"

Material_list = ["Material A", "Material B", "Material C", "Material D", "Material E"]

#%% Load Dataset

#Please ensure the Testing folder is together with this file. If it is an external folder
#Add two dots in the file names below like::   in_file1 = "../Testing/"+Material+"/B_Field.csv"
def load_dataset(in_file1="./Testing/"+Material+"/B_Field.csv", 
                 in_file2="./Testing/"+Material+"/Frequency.csv", 
                 in_file3="./Testing/"+Material+"/Temperature.csv"):
    # Read the first set of data
    b_data = np.array(pd.read_csv(in_file1, header=None))
    #data2 = pd.read_csv('H_waveform.csv', header=None)
    f_data = np.array(pd.read_csv(in_file2, header=None))
    t_data = np.array(pd.read_csv(in_file3, header=None))
    #data5 = pd.read_csv('Volumetric_Loss.csv', header=None)
    return b_data, f_data, t_data

#%% Calculate Core Loss
def core_loss(data_B, data_F, data_T):
        # for FFT results store it in a list named aggregateB
        fft_b = []
        # Iterate through each row 
        for i in range(len(data_B)):
            # Extract the first 1024 values from the row
            B = data_B[i]
            # FFT upto 32 terms: DC offset (B1) at the start and 31 harmonic terms after that (B2 to B32)
            fft_result = np.fft.fft(B)[:32]
            fft_output=np.abs(fft_result).tolist() 
            fft_b.append(fft_output)
        #FFT results to a NumPy array
        fft_b = np.array(fft_b)
        classifyList = []
        for B_row in data_B:
            fft_result = np.fft.fft(B_row)
            magnitude_2nd = np.abs(fft_result[1])
            magnitude_3rd = np.abs(fft_result[2])
            ratio_3rd_2nd = magnitude_2nd / magnitude_3rd
            if ratio_3rd_2nd < 7:
                
                classifyList.append(0)   # 0 FOR TRIANGLE
            elif ratio_3rd_2nd > 100:
               
                classifyList.append(1)   # 1 FOR SINE
            else:
            
                classifyList.append(2)   # 2 FOR TRAPEZOIDAL
        #The models and scalers are in separate files for cleanliness
        classifyList = np.array(classifyList)
        model = load_model('Models/TU_model_'+Material+'.h5')
        loaded_scaler = joblib.load('Scalers/TU_scaler_'+Material+'.joblib')
        scaler = MinMaxScaler(feature_range=(0, 5))

        # Extract time series data and read freq, temp and classification data
        time_series_data = fft_b
        frequency = data_F
        temperature = data_T
        classification = classifyList
        # Normalize 
        frequency = scaler.fit_transform(frequency.reshape(-1, 1))
        temperature = scaler.fit_transform(temperature.reshape(-1, 1))
        # Reshape 
        time_series_data = time_series_data.reshape((-1, 32, 1))
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 5))
        time_series_data = scaler.fit_transform(time_series_data.reshape(-1, 1))
        time_series_data = time_series_data.reshape((-1, 32, 1))
       
        #The prediction begins
        predictions = abs(model.predict([time_series_data, frequency, temperature, classification]))
        predictions_original_scale = loaded_scaler.inverse_transform(predictions).ravel()

        result_name  = "../Result/Volumetric_Loss_"+Material+".csv"

        with open(result_name, "w") as f:
            for values in zip(predictions_original_scale):
                f.write(f"{values[0]}\n")
#%% Main Function for Model Inference
def main():
    # Reproducibility
    random.seed(1)
    np.random.seed(1)
   
    data_B, data_F, data_T = load_dataset()
    print ("Data Arrangement Done!!!")
    core_loss(data_B, data_F, data_T)
    print ("Prediction Done! Check Results Folder for the csv file!!!")
if __name__ == "__main__":
    main()
    
# End