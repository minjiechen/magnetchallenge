import numpy as np
import keras
import pandas as pd
import joblib as jl
import os
from pathlib import Path

MATERIAL = "Material D"
#%% Load Model and Normalization Parameters
model_tri = keras.models.load_model(f".\\Models\\{MATERIAL} tri.keras")
norm_func_tri = pd.read_csv(f".\\Models\\{MATERIAL} info.csv")

model_sin = jl.load(f".\\Models\\{MATERIAL} SVR.joblib")
norm_func_sin = pd.read_csv(f".\\Models\\{MATERIAL} SVR info.csv")

#%% Reading Database to feed to Neural Networks
basepath = Path(__file__).parent
    # print(basepath)
foldername = f'Testing/{MATERIAL}'
filenames = os.listdir(basepath / foldername)
db = pd.DataFrame()

filepath = basepath / foldername / filenames[0]
B_waveform = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0)

filepath = basepath / foldername / filenames[1]
Frequency = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0)

filepath = basepath / foldername / filenames[2]
Temperature = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0)

aux = B_waveform.to_numpy()

B_pkpk = aux.ptp(axis=1)
B_rms = np.sqrt(np.mean(aux**2, axis=1))
B_arv = np.mean(abs(aux), axis=1)
k_f_B = B_rms/B_arv

aux_df = np.stack((B_pkpk, B_rms, B_arv, k_f_B), axis=1)
df = pd.DataFrame(aux_df, columns=['B_pkpk', 'B_rms', 'B_arv', 'k_f_B'])
df['Frequency'] = Frequency.values
# print(df['Energy'])
df['Temperature'] = Temperature.values

fdc = B_pkpk / 2 / B_rms
df['fdc'] = fdc

peak_to_peak_dB = np.ptp(np.diff(aux), axis=1)
rms_dB = np.sqrt(np.mean((np.diff(aux)) ** 2, axis=1))

fdc_dB = peak_to_peak_dB / 2 / rms_dB
df['fdc_dB'] = fdc_dB

#%% Separate only triangular shaped waves
condition_tri = (np.isclose(df['k_f_B'], 2 / (3 ** 0.5), rtol=0.05)) & (np.isclose(df['fdc'], 3 ** 0.5, rtol=0.05))
df_tri = df.loc[condition_tri]
# Calculate Duty Cycle
Duty = []
for x in B_waveform.to_numpy():
    B_pkpk = x.ptp()
    B_rms = np.sqrt(np.mean(x ** 2, axis=0))
    B_arv = np.mean(abs(x), axis=0)
    k_f_B = B_rms / B_arv
    fdc = B_pkpk / 2 / B_rms
    if (np.isclose(k_f_B, 2 / (3 ** 0.5), rtol=0.05)) & (np.isclose(fdc, 3 ** 0.5, rtol=0.05)):
        n_pos = 0
        n_neg = 0
        for j in range(1,1024):
            if x[j-1]>x[j]:
                n_pos += 1
            else:
                n_neg += 1
        duty_ratio = n_pos/(n_pos+n_neg)
        Duty.append(1-duty_ratio)
Duty = np.array(Duty)
df_tri.insert(1, column="Duty", value = Duty)

# print(df_tri.to_string())
#%% Prepare input database for Triangular
input_db = pd.DataFrame(columns = ["PkPk", "F", "T", "Duty"])

#Input is also normalized using the mean and standard deviation of the training datasets (precomputed in the files "Material X info.csv": row 0 is mean, row 1 is std_dev)
input_db['PkPk'] = ( np.log10(df_tri['B_pkpk']) - norm_func_tri['PkPk'].iloc[0] ) / norm_func_tri['PkPk'].iloc[1] 
input_db['F'] = ( np.log10(df_tri['Frequency'].to_numpy()) - norm_func_tri['Freq'].iloc[0] ) / norm_func_tri['Freq'].iloc[1] 
input_db['T'] = ( df_tri['Temperature'].to_numpy() - norm_func_tri['Temp'].iloc[0] ) / norm_func_tri['Temp'].iloc[1] 
input_db['Duty'] = ( df_tri['Duty'].to_numpy() - norm_func_tri['Duty'].iloc[0] ) / norm_func_tri['Duty'].iloc[1] 

#%% Predict Output only for Triangular shaped waves
#model_tri.summary()
output_energy = model_tri.predict(input_db, verbose = 0)     #Model outputs the Energy (Power_Loss / Frequency)

df_tri['Energy'] = output_energy
df_tri['Power_Loss'] = 10 ** df_tri['Energy'] * df_tri['Frequency'] / 1e3

#print(df_tri)

#%% Isolate Sinusoidal waves
condition_sin = ((np.isclose(df['k_f_B'], np.pi / (2 * 2 ** 0.5), rtol=0.008)) & (np.isclose(df['fdc'], 2 ** 0.5, rtol=0.008)) & (np.isclose(df['fdc_dB'], 2 ** 0.5, rtol=0.02)))
df_sin = df.loc[condition_sin]
#print(df_sin)

#%% Prepare input for sinuisoidal

#Generate Simmetric triangular waves for SVR
df_tri_sim = pd.DataFrame()
df_tri_sim['PkPk'] = ( np.log10(df_sin['B_pkpk']) - norm_func_tri['PkPk'].iloc[0] ) / norm_func_tri['PkPk'].iloc[1] 
df_tri_sim['F'] = ( np.log10(df_sin['Frequency'].to_numpy()) - norm_func_tri['Freq'].iloc[0] ) / norm_func_tri['Freq'].iloc[1] 
df_tri_sim['T'] = ( df_sin['Temperature'].to_numpy() - norm_func_tri['Temp'].iloc[0] ) / norm_func_tri['Temp'].iloc[1] 
df_tri_sim['Duty'] = ( 0.5 - norm_func_tri['Duty'].iloc[0] ) / norm_func_tri['Duty'].iloc[1] 

df_tri_sim['W'] = 10 ** np.array(model_tri.predict(df_tri_sim, verbose = 0))


#%% Run SVR


input_db = pd.DataFrame()
input_db['pk_pk'] = (df_sin['B_pkpk'].to_numpy() - norm_func_sin['PkPk'].iloc[0] ) / norm_func_sin['PkPk'].iloc[1]
input_db['f'] = ( df_sin['Frequency'].to_numpy() - norm_func_sin['Freq'].iloc[0] ) / norm_func_sin['Freq'].iloc[1] 
input_db['T'] = ( df_sin['Temperature'].to_numpy() - norm_func_sin['Temp'].iloc[0] ) / norm_func_sin['Temp'].iloc[1] 

yfit = model_sin.predict(input_db)

predictions = yfit.T*df_tri_sim.loc[:,'W']

P_pred=predictions*df_sin.loc[:, 'Frequency']/1e3

df_sin['Power_Loss']=P_pred

#print(df_sin.to_string())
#%% Isolate Trapezioidal waves

indices = set(df.index)
# Get indices of trapezioidal waves
indices_tri = df_tri.index.tolist()
indices_sin = df_sin.index.tolist()
indices_trapz = indices - set(indices_tri) - set(indices_sin)
indices_trapz = list(indices_trapz)

df_trapz = df.iloc[indices_trapz, :].copy()

#%% Split flat trapezioidal waves from the others trapezioidal waves and use composition
filepath = basepath / foldername / filenames[0]
df1 = pd.read_csv(filepath, header=None)
#print(df1)
wave_trapz_old = df1.loc[indices_trapz, :]
#print(wave_trapz_old)

aux1 = np.arange(0, 1023, 8)
aux2 = np.linspace(0, 1023, 1024)
idx_flat = []

for num in indices_trapz:

    wave_trapz = np.interp(aux1, aux2, wave_trapz_old.loc[num])
    freq = df_trapz.loc[num, 'Frequency']
    B_pkpk = df_trapz.loc[num, 'B_pkpk']
    T = 1 / freq

    dt = T / len(wave_trapz)
    der_B = np.diff(wave_trapz) / dt
    aux_input = np.sqrt(np.mean(der_B ** 2))
    f_eq_orig = 0.5 * np.abs(der_B) / B_pkpk
    aux_l = np.where(f_eq_orig < 50e3)
    f_eq = f_eq_orig.copy()
    f_l = f_eq[aux_l].copy()
    f_eq[aux_l] = 50e3
    flat_elem = list(filter(lambda x: x < 5e3, f_eq_orig))
    df_trapz.loc[num, 'flat_time'] = len(flat_elem) * dt / T
    df_trapz.loc[num, 'dB'] = aux_input
    #print(f_eq_orig)
    flat_elem = list(filter(lambda x: x < 5e3, f_eq_orig))
    df_trapz.loc[num, 'flat_time'] = len(flat_elem)*dt/T
    df_trapz.loc[num, 'dB'] = aux_input

    input= pd.DataFrame(columns = ['pk_pk', 'f', 'T', 'duty'])

    input['f'] = ( np.log10(f_eq.T) - norm_func_tri['Freq'].loc[0] ) / norm_func_tri['Freq'].loc[1]
    input['pk_pk'] = (np.log10(B_pkpk)-norm_func_tri['PkPk'].loc[0]) / norm_func_tri['PkPk'].loc[1]
    input['T'] = (df_trapz.loc[num, 'Temperature'] - norm_func_tri['Temp'].loc[0]) / norm_func_tri['Temp'].loc[1]
    input['duty'] = (0.5 -norm_func_tri['Duty'].loc[0]) / norm_func_tri['Duty'].loc[1]
        #print(input.to_string)
    tri_sim_pred = 1e-3 * 10 ** model_tri.predict(input, verbose=0)

    power_loss = freq ** 2 * np.sum(tri_sim_pred) * dt
    df_trapz.loc[num, 'Power_Loss'] = power_loss


#print(df_trapz.to_string())
#%% Assembly togheter all dataframes
final_df = pd.concat([pd.DataFrame(), df_tri])
final_df = pd.concat([final_df, df_sin])
final_df = pd.concat([final_df, df_trapz])

final_df.sort_index(inplace=True)
#print(final_df.to_string())

# %%
final_df['Power_Loss'].to_csv(f".\\Testing\\{MATERIAL}\\Volumetric_Loss_{MATERIAL}.csv", header=None, index= False)
print('Model inference finished')
