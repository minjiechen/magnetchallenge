MagNet Challenge by the University of Manchester team

We have implemented a direct data interpolation method for predicting the BH loop shape given arbitrary H-field waveform. The source data is from Princeton University's MagNet open-source database. The direct interpolation is based on the hypothesis of dB/dH's dependence on instant B, H, dH/dt, and temperature values. The BH loop is regenerated from accumulative extracted/interpolated dB/dH values from the database. The growth of the BH curve may starting from an initial state that eventually will stabilised. There are no extrapolation, no "prediction" data is used for generation.

Only N87 test results (40616 training data - id from 1 to 40616, 5000 validation data - id from 40617 to 45616) are provided due to the following two reasons:
1. The developed direct interpolation method is a very general method with only a few hundred code lines and a total file size of approximately 14 kb. Method parameters are independent of a specific material, but only related to the coverage/wideness of the measurement data. Therefore, when the input data complexity of different materials is at the same level (temperature, frequency, data volume, etc.), the data of the N87 material can be used to represent the effect of the interpolation method.
2. A key idea in this method is to build a database for a material. However, for the provided training data for each material (around 40000 lines), the time needed to build its database is around 5 days (on a laptop, but it can be way faster when a workstation or desktop is provided). 

Data Preparation: 
Extract original data to ./rawdata/{material name}/{filenames}.csv, using [load_raw_data_training.py]
Prepare database bDB using [load_diffBH.py]

BH-loop Construction/Prediction:
Use [regenerateBH.py]



