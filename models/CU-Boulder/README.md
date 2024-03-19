This folder contains the MagNet challenge submission from the CU Boulder team. The format of the folder is such that, by executing the "CU_Boulder_Model_Inference.py" file within the "Model" subfolder, a .csv file of predictions based on the contents of the "Testing" subfolder will be produced and placed in the "Results" subfolder.

The structure of the model is such that all materials use the same model (stored as a .joblib file). To generate or replicate results for a given material simply specify the material name by changing the Material variable within "CU_Boulder_Model_Inference.py" and execute that file. All other files are auxiliary and imported with local paths. 

If any issues or questions arise, please direct them to basa4247@colorado.edu

