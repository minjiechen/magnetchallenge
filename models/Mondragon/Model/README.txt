The necessary parameters needed to evaluate the final 5 materials are dedinfed in the "MagNetModel.mat" and "MagNetModel2.mat" files.
→ MagNetModel.mat has the hysteresis, eddy, relaxation obtained from fitting (3) and (5), plus the sinusoidal parameters.
→ MagNetModel2.mat has the hysteresis, eddy and relaxation parameters refitted directly from (6).

The script "Model_Inference.m" is used to generate the final results as asked for the final submission.
Additionaly the script "Model_Inference_TestingP.m" is also added to check the fitting results using the fitting data (TABLE II).

To steps for the generation of the necessary parameters are 100% detailed in the final report, where the only unclarified part is the waveform classification algorithm.
This algorithm is be found in both scripts presented, in the "Classify and simplify waveforms" section.
If necessary, we can send the functions used for the automated generation of these parameters.
