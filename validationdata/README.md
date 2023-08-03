## Validation Data

Data for 10 Materials (Dropbox link to come)

Attached are the validation datasets for the ten challenge materials, to evaluate model performance, each consisting of 5,000 randomly sampled datapoints with 0 DC-bias. These datapoints are not intended for training but are only intended for model evaluation. The included core loss measurements should only be used to compare model predictions.

To calculate error between model predictions and the MagNet measurements, relative error should be used (with the absolute value for the difference between the predicted and measured values). The metrics of interest are average relative error, the 95th percentile, as well as the maximum error. 

Also attached is a template for reporting the error distributions for the ten materials on this validation data, as well as a MatLab script utilized to generate the graphs as an example. 
