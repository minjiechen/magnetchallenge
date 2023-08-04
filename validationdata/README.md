## Validation Data

Data for 10 Materials [Dropbox](https://www.dropbox.com/sh/4ppuzu7z4ky3m6l/AAApqXcxr_Fnr5x9f5qDr8j8a?dl=0)

Attached are the validation datasets for the ten challenge materials, each consisting of 5,000 randomly sampled datapoints with 0 DC-bias, to evaluate model performance. The data contains $B(t)$ waveforms as well as temperature $T$ and frequency $f$ information. $H(t)$ sequences are also included for further exploration if desired. These datapoints are not intended for training but are only intended for model evaluation. The included core loss measurements should only be used to compare model predictions.

To calculate error between model predictions and the MagNet measurements, relative error should be used (with the absolute value for the difference between the predicted and measured values). The metrics of interest are average relative error, the 95th percentile, as well as the maximum error.

$Percent\ Relative\  Error = \frac{\left |meas-pred \right |}{meas}\cdot 100$, where $meas$ is MagNet's Core Loss measurement and $pred$ is the model prediction.

The PDF report for the preliminary submission should include the error distribution histograms for all ten materials utilizing this validation data, with the metrics of interests clearly visible. Attached is an example template for reporting the error distributions for the ten materials on the validation data, as well as a MATLAB script utilized to generate the graphs as an example.

(PDF template for ten materials coming soon)

<img src="../img/ExampleDistributionGraph.jpg" alt="ExampleDistributionGraph" width="540"/>
