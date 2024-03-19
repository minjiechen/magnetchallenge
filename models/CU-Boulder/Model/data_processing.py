import numpy as np
import os
import iGSE
import pickle
import matplotlib.pyplot as plt
import math
import matplotlib.colors as mcolors
import random
from scipy.optimize import minimize
import csv

class coreMaterial:
    # k is in MKS units (must be divided by 1000 before being used in iGSE)
    def __init__(self, name, alpha, beta, k, i, maxFreq, permeability):
        # Material name
        self.name = name
        # S-params
        self.alpha = alpha
        self.beta = beta
        self.k = k
        # index in one-hot encoding array
        self.index = i
        # additional possible material features
        self.maxFreq = maxFreq
        # Core material permeability
        self.permeability = permeability

# List for the MagNet materials - S-params have been replaced with optimal [for sinusoids]
materials = [
    coreMaterial("N87", 1.6510549518850788, 2.5684286935373444, 0.22055180357325818, 0, 500e3, 2.2e3),
    coreMaterial("N49", 1.60000000023557, 2.69999999985035, 0.407238401988939, 1, 1000e3, 1.5e3),
    coreMaterial("N30", 1.8626024147577311, 2.3206505881723425, 0.013038781255903584, 2, 400e3, 4.3e3),
    coreMaterial("N27", 1.6576804020477185, 2.5407547999961384, 0.1780713195465577, 3, 150e3, 2e3),
    coreMaterial("78", 1.8100082442688465, 2.5639321086315663, 0.02587545313201249, 4, 500e3, 2.3e3),
    coreMaterial("77", 1.7506925886801696, 2.522174370283959, 0.04867133538302141, 5, 100e3, 2e3),
    coreMaterial("3F4", 1.60000000012103, 2.69999999996669, 0.881119858361143, 6, 2000e3, .9e3),
    coreMaterial("3E6", 1.88110659998852, 2.29999999998581, 0.0103084497401855, 7, 100e3, 12e3),
    coreMaterial("3C94", 1.7467863426492043, 2.494227634660561, 0.03668179851965532, 8, 300e3, 2.3e3),
    coreMaterial("3C90", 1.6677460483494921, 2.6318944941115, 0.14107483868460985, 9, 200e3, 2.3e3),
    # Steinmetz parameters optimized and constrained on mystery materials
    coreMaterial("A", 1.9766324314273338, 2.50971829136747, 0.006710336398570659, 10, 0, 0),
    coreMaterial("B", 1.9999999904654548, 2.3000000126033426, 0.002200573427861818, 11, 0, 0),
    coreMaterial("C", 2.0000000000000497, 2.5616543843453896, 0.0018325958912108623, 12, 0, 0),
    coreMaterial("D", 1.5999999999933523, 2.661628051178331, 0.2883281947124199, 13, 0, 0),
    coreMaterial("E", 1.5999999999991483, 2.700000000000582, 0.3544047687117679, 14, 0, 0),
]

# # List for the MagNet materials - uses hand-calculated S-params from datasheet
# materials = [
#     coreMaterial("N87", 1.3010299957, 2.3461982066, 8.1832052208),
#     coreMaterial("N49", 1.1386468839, 2.6709952092, 100.7091232271),
#     coreMaterial("N27", 1.1760912591, 2.0000000000, 33.619496291),
#     coreMaterial("78", 1.0849625007, 2.3584991697, 99.0485629647),
#     coreMaterial("77", 1.0511525224, 2.0405243075, 116.6146825295),
#     coreMaterial("3F4", 1.0000000000, 3.3755463477, 4748.7175992285),
#     coreMaterial("3C94", 1.3684827971, 2.736955942, 4.7065519763),
#     coreMaterial("3C90", 1.5145731728, 2.736955942, 1.2506993968)
# ]

class coreSample:
    def __init__(self, material: coreMaterial, B_waveform: np.ndarray, Frequency: float, H_waveform: np.ndarray,
                 Temperature: float,
                 Volumetric_losses: float):
        # Sample characteristics that come from data
        self.material = material
        self.B_waveform = B_waveform
        # Bpk is half of pk to pk of B (emulates peak of a sinusoid)
        self.Bpk = (max(self.B_waveform) - min(self.B_waveform)) / 2
        self.Frequency = Frequency
        self.H_waveform = H_waveform
        self.Temperature = Temperature
        self.Volumetric_losses = Volumetric_losses

        # Get waveshape from classifier function
        # one-hot format [sinusoid, triangle, trapezoid]
        self.waveform_oneHot, self.waveform_type = classify_waveform(self.B_waveform, plot=False, percent_to_plot=.005)
        # One-hot encode material
        self.material_oneHot = [0] * len(materials)
        self.material_oneHot[material.index] = 1
        self.mysteryBoolean = 0

    def iGSE(self) -> float:
        # define time value array based on freq (start, stop, length)
        tvs = np.linspace(0, (1 / self.Frequency), num=len(self.B_waveform))

        # get iGSE and GSE losses
        # k that iGSE takes is for mW/cm^3 so must divide by 1000
        iGSE_loss = iGSE.coreloss(tvs, self.B_waveform, self.material.alpha, self.material.beta,
                                      k=self.material.k / 1000)
        self.iGSE_loss = iGSE_loss
        if self.Volumetric_losses:
            self.iGSE_PE = round(100 * (self.Volumetric_losses - self.iGSE_loss) / self.Volumetric_losses, 2)
        return iGSE_loss

    def GSE(self) -> float:

        GSE_loss = self.material.k * (self.Frequency ** self.material.alpha) * (self.Bpk ** self.material.beta)

        self.GSE_loss = GSE_loss
        if self.Volumetric_losses:
            self.GSE_PE = round(100 * (self.Volumetric_losses - self.GSE_loss) / self.Volumetric_losses, 2)
        return GSE_loss

    # Make material a mystery material
    def mystify(self):
        for i in range(0, len(self.material_oneHot)):
            self.material_oneHot[i] = 0
        self.mysteryBoolean = 1

# Takes a folder with core loss csv data for a material, returns an array of coreSample items
# Note: the folder path should not include a trailing slash
def getCoreData(folder_path: str, material: coreMaterial):
    B, f, H, T, Pv = [], [], [], [], []

    # Get flux waveform
    filepath = os.path.join(folder_path, "B_waveform[T].csv")
    B = np.loadtxt(filepath, delimiter=',')

    # Get frequency data
    filepath = os.path.join(folder_path, "Frequency[Hz].csv")
    f = np.loadtxt(filepath, delimiter=',')

    # Get magnetic field strength waveform
    filepath = os.path.join(folder_path, "H_waveform[Am-1].csv")
    H = np.loadtxt(filepath, delimiter=',')

    # Get temperature data
    filepath = os.path.join(folder_path, "Temperature[C].csv")
    T = np.loadtxt(filepath, delimiter=',')

    # Get loss data
    filepath = os.path.join(folder_path, "Volumetric_losses[Wm-3].csv")
    Pv = np.loadtxt(filepath, delimiter=',')

    coreArr = []

    # Make an array of coreSample objects
    for i in range(0, B.shape[0]):
        coreArr.append(coreSample(material, B[i].tolist(), f[i], H[i].tolist(), T[i], Pv[i]))

    return coreArr


def pickle_data(material_name: str, material_data, mode: str):
    if mode == "write":
        with open(material_name, 'wb') as file:
            pickle.dump(material_data, file)
            return

    elif mode == "read":
        with open(material_name, 'rb') as file:
            return pickle.load(file)

    else:
        raise ValueError("Mode must be read or write")


def unique_derivatives(waveform, plot=False, verbose=False) -> int:
    derivative = []
    # take derivatives of waveform
    for i in range(1, len(waveform)):
        slope = waveform[i] - waveform[i - 1]
        # round to 1 sigfig
        if slope:
            slope_rounded = round(slope, -int(math.floor(math.log10(abs(slope))) - 1))
        else:
            slope_rounded = slope
        derivative.append(slope_rounded)

    # count the unique slopes
    derivative = np.asarray(derivative, dtype=float)
    unique, unique_counts = np.unique(derivative, return_counts=True)

    if verbose:
        print(
            f"\nOriginally, there were {len(unique)} unique derivatives: {unique} that appear at these counts {unique_counts}")

    # duty cycle would have to be less than 1% for this to eliminate anything useful
    # throw out slopes that don't appear frequently
    useless_indices = np.where(unique_counts < 10)
    unique = np.delete(unique, useless_indices)
    unique_counts = np.delete(unique_counts, useless_indices)

    if verbose:
        print(
            f"Then after checking for >X appearances, there were {len(unique)} unique derivatives: {unique} that appear at these counts {unique_counts}")

    # strip out slopes that have been counted more than once
    useless_indices = []
    for i in range(len(unique)):
        for j in range(len(unique)):
            # makes sure that similarities aren't double counted
            if j not in useless_indices and j != i:
                percent_difference = abs((unique[i] - unique[j]) / unique[j])
                if unique[j] == 0:
                    pass
                # if not more than 5% different, probably is just the same slope
                if percent_difference < 0.1:
                    useless_indices.append(i)
                    break
            else:
                pass
    useless_indices = np.asarray(useless_indices, dtype=int)
    unique = np.delete(unique, useless_indices)
    unique_counts = np.delete(unique_counts, useless_indices)

    if verbose:
        print(
            f"After stripping close ones there are {len(unique)} unique derivatives: {unique} that appear at these counts {unique_counts}")

    # plot = random.random() < .002
    if plot:
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(np.arange(0, len(derivative), 1), derivative)  # Plot some data on the axes.
        ax.set_title(f"Derivative of non-sinusoidal waveform")
        plt.show()

    # return number of unique slopes
    return len(unique)

def classify_waveform(waveform, plot=False, percent_to_plot=1.0):
    # Generate fourier transform and power spectral density
    extension = np.tile(waveform, 5)
    fft = np.fft.fft(extension)
    power_spectral_density = np.abs(fft) ** 2

    # Freq w/max power
    max_power_freq = np.argmax(power_spectral_density)

    # Ratio of max power to total frequency
    power_ratio = power_spectral_density[max_power_freq] / np.sum(power_spectral_density)

    if power_ratio > 0.499:
        oneHot = [1, 0, 0]
        waveform_type = 'sinusoid'
        if plot:
            plot = random.random() < percent_to_plot
        if plot:
            fig, ax = plt.subplots()  # Create a figure containing a single axes.
            ax.plot(np.arange(0, len(extension), 1), extension)  # Plot some data on the axes.
            ax.set_title(f"Sinusoid, power_ratio={power_ratio}")
            plt.show()
        return oneHot, waveform_type

    else:
        num_slopes = unique_derivatives(waveform, plot=False, verbose=False)

        if num_slopes <= 2:
            waveform_type = 'triangle'
            oneHot = [0, 1, 0]
        elif num_slopes >= 3:
            waveform_type = 'trapezoid'
            oneHot = [0, 0, 1]

        if plot:
            plot = random.random() < percent_to_plot

        if plot:
            fig, ax = plt.subplots()  # Create a figure containing a single axes.
            ax.plot(np.arange(0, len(extension), 1), extension)  # Plot some data on the axes.
            ax.set_title(f"{waveform_type}, power_ratio={power_ratio}")
            plt.show()

        return oneHot, waveform_type


def calc_avg_PE(coreData, material, spotCheck = False, verbose=False):
    number_sinusoids = len([x for x in coreData if x.waveform_type == 'sinusoid'])
    number_triangles = len([x for x in coreData if x.waveform_type == 'triangle'])
    number_trapezoids = len([x for x in coreData if x.waveform_type == 'trapezoid'])

    if verbose:
        print(f"Num sinusoids: {number_sinusoids}\n"
              f"Num triangles: {number_triangles}\n"
              f"Num trapezoids: {number_trapezoids}")

    for i, sample in enumerate(coreData):
        waveform_type = sample.waveform_type

        if spotCheck:
            # For spot checking waveform classification
            plot = random.random() < .01
            if plot:
                fig, ax = plt.subplots()  # Create a figure containing a single axes.
                ax.plot(np.arange(0, len(sample.B_waveform), 1), sample.B_waveform)  # Plot some data on the axes.
                ax.set_title(f"{waveform_type}, B excitation")
                plt.show()

        sample.iGSE()
        sample.GSE()

    # Calculate average absolute percent error
    overall_iGSE_mean_percentError = np.mean(
        np.asarray([np.abs(sample.iGSE_PE) for sample in coreData]))
    overall_GSE_mean_percentError = np.mean(
        np.asarray([np.abs(sample.GSE_PE) for sample in coreData]))
    if verbose:
        print(f"\"{material.name}\" iGSE Absolute Mean Percent Error (all waveshapes): ", round(overall_iGSE_mean_percentError, 2),
              "%")
        print(f"\"{material.name}\" GSE Absolute Mean Percent Error (all waveshapes): ", round(overall_GSE_mean_percentError, 2),
              "%")

    sine_iGSE_mean_percentError = np.mean(
        np.asarray([np.abs(sample.iGSE_PE) for sample in coreData if sample.waveform_type == "sinusoid"]))
    sine_GSE_mean_percentError = np.mean(
        np.asarray([np.abs(sample.GSE_PE) for sample in coreData if sample.waveform_type == "sinusoid"]))
    if verbose:
        print(f"\"{material.name}\" iGSE Absolute Mean Percent Error (sinusoids): ", round(sine_iGSE_mean_percentError, 2), "%")
        print(f"\"{material.name}\" GSE Absolute Mean Percent Error (sinusoids): ", round(sine_GSE_mean_percentError, 2), "%")

    triangle_iGSE_mean_percentError = np.mean(
        np.asarray([np.abs(sample.iGSE_PE) for sample in coreData if sample.waveform_type == "triangle"]))
    triangle_GSE_mean_percentError = np.mean(
        np.asarray([np.abs(sample.GSE_PE) for sample in coreData if sample.waveform_type == "triangle"]))
    if verbose:
        print(f"\"{material.name}\" iGSE Absolute Mean Percent Error (triangles): ", round(triangle_iGSE_mean_percentError, 2),
              "%")
        print(f"\"{material.name}\" GSE Absolute Mean Percent Error (triangles): ", round(triangle_GSE_mean_percentError, 2),
              "%")

    trapezoid_iGSE_mean_percentError = np.mean(
        np.asarray([np.abs(sample.iGSE_PE) for sample in coreData if sample.waveform_type == "trapezoid"]))
    trapezoid_GSE_mean_percentError = np.mean(
        np.asarray([np.abs(sample.GSE_PE) for sample in coreData if sample.waveform_type == "trapezoid"]))
    if verbose:
        print(f"\"{material.name}\" iGSE Absolute Mean Percent Error (trapezoids): ", round(trapezoid_iGSE_mean_percentError, 2),
              "%")
        print(f"\"{material.name}\" GSE Absolute Mean Percent Error (trapezoids): ", round(trapezoid_GSE_mean_percentError, 2),
              "%")

def GSE_objective(params, X, y):
    alpha, beta, k = params
    y_hat = k + (X[:,0]*alpha) + (X[:,1]*beta)
    residuals = y - y_hat

    return np.sum(residuals**2)

# Solve for optimal S-params by taking log of equation and doing multiple linear regression
def optimize_steinmetz_params_solver( coreData, verbose=False):
    # Multiple linear regression to curve fit Steinmetz parameters to the provided core loss data
    if verbose:
        print(f"Using OLS model to solve for optimal Steinmetz parameters for {coreData[0].material.name}")

    # optimizes for sinusoids (because that's what GSE is for)
    samples = np.asarray([sample for sample in coreData if sample.waveform_type == 'sinusoid'])
    material = coreData[0].material

    frequencies = np.asarray([sample.Frequency for sample in samples])
    Bpks = np.asarray([sample.Bpk for sample in samples])
    y = np.asarray([sample.Volumetric_losses for sample in samples])

    X = np.column_stack((np.log(frequencies), np.log(Bpks)))
    Y = np.log(y)

    # constrain the S-params so their predictive behavior is similar
    constraints = [
        {'loc': 'x0', 'type': 'ineq', 'fun': lambda params: params[0] - 1.6},  # Constrain >= 1
        {'loc': 'x0', 'type': 'ineq', 'fun': lambda params: 2.0 - params[0]},  # Constrain <= 2
        {'loc': 'x1', 'type': 'ineq', 'fun': lambda params: params[1] - 2.3},
        {'loc': 'x1', 'type': 'ineq', 'fun': lambda params: 2.7 - params[1]},
    ]
    initial_params = [1.6,2.7,0.1]

    # Fit the model with constraints
    result = minimize(GSE_objective, initial_params, args=(X, Y), constraints=constraints)

    # Extract optimized parameters
    optimal_k = np.exp(result.x[2])
    optimal_alpha = result.x[0]
    optimal_beta = result.x[1]

    material.alpha, material.beta, material.k = optimal_alpha, optimal_beta, optimal_k
    # Get GSE for each sample should be fast
    for sample in samples:
        sample.GSE()

    # Find the avg absolute pct error
    best_percent_error = np.mean([np.abs(sample.GSE_PE) for sample in samples])

    # # Tries to improve optimized S-params by removing outliers, doesn't work super well
    # outliers = 0
    # samples_to_delete = []
    # print(f"There are {len(samples)} samples")
    # for i, sample in enumerate(samples):
    #     if sample.GSE_PE > 1.75 * best_percent_error:
    #         samples_to_delete.append(i)
    #         outliers += 1
    # samples = np.delete(samples, i)
    # print(f"Number of outliers: {outliers}")
    # if outliers > 0:
    #     optimal_alpha, optimal_beta, optimal_k, best_percent_error = optimize_steinmetz_params_solver(samples, verbose=True)

    if verbose:
        print(f"Optimization complete, setting optimal Steinmetz parameters for {coreData[0].material.name}:")
        print(f"alpha = {optimal_alpha}\nbeta = {optimal_beta}\nki = {optimal_k}")
        print(f"The average GSE  (abs) percent error for sinusoid samples with optimal parameters is: {best_percent_error}")

    return optimal_alpha, optimal_beta, optimal_k, best_percent_error


def make_RF_heatmap(allData, numFrequencyBins:int, numExcitationBins:int, model = 'RF', material_name: str = "all", waveshape: str = "all", absolute=True, new_color_range=True, vmax=100, vmin=0):
    print("Making Heatmap Plot")

    if absolute:
        sign = "Unsigned"
    else:
        sign = "Signed"

    if material_name != "all":
        relevantData = [sample for sample in allData if sample.material.name == material_name]
    else:
        relevantData = allData
    if waveshape != "all":
        relevantData = [sample for sample in relevantData if sample.waveform_type == waveshape]

    max_frequency = max(relevantData, key=lambda sample: sample.Frequency).Frequency
    min_frequency = min(relevantData, key=lambda sample: sample.Frequency).Frequency
    max_B = max(relevantData, key=lambda sample: sample.Bpk).Bpk
    min_B = min(relevantData, key=lambda sample: sample.Bpk).Bpk
    frequency_bin_size = (max_frequency - min_frequency) / numFrequencyBins
    excitation_bin_size = (max_B - min_B) / numExcitationBins
    # assumes frequency on x-axis
    material_distribution = np.empty((numFrequencyBins, numExcitationBins), dtype=object)

    # Fill the array with Python lists
    for i in range(numFrequencyBins):
        for j in range(numExcitationBins):
            material_distribution[i, j] = []

    # tally the number of samples in each combination of quantizations
    for sample in relevantData:
        frequency_bin = min(numFrequencyBins - 1, int((sample.Frequency - min_frequency) // frequency_bin_size))
        excitation_bin = min(numExcitationBins - 1, int((sample.Bpk - min_B) // excitation_bin_size))
        # Add sample to correct bin
        material_distribution[frequency_bin][excitation_bin].append(sample)

    X = []
    Y = []
    Z = []
    volume = []

    for m in range(0, numFrequencyBins):
        for n in range(0, numExcitationBins):
            num_samples = len(material_distribution[m][n])
            if num_samples == 0:
                continue
            if not absolute:
                RF_PE_arr = [sample.randomForest_PE for sample in material_distribution[m][n]]
                iGSE_PE_arr = [sample.iGSE_PE for sample in material_distribution[m][n]]
                if model == 'iGSE':
                    Z.append(np.mean(iGSE_PE_arr))
                else:
                    Z.append(np.mean(RF_PE_arr))
            elif absolute:
                RF_PE_arr = [abs(sample.randomForest_PE) for sample in material_distribution[m][n]]
                iGSE_PE_arr = [abs(sample.iGSE_PE) for sample in material_distribution[m][n]]
                if model == 'iGSE':
                    Z.append(np.mean(iGSE_PE_arr))
                else:
                    Z.append(np.mean(RF_PE_arr))
            freq = min_frequency + (m + .5) * frequency_bin_size
            Bpk = min_B + (n + .5) * excitation_bin_size
            X.append(freq)
            Y.append(Bpk)
            volume.append(num_samples)

    volume_min = 0
    volume_max = max(volume)

    # Normalize volume plot to match reasonable sizes in python plot
    volume = np.asarray(volume)
    volume = ((volume - volume_min)*(1250/(float(numExcitationBins)/25)) / (volume_max - volume_min))

    if( new_color_range ):
        # Determine the common range for the color mapping RF (iGSE/GSE Range is too high for clarity)
        vmax = max(abs(max(Z)), abs(min(Z)))
        vmin = -vmax
    RF_norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=vmax/2, vmax=vmax)

    fig_RF = plot_heatmap(np.asarray(X), np.asarray(Y), np.asarray(Z), np.asarray(volume), RF_norm,
                           f"{sign} {model} Error vs. Bpk, freq for {material_name} material with {waveshape} waveshapes\n")
                           # f"Avg % Error: {np.mean([abs(sample.randomForest_PE) for sample in relevantData])}")
    plt.show()
    return vmax, vmin

def plot_histogram(errorData, material):
    fig_hist = plt.figure()
    plt.hist(errorData, bins=100, density=True, color='blue', alpha=0.5, edgecolor='black')
    _points = [np.mean(errorData), np.percentile(errorData, 95), np.percentile(errorData, 99), np.max(errorData)]

    plt.axvline(np.mean(errorData), color='green', linestyle='--', linewidth=2,
                label=f'Mean Error [%]: {np.mean(errorData)}')
    plt.axvline(np.percentile(errorData, 95), color='yellow', linestyle='--', linewidth=2,
                label=f'95th Percentile Error [%]: {np.percentile(errorData, 95)}')
    plt.axvline(np.percentile(errorData, 99), color='magenta', linestyle='--', linewidth=2,
                label=f'99th Percentile Error [%]: {np.percentile(errorData, 99)}')
    plt.axvline(np.max(errorData), color='red', linestyle='--', linewidth=2,
                label=f'Max Error [%]: {np.max(errorData)}')


    # Add labels and legend
    plt.xlabel('Relative Error of Sample [%]')
    plt.ylabel('Ratio of Data Points')
    plt.title(f'{material.name} Error Distribution')
    plt.legend()

    # Show the plot
    plt.show()

def average_neighboring_points(X,Y,Z):
    # Round x and y arrays so that nearby points can be grouped
    # otherwise heatmap is unreadable due to overlapping points
    X = np.asarray([round(sample, -4) for sample in X])
    Y = np.asarray([round(sample, 2) for sample in Y])
    # Create a dictionary to store the averaged z-values for each (x, y) coordinate
    averaged_z = {}
    # Calculate the average z-value for each (x, y) coordinate
    for xi, yi, zi in zip(X, Y, Z):
        key = (xi, yi)
        if key in averaged_z:
            averaged_z[key].append(zi)
        else:
            averaged_z[key] = [zi]

    x_avg, y_avg, z_avg, volume = zip(*[(key[0], key[1], np.mean(values), len(values)) for key, values in averaged_z.items()])
    return x_avg, y_avg, z_avg, volume

def plot_heatmap(X, Y, Z, size, norm, title):
    fig = plt.figure()
    # Plotting the averaged points
    plt.scatter(X, Y, c=Z, cmap='Reds', s=size, alpha=1, edgecolors='black', linewidths=1, norm=norm)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Absolute Core Loss Error (% Error)', fontsize=22)
    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('Bpk (T)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=22)
    return fig

def calc_rmse(sample_array):
    iGSE_squared_errors = [(sample.iGSE_loss - sample.Volumetric_losses)**2 for sample in sample_array]
    RF_squared_errors = [(sample.randomForest_loss - sample.Volumetric_losses)**2 for sample in sample_array]

    iGSE_rmse = np.sqrt(np.mean(np.asarray(iGSE_squared_errors)))
    RF_rmse = np.sqrt(np.mean(np.asarray(RF_squared_errors)))

    return iGSE_rmse, RF_rmse

def make_final_prediction_csv(coreData:list[coreSample], filename:str):
    with open(filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for sample in coreData:
            csv_writer.writerow([sample.randomForest_loss])

    print(f'CSV of predictions completed under filename {filename}')