import data_processing
import ML_magnetModels
import numpy as np
from joblib import dump, load

# List for the MagNet materials - S-params have been replaced with optimal [for sinusoids]
materials = {
    "N87": data_processing.coreMaterial("N87", 1.6510549518850788, 2.5684286935373444, 0.22055180357325818, 0, 500e3, 2.2e3),
    "N49": data_processing.coreMaterial("N49", 1.60000000023557, 2.69999999985035, 0.407238401988939, 1, 1000e3, 1.5e3),
    "N30": data_processing.coreMaterial("N30", 1.8626024147577311, 2.3206505881723425, 0.013038781255903584, 2, 400e3, 4.3e3),
    "N27": data_processing.coreMaterial("N27", 1.6576804020477185, 2.5407547999961384, 0.1780713195465577, 3, 150e3, 2e3),
    "78": data_processing.coreMaterial("78", 1.8100082442688465, 2.5639321086315663, 0.02587545313201249, 4, 500e3, 2.3e3),
    "77": data_processing.coreMaterial("77", 1.7506925886801696, 2.522174370283959, 0.04867133538302141, 5, 100e3, 2e3),
    "3F4": data_processing.coreMaterial("3F4", 1.60000000012103, 2.69999999996669, 0.881119858361143, 6, 2000e3, .9e3),
    "3E6": data_processing.coreMaterial("3E6", 1.88110659998852, 2.29999999998581, 0.0103084497401855, 7, 100e3, 12e3),
    "3C94": data_processing.coreMaterial("3C94", 1.7467863426492043, 2.494227634660561, 0.03668179851965532, 8, 300e3, 2.3e3),
    "3C90": data_processing.coreMaterial("3C90", 1.6677460483494921, 2.6318944941115, 0.14107483868460985, 9, 200e3, 2.3e3),
    # Steinmetz parameters optimized and constrained on mystery materials
    "Material A": data_processing.coreMaterial("Material A", 1.9766324314273338, 2.50971829136747, 0.006710336398570659, 10, 0, 0),
    "Material B": data_processing.coreMaterial("Material B", 1.9999999904654548, 2.3000000126033426, 0.002200573427861818, 11, 0, 0),
    "Material C": data_processing.coreMaterial("Material C", 2.0000000000000497, 2.5616543843453896, 0.0018325958912108623, 12, 0, 0),
    "Material D": data_processing.coreMaterial("Material D", 1.5999999999933523, 2.661628051178331, 0.2883281947124199, 13, 0, 0),
    "Material E": data_processing.coreMaterial("Material E", 1.5999999999991483, 2.700000000000582, 0.3544047687117679, 14, 0, 0),
}

# Simply change the material to regenerate predictions for that material (assuming it is present in the testing folder)
Material = "Material A"

def load_dataset(in_file1="../Testing/" + Material + "/B_Field.csv",
                 in_file2="../Testing/" + Material + "/Frequency.csv",
                 in_file3="../Testing/" + Material + "/Temperature.csv"):
    data_B = np.genfromtxt(in_file1, delimiter=',')  # N by 1024, in T
    data_F = np.genfromtxt(in_file2, delimiter=',')  # N by 1, in Hz
    data_T = np.genfromtxt(in_file3, delimiter=',')  # N by 1, in C

    return data_B.tolist(), data_F.tolist(), data_T.tolist()

def core_loss(data_B, data_F, data_T):
    material_data = []
    for i in range(len(data_F)):
        material_data.append(
            data_processing.coreSample(
                materials[Material],
                data_B[i],
                data_F[i],
                None, # No H waveform provided in testing
                data_T[i],
                None # No loss provided in testing
            )
        )

    # Load model
    rf_model = load('rf_coreLossModel.joblib')
    rf_scaler = load('rf_coreLossScaler.joblib')

    model_params = rf_model.get_params()
    print(f"Model Parameters:\n {model_params}")
    num_parameters = sum(param.size if hasattr(param, 'size') else 1 for param in model_params.values())
    print(f"\nNumber of Parameters: {num_parameters}")

    # Extract features from custom class
    test_features = ML_magnetModels.get_features(material_data)

    # Scale the input features
    test_features = rf_scaler.transform(test_features)

    # Generate multiplicative correction factor
    test_gammas = rf_model.predict(test_features)

    # Generate predictions by combining with equation-based model
    test_predictions = np.zeros(len(data_F))
    for i, sample in enumerate(material_data):
        test_predictions[i] = test_gammas[i] * sample.iGSE()

    # Save predictions
    with open("../Results/Volumetric_Loss_" + Material + ".csv", "w") as file:
        np.savetxt(file, test_predictions)
        file.close()

    print(f'Model inference for {Material} is finished!')
    return test_predictions

if __name__ == '__main__':
    B,f,T = load_dataset()
    core_loss(B,f,T)