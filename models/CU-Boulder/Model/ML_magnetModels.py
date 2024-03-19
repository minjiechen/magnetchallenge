import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from data_processing import coreSample

def error_95percentile(y_true, y_pred):
    errors = np.abs((y_pred - y_true) / y_true)
    return -np.percentile(errors, 95)

def randomForest_magnetModel(X_train, verbose=False):
    if verbose:
        print("Training Random Forest Regression Model")

    # Split up data into features and targets
    training_features = get_features(X_train)

    training_targets = np.asarray([(sample.Volumetric_losses/sample.iGSE_loss) for sample in X_train])

    # Scale the input features
    scaler = StandardScaler()
    training_features = scaler.fit_transform(training_features)

    # Train and test the model, random state gives repeatability to model
    my_model = RandomForestRegressor(n_estimators=100, random_state=37, n_jobs=-1, max_depth=30)
    my_model.fit(training_features, training_targets)
    train_predictions = my_model.predict(training_features)

    # update the samples with the new data
    for i, sample in enumerate(X_train):
        sample.randomForest_loss = train_predictions[i] * sample.iGSE_loss
        sample.randomForest_PE = round(100 * (sample.randomForest_loss - sample.Volumetric_losses) / sample.Volumetric_losses, 2)

    if verbose:
        print("[Training] Mean Absolute % Error (RF)", np.mean(np.asarray([np.abs(sample.randomForest_PE) for sample in X_train])))

    return my_model, scaler

def RF_test_model(model, scaler, X_test, verbose=False):
    if verbose:
        print(f"Starting RF testing on a set of {len(X_test)} samples")
    test_features = get_features(X_test)

    # Scale the input features
    test_features = scaler.transform(test_features)
    # Use model to generate predictions
    test_predictions = model.predict(test_features)

    for i, sample in enumerate(X_test):
        sample.randomForest_loss = test_predictions[i] * sample.iGSE_loss
        sample.randomForest_PE = round(100 * (sample.Volumetric_losses - sample.randomForest_loss) / sample.Volumetric_losses, 2)
        # # Print outliers to debug
        # if sample.randomForest_PE > 200 or sample.randomForest_PE < -200:
        #     plt.plot(np.arange(len(sample.B_waveform)), sample.B_waveform)
        #     plt.show()
        #     plt.plot(np.arange(len(sample.H_waveform)), sample.H_waveform)
        #     plt.show()
        #     print("Outlier, debug")
    if verbose:
        GSE_errors = np.asarray([np.abs(sample.GSE_PE) for sample in X_test])
        iGSE_errors = np.asarray([np.abs(sample.iGSE_PE) for sample in X_test])
        errorData = np.asarray([np.abs(sample.randomForest_PE) for sample in X_test])

        print("[Test] Mean Absolute % Error (GSE):", np.mean(GSE_errors))
        print("GSE 95th % Error -->", np.percentile(GSE_errors, 95))
        print("% Error Sinusoid (RF)", np.mean(np.asarray([np.abs(sample.GSE_PE) for sample in X_test if sample.waveform_oneHot[0] == 1])))
        print("% Error Triangle (RF)", np.mean(np.asarray([np.abs(sample.GSE_PE) for sample in X_test if sample.waveform_oneHot[1] == 1])))
        print("% Error Trapezoid (RF)", np.mean(np.asarray([np.abs(sample.GSE_PE) for sample in X_test if sample.waveform_oneHot[2] == 1])))

        print("[Test] Mean Absolute % Error (iGSE):", np.mean(iGSE_errors))
        print("iGSE 95th % Error -->", np.percentile(iGSE_errors, 95))
        print("% Error Sinusoid (RF)", np.mean(np.asarray([np.abs(sample.iGSE_PE) for sample in X_test if sample.waveform_oneHot[0] == 1])))
        print("% Error Triangle (RF)", np.mean(np.asarray([np.abs(sample.iGSE_PE) for sample in X_test if sample.waveform_oneHot[1] == 1])))
        print("% Error Trapezoid (RF)", np.mean(np.asarray([np.abs(sample.iGSE_PE) for sample in X_test if sample.waveform_oneHot[2] == 1])))

        print("[Test] Mean Absolute % Error (RF):", np.mean(errorData))
        print("95th Percentile % Error -->", np.percentile(errorData, 95))
        print("% Error Sinusoid (RF)", np.mean(np.asarray([np.abs(sample.randomForest_PE) for sample in X_test if sample.waveform_oneHot[0] == 1])))
        print("% Error Triangle (RF)", np.mean(np.asarray([np.abs(sample.randomForest_PE) for sample in X_test if sample.waveform_oneHot[1] == 1])))
        print("% Error Trapezoid (RF)", np.mean(np.asarray([np.abs(sample.randomForest_PE) for sample in X_test if sample.waveform_oneHot[2] == 1])))

def RF_predict_single_sample(model, scaler, X_single, verbose=False):
    test_features = get_features(X_single)
    # Scale the input features
    test_features = scaler.transform(test_features)
    test_prediction = model.predict(test_features)

    return test_prediction

def RF_load_and_predict(material, frequency, B, temp):
    rf_model = load('rf_coreLossModel.joblib')
    rf_scaler = load('rf_coreLossScaler.joblib')

    # Volumetric losses and H not necessarily known - given as None and 50e3 - shouldn't affect outcome
    sample = coreSample(material, B, frequency, None, temp, 50e3)

    # generate multiplicative correction factor
    gamma = RF_predict_single_sample(rf_model, rf_scaler, [sample], verbose=True)

    predicted_loss = gamma[0] * sample.iGSE()

# Function returns array of features - allows uniform changing of used features across functions
def get_features( samples: list[coreSample] )->list:
    return [[sample.Bpk,                   # Peak flux density [mT]
            sample.Frequency,             # Frequency [Hz]
            sample.Temperature,           # Temperature [C]
            sample.waveform_oneHot[0],    # One Hot Encoding of Sinusoid
            sample.waveform_oneHot[1],    # One Hot Encoding of Triangle
            sample.waveform_oneHot[2],    # One Hot Encoding of Trapezoid
            sample.material_oneHot[0],
            sample.material_oneHot[1],
            sample.material_oneHot[2],
            sample.material_oneHot[3],
            sample.material_oneHot[4],
            sample.material_oneHot[5],
            sample.material_oneHot[6],
            sample.material_oneHot[7],
            sample.material_oneHot[8],
            sample.material_oneHot[9],
            sample.material_oneHot[10],
            sample.material_oneHot[11],
            sample.material_oneHot[12],
            sample.material_oneHot[13],
            sample.material_oneHot[14],
            sample.material.alpha,
            sample.material.beta,
            sample.material.k
            #TODO: note that mystery boolean has been removed
            ] for sample in samples]