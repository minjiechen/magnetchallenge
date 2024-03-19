"""This is the main inference script for the competition hosts.
Make sure your current working directory (cwd) is at the same dir as this
file."""

import random
import numpy as np
import pandas as pd
from pathlib import Path

from run_cnn_inference import run_inference
from utils.data import ALL_B_COLS

Material = "Material E"

# in order for the Path to be correct, please execute Model_inference.py while
#  your current working directory (cwd) is in the same folder
test_folder_path = Path.cwd().parent / "data" / "input" / "test" / "Testing"
dump_folder_path = Path.cwd().parent.parent / "Result"

# %% Load Dataset


def load_dataset(
    in_file1=test_folder_path / Material / "B_Field.csv",
    in_file2=test_folder_path / Material / "Frequency.csv",
    in_file3=test_folder_path / Material / "Temperature.csv",
):
    data_B = np.genfromtxt(in_file1, delimiter=",")  # N by 1024, in T
    data_F = np.genfromtxt(in_file2, delimiter=",")  # N by 1, in Hz
    data_T = np.genfromtxt(in_file3, delimiter=",")  # N by 1, in C

    return data_B, data_F, data_T


# %% Calculate Core Loss
def core_loss(data_B, data_F, data_T):
    # ================ Wrap your model or algorithm here=======================#
    material_2_model_uid = {
        "A": "d893c778",
        "B": "b6a920cc",
        "C": "c1ced7b6",
        "D": "11672810",
        "E": "5ae50f9e",
    }
    mat_lbl = Material[-1]
    # pick corresponding model uid
    mdl_uid = material_2_model_uid[mat_lbl]
    print(f"Run inference for {Material} with model uid {mdl_uid}")
    # create dataframe
    ##  This is a redundant step. The function "run_inference" would load the correct Testing
    ##  data on its own by detecting the material that corresponds to the model uid in our
    ##  *meta.csv files. For the sake of transparency, we work with the data loaded here.
    df = pd.DataFrame(data_B, columns=ALL_B_COLS).assign(
        freq=data_F, temp=data_T, material=mat_lbl
    )

    # run inference step
    data_P = run_inference(mdl_uid, df)

    # =========================================================================#

    with open(dump_folder_path/ f"Volumetric_Loss_{Material}.csv", "w") as file:
        np.savetxt(file, data_P)
        file.close()

    print("Model inference is finished!")

    return


# %% Main Function for Model Inference


def main():
    # Reproducibility
    random.seed(1)
    np.random.seed(1)

    data_B, data_F, data_T = load_dataset()

    core_loss(data_B, data_F, data_T)


if __name__ == "__main__":
    main()

# End
