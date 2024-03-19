# Notes on running Model_Inference.py
# - ./data/interim contains intermediate processed data from the given dataset.
# - ./data/cGANcheckpoint is where the model weights are saved
#  


import random
import numpy as np
from cgan import cGAN, MagNetDatasetPreparation
Material = "Material E"

#%% Load Dataset

def load_dataset():
    MagNetdatasetPrep = MagNetDatasetPreparation()
    MagNetdatasetPrep.trainingDatasetPreparation(material_type=Material[-1],
                                        training_dataset_path='.\\..\\..\\dataset\\train')
    MagNetdatasetPrep.testingDatasetPreparation(material_type=Material[-1],
                                        testing_dataset_path='.\\..\\..\\dataset\\test')


#%% Calculate Core Loss
def core_loss():
    
    #================ Wrap your model or algorithm here=======================#
    model_cGAN = cGAN(material_type=Material[-1],
                      LOAD_WEIGHT_FROM_H5=True,
                      interim_path=".\\..\\data\\interim",
                      checkpoint_path=".\\..\\data\\cGANcheckpoint",
                      train_mode=False)
    data_P = model_cGAN.predict(model_cGAN.x_test)
     
    #=========================================================================#
    
    with open("./../../Result/Volumetric_Loss_"+Material+".csv", "w") as file:
        np.savetxt(file, data_P)
        file.close()

    print(f'Model inference for {Material} is finished!')
    
    return

#%% Main Function for Model Inference

def main():
    
    # Reproducibility
    random.seed(1)
    np.random.seed(1)

    # load_dataset()  #  test/train *.mat* data for cGAN model included in the /KU Leuven/Model/data/interim, uncomment this line to process the .csv file to *.mat* file
    
    core_loss()
    
if __name__ == "__main__":
    main()
    
# End