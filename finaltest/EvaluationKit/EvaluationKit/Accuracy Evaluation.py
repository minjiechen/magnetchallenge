import numpy as np

def main():

    material_list = ['Material A','Material B','Material C','Material D','Material E']
    
    for material in material_list:
    
        with open('./Measured_Volumetric_Loss_'+material+'.csv','r') as csv_file:
            loss_meas = np.loadtxt(csv_file, delimiter=",")
            
        with open('./Volumetric_Loss_'+material+'.csv','r') as csv_file:
            loss_pred = np.loadtxt(csv_file, delimiter=",")
            
        if (np.shape(loss_pred) != np.shape(loss_meas) ):
            print('Mismatched matrix shape!')
            
        else:
            
            err = np.abs( (loss_pred - loss_meas) / loss_meas ) *100
            err_avg = np.mean(err)
            err_rms = np.sqrt(np.mean(err**2))
            err_per = np.percentile(err, 95)
            err_max = np.max(err)
            
            df = {
                "Average": np.round(err_avg, 4),
                "RMS": np.round(err_rms, 4),
                "95_Pctl": np.round(err_per, 4),
                "Maximum": np.round(err_max, 4),
                }
                
            print(material)        
            print(df)
    
    print('Evaluation is finished!')
    
if __name__ == "__main__":
    main()