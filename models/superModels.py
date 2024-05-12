import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
path="."

from scipy.stats import gmean
import matplotlib.pyplot as plt
material="Volumetric_Loss_Material D.csv"
materials=["Volumetric_Loss_Material A.csv",
        "Volumetric_Loss_Material B.csv",
        "Volumetric_Loss_Material C.csv",
        "Volumetric_Loss_Material D.csv",
        "Volumetric_Loss_Material E.csv"]
schools = []
for material in materials:
    data={}
    for school in os.listdir(path):  # list of subdirectories and files
        pathTrue=os.path.isdir(school)
        # print(school,os.path.isdir(school))
        if pathTrue:
            # print(school)
            matpath= f"{path}/{school}/Result/{material}"
            if os.path.isfile(matpath):
                datai=  pd.read_csv(f"{path}/{school}/Result/{material}",header=None)
                schools.append(school)
                # print("Datai", datai.describe())
                print("school data" , matpath)
                data[school]=np.abs(datai.values.reshape(-1))
                # print("data", data.describe())
            else:
                print(f"{school} not found", matpath)

    data=pd.DataFrame.from_dict(data)
    del data["Mondragon"]
    print(data.columns)
    # Select university to use in the modeling of std correlation with percentile errors
    # uncomment selection to disable selection 
    # data=data[[ 'Paderborn', 'Sydney','Tsinghua', 'TUDelft', 'XJTU']]
    # data=data[[ 'ASU', 'Tribhuvan']]

    # data['av'] = data.geomean(axis=1)
    # print(data.describe())
    dataMean = data.mean(axis=1)
    datastd = data.std(axis=1)

    data["gmean"] = data.apply(gmean, axis=1)
    data['std']=datastd/dataMean
    # data['gmean'] =dataMean

    data.to_csv(material+"_ave.csv")

    correct="../finaltest/EvaluationKit/EvaluationKit/Measured_"+material
    # print(correct)
    datac = pd.read_csv(correct,header=None)
    print("school, median error, 95 th percentile,99 percentile,99.9 percentile")

    for each in schools:
        if each in data.columns:
            data['error']=np.abs(data[each].values.reshape(-1)-datac.values.reshape(-1))/datac.values.reshape(-1)
            print(each, np.percentile(data['error'],50),np.percentile(data['error'],95),np.percentile(data['error'],99),np.percentile(data['error'],99.9))

    data['error']=np.abs(data["Paderborn"].values.reshape(-1)-datac.values.reshape(-1))/datac.values.reshape(-1)
    print("gmean", np.percentile(data['error'],50),np.percentile(data['error'],95),np.percentile(data['error'],99),np.percentile(data['error'],99.9))

    plt.figure()

    plt.plot(data['std'],data['error'],"*")
    plt.ylabel("Error(%)")
    plt.xlabel("Standard deviation")
    plt.savefig(f"{material}_cv.png")
    
    
    data.to_csv(material+"_hist.csv")
    
    plt.figure()
    print(np.corrcoef(data['std'],data['error']))
    datax= data.copy(True)
    datax = datax-data['gmean']
    # corr = datax.cov()
    # print(corr)
    # corr.style.background_gradient(cmap='coolwarm')
    # print(datax.corr())
    # plt.show()
   
    factor = "gmean"
    factors = [x  for x in list(data.columns) if x in schools]
    factors.append("gmean")
    print("Hi htere")
    plt.figure()

    for factor in factors:
        n = len(data)

        datacopy=data.copy(True)
        dataccopy=datac.copy(True)
        percentileErrors = []
        percentileRejectionRatio = []
        rejectThreshold=[]
        for RejectionRatio in range(1,99):  # percentage of data select by std across models
            reject=np.percentile(data['std'].values,RejectionRatio)
            # Select a subset of the data  
            dataccopy= datac[data['std'].values<=reject]
            datacopy=data[data['std'].values<=reject]

            datacopy['error']=np.abs(datacopy[factor].values.reshape(-1)-dataccopy.values.reshape(-1))/dataccopy.values.reshape(-1)
            # print(len(datacopy))
            percentileErrors.append(np.percentile(datacopy['error'],95)*100)
            # percentileErrors.append(np.max(datacopy['error'])*100)

            percentileRejectionRatio.append(RejectionRatio*100)
            rejectThreshold.append(datacopy['std'].mean()*100)
            # print(len(data)/n*100, "%","remaining")
        if factor == "gmean":
            plt.plot(rejectThreshold,percentileErrors,"+-")
        else:
            plt.plot(rejectThreshold,percentileErrors)

        plt.yscale('log')

    plt.title(f"95th Percentile Errors(%) vs rejection threshold\n normalized std error(%) {material}")
    plt.plot(rejectThreshold,rejectThreshold,"*-")
    plt.ylabel("95th percentile Error(%)")
    plt.xlabel("mean (subset) coefficient of variation(%)")
    x=factors.copy()
    x.append("coefficient of variation")
    plt.legend(x,loc='center left')
        # plt.savefig(factor+"_"+material+".png")
    plt.grid(which="both")
    plt.tight_layout()

    # plt.savefig(material+".png")
    plt.savefig(material+".pdf", format="pdf", bbox_inches="tight")
