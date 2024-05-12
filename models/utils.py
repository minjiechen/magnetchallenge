import os
import numpy as np
import pandas as pd
from scipy.stats import norm

path="."

from scipy.stats import gmean
import matplotlib.pyplot as plt

def cv_data(material,pointN):
    # material="Volumetric_Loss_Material D.csv"
    materials=[f"Volumetric_Loss_Material {material}.csv"]
    print(materials)
    schools = []
    for material in materials:
        data={}
        for school in os.listdir(path):  # list of subdirectories and files
            pathTrue=os.path.isdir(school)
            if pathTrue:
                matpath= f"{path}/{school}/Result/{material}"
                if os.path.isfile(matpath):
                    datai=  pd.read_csv(f"{path}/{school}/Result/{material}",header=None)
                    schools.append(school)
                    data[school]=np.abs(datai.values.reshape(-1))
                else:
                    print(f"{school} not found", matpath)

        data=pd.DataFrame.from_dict(data)
        data=data.values
        data=data[pointN,:]

        correct="../finaltest/EvaluationKit/EvaluationKit/Measured_"+material
        datac = pd.read_csv(correct,header=None)
        datac=datac.values
        datac=datac[pointN,:]
        datacorrect=datac
        Error_mean = (np.mean(data)-datacorrect)/datacorrect*100
        return data, Error_mean,datacorrect

    return data,data,datacorrect

    # material="D"
for material in "ABCDE":
    for i in range(1000):

        A, B, datacorrect= cv_data(material,i)
        # plt.figure()
        cv= np.std(A)/np.mean(A)*100
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        # ax1.set_xlabel('time (s)')
        ax1.set_ylabel('#Model outputs', color=color)
        # ax1.plot(t, data1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        # ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
        # ax1.plot(t, data2, color=color)
        # ax1.tick_params(axis='y', labelcolor=color)
        ax1.hist(A)
        ax1.plot(np.mean(A),1, "*")
        ax1.plot(datacorrect,1, "+")
        ax1.axvline(x = np.mean(A)+2*np.std(A)/np.sqrt(len(A)), color = 'b', label = 'right bound')
        ax1.axvline(x = np.mean(A)-2*np.std(A)/np.sqrt(len(A)), color = 'g', label = 'left bound')
        mean =np.mean(A)
        sd =np.std(A)
          # plt.figure()
        ax2.set_ylabel('Probab', color=color)  # we already handled the x-label with ax1

        x_axis = np.arange(np.min(A), np.max(A), (np.max(A)-np.min(A))/len(A))

        Anormal = norm.pdf(x_axis, mean, sd)
        # ax1.legend(['mean value', "true value", "right=mean+2sigma/sqrt(N)","left=mean-2sigma/sqrt(N)","histogram","normal dist(scaled for graphing)"])

        # ax1.legend(['mean value', "true value", "right=mean+2sigma/sqrt(N)","left=mean-2sigma/sqrt(N)","histogram","normal dist(scaled for graphing)"],loc ="upper right")

        Anormalmax= np.max(Anormal)
        ax2.plot(x_axis, Anormal/Anormalmax,"--")
        ax2.set_ylabel("Probability")
        ax1.legend(['Mean value', "True value", "Right=mean+2sigma/sqrt(N)","Left=mean-2sigma/sqrt(N)","Histogram","Normal dist"])

        # ax2.legend(["Analytic normal dist"])

        ax1.set_xlabel("Core loss(W/m^3)")



        plt.title(f"Mean error: {float(B):0.1f}%, cv: {float(cv):0.1f}%, 2*cv/sqrt(#models) : {float(2*cv/np.sqrt(len(A))):0.1f} %\n Material : {material}, data point index: {i+1}")
        # cv/sqrt(N) is inspired by confidence interval concept given N samples (1sigma-ish)
        plt.xlabel("Core loss(W/m^3)")
        plt.savefig(f"images/{material}/{i}.png")
        # plt.show()
        plt.cla()
        plt.clf()
    
      

        plt.close()