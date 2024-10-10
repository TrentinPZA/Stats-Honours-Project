import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def geom_interp():
    appliance_energy_data = pd.read_csv("data/energydata_complete.csv", sep=',',decimal=".",header=0) #Reading in the data, and specifying that the decimal points are commas so that they are instantly imported as floats.  
    appliance_energy_data = appliance_energy_data.iloc[10:26]# Keep only the first half of the rows
    appliance_energy_data=appliance_energy_data[["T1","T2"]] 
    
    x = appliance_energy_data["T1"] 
    y = appliance_energy_data["T2"] 
    
    fig, ax = plt.subplots()
    ax.fill_between(x, y, min(y), color='lightblue', label=r'$S^{(j,i)}(X)$', alpha=0.7)
    max_y = max(y)
    ax.fill_between(x, y, max_y, where=(y <= max_y), color='orange', label=r'$S^{(i,j)}(X)$', alpha=0.5)

    ax.plot(x, y, marker='o', color='black')
    ax.text(21.185, 19.9, r'$S^{(j)}(X)$', fontsize=16, color='black', verticalalignment='center', horizontalalignment='right')
    ax.text(20.65, 19.4, r'$S^{(i)}(X)$', fontsize=16, color='black', verticalalignment='center', horizontalalignment='right')
    ax.arrow(max(x),min(y), 0, max(y)-min(y)-0.1, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(min(x),min(y), max(x)-min(x)-0.1, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.set_xlabel(r'$X^i$')
    ax.set_ylabel(r'$X^j$')
    ax.grid(True)
    ax.set_xlim([min(x)-0.1, max(x)+0.1])
    ax.set_ylim([min(y)-0.1,max(y)+0.1])
    plt.legend(fontsize=16)
    plt.show()

def plot_MSEs(MSEs,d):
    data=MSEs
    labels = ['fPCR', 'BSpline', 'Fourier', "SLM"]
    sns.boxplot(data=data)
    plt.xticks([0, 1, 2, 3], labels)
    plt.title("d="+str(d))
    plt.ylabel('MSE')
    plt.show()
