import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def geom_interp(): #Code that produces the plot displayed in the geometric interpretation
    appliance_energy_data = pd.read_csv("data/energydata_complete.csv", sep=',',decimal=".",header=0)   
    appliance_energy_data = appliance_energy_data.iloc[10:26]
    appliance_energy_data=appliance_energy_data[["T1","T2"]] 
    
    x = appliance_energy_data["T1"] 
    y = appliance_energy_data["T2"] 
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.5, bottom=0.15)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
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
    plt.legend(fontsize=30)
    plt.xlabel('X(t)', fontsize=32)
    plt.ylabel('Y(t)', fontsize=32)
    plt.tick_params(axis='both', which='major', labelsize=30)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')  # Set to black color
        spine.set_linewidth(1)
    plt.show()

def plot_MSEs(MSEs,d): #Create the boxplots for the different models and their test MSEs
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.subplots_adjust(left=0.5, bottom=0.15)
    data=MSEs
    labels = ['fPCR', 'BSpline', 'Fourier', "SLM"]
    sns.boxplot(data=data)
    plt.xticks([0, 1, 2, 3], labels)
    
    plt.title("d="+str(d),fontsize=32)
    plt.ylabel('MSE',fontsize=32)
    plt.tick_params(axis='both', which='major', labelsize=30)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black') 
        spine.set_linewidth(1)
    plt.show()

def plot_coeff_heatmap(d, model):#Plotting the regression coefficients that correspond to signature terms
    k = 2
    sig_size = int((d**(k+1)-1)/(d-1))
    sig_reg_coeff = np.concatenate([[model.model.intercept_], model.model.coef_])
    coefficients = sig_reg_coeff[:sig_size]
    coefficients = coefficients.reshape(1, -1)
    
    if d > 5:
        mask = np.abs(coefficients) >= 0.35
    else:
        mask = np.abs(coefficients) >= 0.0025
        
    
    coefficients = coefficients[mask]
    coefficients = coefficients.reshape(1, -1)
    
    custom_labels = ['()']
    for i in range(1, k+1):
        for item in itertools.product(range(1, d+1), repeat=i):
            custom_labels.append(str(item))
            
    custom_labels = np.array(custom_labels)
    custom_labels = custom_labels[mask[0]]
    
  
    plt.figure(figsize=(20, 2)) 
    heatmap=sns.heatmap(coefficients, annot=np.array([custom_labels]), center=0, 
                cbar_kws={"orientation": "horizontal"}, fmt='', xticklabels=False, yticklabels=False,annot_kws={"size": 22})
    
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=22)
    
    plt.show()
    
