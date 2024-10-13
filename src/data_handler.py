import pandas as pd
import numpy as np


def import_air_qual(multivariate=False):
    air_quality_data = pd.read_csv("data/AirQualityUCI.csv", sep=';', decimal=',', header=0) #Reading in the data, and specifying that the decimal points are commas so that they are instantly imported as floats.
    air_quality_data=air_quality_data.dropna(axis=0,how="all") #Dropping rows in the DF where all variables are NA
    #Keeping only the variable of interest
    air_quality_data=air_quality_data[["PT08.S4(NO2)","T","RH","NO2(GT)"]] 

    #Replacing -200 values which represent NaNs in UCI AirQuality with forward filled values (replaced with last known non-missing value in dataset)
    #If observation i is NaN it is replaced with observation i-1 (assuming i-1 holds an actual value).
    air_quality_data = air_quality_data.replace(to_replace=-200, value=None)
    air_quality_data = air_quality_data.fillna(method='ffill')

    #We want to use the past NO2 measurements at each hour of the past 7 days (past 168 hours) to predict the next hour. 
    #Therefore we need to create a set of lagged variables.
    window = 24*7
    x_unstacked=[]
    y_unstacked=[]

    # range(0,(len(air_quality_data)-window)) , This is how many windows of a specific size can be 'made' in a dataset of a specific length.
    for i in range(0,(len(air_quality_data)-window)):
        if multivariate==False:
            x_window = air_quality_data['PT08.S4(NO2)'][i:window + i].to_numpy() #Use PT08.S4(NO2) readings to try predict the true N02 value which is given by NO2(GT).
            x_window = np.expand_dims(x_window, axis=1)
        else:
            x_window = air_quality_data[['PT08.S4(NO2)', 'T', 'RH']][i:window + i].to_numpy()  #To make it multivariate include T=Temperature and RH = Relative Humidity

        x_unstacked.append(x_window) #Each Observation gets added to an array
        y_unstacked.append(air_quality_data["NO2(GT)"][window + i]) #Each target of the observation gets added to an array

    #Puts the arrays into a 3D array (think of using cbind to create a dataframe in R)
    #.shape gives d1 = number of observations,d2 = number of time points / data points in an observation, d3 = dimension
    X = np.stack(x_unstacked)
    Y = np.stack(y_unstacked)

    #Fermanian scales this and I'm not sure why but it seems to be the right thing to do as it makes MSE small
    Y =Y/100
    return(X,Y)

def import_energy_data(dim,multivariate=False):
    appliance_energy_data = pd.read_csv("data/energydata_complete.csv", sep=',',decimal=".",header=0) #Reading in the data, and specifying that the decimal points are commas so that they are instantly imported as floats.
    half_length = len(appliance_energy_data) // 2
    appliance_energy_data = appliance_energy_data.iloc[:half_length]# Keep only the first half of the rows
    appliance_energy_data=appliance_energy_data.dropna(axis=0,how="all")
    
    appliance_energy_data=appliance_energy_data[["Appliances","T1","T2","T3","T4","T5","T6","T7","T8","T9","T_out","Windspeed"]] 
    
    window = 1440//10 #Using the past 24 hours to predict the next 10 minutes, must be integer division for For loop
    x_unstacked=[]
    y_unstacked=[]

    # range(0,(len(appliance_energy_data)-window)) , This is how many windows of a specific size can be 'made' in a dataset of a specific length.
    for i in range(0,(len(appliance_energy_data)-window)):
        if multivariate==False:
            x_window = appliance_energy_data['T_out'][i:window + i].to_numpy() 
            x_window = np.expand_dims(x_window, axis=1)
        else:
            if dim =="8":
                x_window = appliance_energy_data[["T1","T2","T3","T4","T5","T8","T9","T_out"]][i:window + i].to_numpy()  
            elif dim =="9":
                x_window = appliance_energy_data[["T1","T2","T3","T4","T5","T7","T8","T9","T_out"]][i:window + i].to_numpy()  
            elif dim =="11":
                x_window = appliance_energy_data[["T1","T2","T3","T4","T5","T6","T7","T8","T9","T_out","Windspeed"]][i:window + i].to_numpy()  

        x_unstacked.append(x_window) #Each Observation gets added to an array
        y_unstacked.append(appliance_energy_data["Appliances"][window + i]) #Each target of the observation gets added to an array

    #Puts the arrays into a 3D array (think of using cbind to create a dataframe in R)
    #.shape gives d1 = number of observations,d2 = number of time points / data points in an observation, d3 = dimension
    X = np.stack(x_unstacked)
    Y = np.stack(y_unstacked)
    Y =Y/100
    print(X.shape)
    return(X,Y)
    