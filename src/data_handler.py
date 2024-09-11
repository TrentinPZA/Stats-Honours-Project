import pandas as pd
import numpy as np


def import_air_qual():
    air_quality_data = pd.read_csv("AirQualityUCI.csv", sep=';', decimal=',', header=0) #Reading in the data, and specifying that the decimal points are commas so that they are instantly imported as floats.
    air_quality_data=air_quality_data.dropna(axis=0,how="all") #Dropping rows in the DF where all variables are NA

    #Keeping only the variable of interest
    air_quality_data=air_quality_data[["PT08.S4(NO2)","NO2(GT)"]]

    #Replacing -200 values which represent NaNs in UCI AirQuality with forward filled values (replaced with last known non-missing value in dataset)
    #If observation i is NaN it is replaced with observation i-1 (assuming i-1 holds an actual value).
    air_quality_data = air_quality_data.replace(to_replace=-200, value=None)
    air_quality_data = air_quality_data.fillna(method='ffill')

    #We want to use the past NO2 measurements at each hour of the past 7 days (past 168 hours) to predict the next hour. 
    #Therefore we need to create a set of lagged variables.
    window = 24*7
    x_unstacked=[]
    y_unstacked=[]

    #
    # range(0,(len(air_quality_data)-window))
    for i in range(0,(len(air_quality_data)-window)):
        x_window = air_quality_data["PT08.S4(NO2)"][i:window + i].to_numpy() #Use PT08.S4(NO2) readings to try predict the true N02 value which is given by NO2(GT).
        x_window = np.expand_dims(x_window, axis=1)
        x_unstacked.append(x_window) #Each Observation gets added to an array
        y_unstacked.append(air_quality_data["NO2(GT)"][window + i]) #Each target of the observation gets added to an array

    #Puts the arrays into a 3D array (think of using cbind to create a dataframe in R)
    #.shape gives d1 = number of observations,d2 = number of time points / data points in an observation, d3 = lagged variables
    X = np.stack(x_unstacked)
    Y = np.stack(y_unstacked)

    #Fermanian scales this and I'm not sure why but it seems to be the right thing to do as it makes MSE small
    Y =Y/100
    return(X,Y)