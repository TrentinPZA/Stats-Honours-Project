
import pandas as pd
import numpy as np
import skfda
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import Fourier,VectorValued,Monomial,BSpline
from skfda.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

#Splits the data into training/test data
Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.33)

dim_points = np.linspace(0,1,Xtrain.shape[1]) #Create a sequence of Xtrain.shape[1] points from 0 - 1.
Xtrain_fdg = FDataGrid(Xtrain,dim_points)
Xtest_fdg = FDataGrid(Xval,dim_points)

#print(Xtrain_fdg)
# Xtrain_fdg["dataset"].plot()
# plt.show



basis_vectors = [Fourier(n_basis=12) for _ in range(X.shape[2])] #Specifying that each dimension of the functional data will be expanded using a Fourier expansion
fourier_basis = VectorValued(basis_vectors) #Creating the Vector-Valued basis expansion object

#Performing Fourier basis expansion
Xtrain__fourier_basis = Xtrain_fdg.to_basis(fourier_basis) 
Xtest__fourier_basis = Xtest_fdg.to_basis(fourier_basis)
# fig_fourier = Xtrain__fourier_basis.plot()

#Plotting the functional data using the basis expansions
# Xtrain_fdg.plot() #This just plots the data without basis expansions, as is
# plt.show()

#Fitting a Linear SoFR using Fourier Basis Expansion
reg_model = LinearRegression()
reg_model.fit(Xtrain__fourier_basis,Ytrain)

#Getting predictions using the Linear SoFR model and calculate the MSE
preds=reg_model.predict(Xtest__fourier_basis)
print(np.mean((preds-Yval) ** 2))









