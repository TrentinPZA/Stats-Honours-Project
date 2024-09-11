
import pandas as pd
import numpy as np
import skfda
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import Fourier,VectorValued,Monomial,BSpline
from skfda.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from data_handler import import_air_qual

X,Y = import_air_qual()

#Splits the data into training/test data
Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.33)

dim_points = np.linspace(0,1,Xtrain.shape[1]) #Create a sequence of Xtrain.shape[1] points from 0 - 1.
Xtrain_fdg = FDataGrid(Xtrain,dim_points)
Xtest_fdg = FDataGrid(Xval,dim_points)

#print(Xtrain_fdg)
# Xtrain_fdg["dataset"].plot()
# plt.show

basis_nums = [8,9,10,11,12,13,14,15,16]
cv_reg_model = LinearRegression()
basis_cv_mses = []
for num_basis in basis_nums:
    kfold_cv = KFold(n_splits=5)
    current_mse = []
    for train,test in kfold_cv.split(X):
        basis_vectors = [Fourier(n_basis=num_basis) for _ in range(X.shape[2])] #Specifying that each dimension of the functional data will be expanded using a Fourier expansion
        fourier_basis = VectorValued(basis_vectors) #Creating the Vector-Valued basis expansion object

        Xtrain_CV,Ytrain_CV = X[train],Y[train]
        Xtest_CV,Ytest_CV = X[test],Y[test]

        Xtrain_CV_fdg = FDataGrid(Xtrain_CV,dim_points)
        Xtest_CV_fdg = FDataGrid(Xtest_CV,dim_points)

        Xtrain_CV_fourier_basis = Xtrain_CV_fdg.to_basis(fourier_basis) 
        Xtest_CV_fourier_basis = Xtest_CV_fdg.to_basis(fourier_basis)
        
        cv_reg_model.fit(Xtrain_CV_fourier_basis,Ytrain_CV)
        cv_preds = cv_reg_model.predict(Xtest_CV_fourier_basis)
        current_mse.append(np.mean((cv_preds-Ytest_CV) ** 2))
    basis_cv_mses.append(np.mean(current_mse))

optimal_basis_num=basis_nums[np.argmin(basis_cv_mses)]    
print(optimal_basis_num)

#Fitting a Linear SoFR using Fourier Basis Expansion with the optimal number of basis functions

basis_vectors = [Fourier(n_basis=optimal_basis_num) for _ in range(X.shape[2])] #Specifying that each dimension of the functional data will be expanded using a Fourier expansion
fourier_basis = VectorValued(basis_vectors)

Xtrain_fourier_basis=Xtrain_fdg.to_basis(fourier_basis)
Xtest_fourier_basis=Xtest_fdg.to_basis(fourier_basis)

reg_model = LinearRegression()
reg_model.fit(Xtrain_fourier_basis,Ytrain)

#Getting predictions using the Linear SoFR model and calculate the MSE
preds=reg_model.predict(Xtest_fourier_basis)
print(np.mean((preds-Yval) ** 2))











