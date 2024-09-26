
import pandas as pd
import numpy as np
import skfda
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import Fourier,VectorValued,Monomial,BSpline
from skfda.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from skfda.preprocessing.dim_reduction.projection import FPCA
from data_handler import import_air_qual


def fitRegression(X,Y,type):


    #Splits the data into training/test data
    Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.33)
                #np.linspace just used to create sequences like in R
    dim_points = np.linspace(0,1,Xtrain.shape[1]) #Create a sequence of Xtrain.shape[1] points from 0 - 1. (If 1 observation has 168 data points then the sequence will be 168 entries long)
    Xtrain_fdg = FDataGrid(Xtrain,dim_points)
    Xtest_fdg = FDataGrid(Xval,dim_points)

    #print(Xtrain_fdg)
    # Xtrain_fdg["dataset"].plot()
    # plt.show
    if type == "Fourier" or type == "BSpline":
        basis_nums = [8,9,10,11,12,13] #Fermanian did less than this but higher orders perform better.
        cv_reg_model = LinearRegression()
        basis_cv_mses = []
        for num_basis in basis_nums:
            kfold_cv = KFold(n_splits=5)
            current_mse = []
            for train_indices,test_indices in kfold_cv.split(X):
                if type == "Fourier":
                    basis_vectors = [Fourier(n_basis=num_basis) for _ in range(X.shape[2])] #Specifying that each dimension of the functional data will be expanded using a Fourier expansion
                else:
                    basis_vectors = [BSpline(n_basis=num_basis) for _ in range(X.shape[2])] #Specifying that each dimension of the functional data will be expanded using a BSpline expansion

                fourier_basis = VectorValued(basis_vectors) #Creating the Vector-Valued basis expansion object

                Xtrain_CV,Ytrain_CV = X[train_indices],Y[train_indices]
                Xtest_CV,Ytest_CV = X[test_indices],Y[test_indices]

                Xtrain_CV_fdg = FDataGrid(Xtrain_CV,dim_points)
                Xtest_CV_fdg = FDataGrid(Xtest_CV,dim_points)

                Xtrain_CV_fourier_basis = Xtrain_CV_fdg.to_basis(fourier_basis) 
                Xtest_CV_fourier_basis = Xtest_CV_fdg.to_basis(fourier_basis)
                
                cv_reg_model.fit(Xtrain_CV_fourier_basis,Ytrain_CV)
                cv_preds = cv_reg_model.predict(Xtest_CV_fourier_basis)
                current_mse.append(np.mean((cv_preds-Ytest_CV) ** 2))

            basis_cv_mses.append(np.mean(current_mse))

        optimal_basis_num=basis_nums[np.argmin(basis_cv_mses)]    
        #print(optimal_basis_num)
    else:
        optimal_basis_num=7







    #Fitting a Linear SoFR using Fourier Basis Expansion with the optimal number of basis functions
    if type == "Fourier":
        basis_vectors = [Fourier(n_basis=optimal_basis_num) for _ in range(X.shape[2])] #Specifying that each dimension of the functional data will be expanded using a Fourier expansion
    elif type == "BSpline":
        basis_vectors = [BSpline(n_basis=optimal_basis_num) for _ in range(X.shape[2])] #Specifying that each dimension of the functional data will be expanded using a BSpline expansion
    else:
        basis_vectors = [BSpline(n_basis=optimal_basis_num) for _ in range(X.shape[2])] #Specifying that each dimension of the functional data will be expanded using a BSpline expansion

    fourier_basis = VectorValued(basis_vectors)

    Xtrain_fourier_basis=Xtrain_fdg.to_basis(fourier_basis)
    Xtest_fourier_basis=Xtest_fdg.to_basis(fourier_basis)
    if type == "fPCR":
        fpca_basis = FPCA(5)
        train_fpca_basis = fpca_basis.fit(Xtrain_fourier_basis)
        test_fpca_basis = fpca_basis.fit(Xtest_fourier_basis)
        Xtrain_fourier_basis = train_fpca_basis.transform(Xtrain_fourier_basis)
        Xtest_fourier_basis = test_fpca_basis.transform(Xtest_fourier_basis)


    reg_model = LinearRegression()
    reg_model.fit(Xtrain_fourier_basis,Ytrain)

    #Getting predictions using the Linear SoFR model and calculate the MSE
    preds=reg_model.predict(Xtest_fourier_basis)
    return(np.mean((preds-Yval) ** 2))


X,Y = import_air_qual(multivariate=False)

FourierMSE = fitRegression(X,Y,"Fourier")
BSplineMSE = fitRegression(X,Y,"BSpline")
fPCRMSE = fitRegression(X,Y,"fPCR")

print(FourierMSE)
print(BSplineMSE)
print(fPCRMSE)

# FourierMSEs = np.array([fitRegression(X, Y, "Fourier") for _ in range(20)])
# BSplineMSEs = np.array([fitRegression(X, Y, "BSpline") for _ in range(20)])
# fPCRMSEs = np.array([fitRegression(X, Y, "fPCR") for _ in range(20)])

# plt.boxplot([fPCRMSEs, BSplineMSEs, FourierMSEs])
# plt.xticks([1, 2, 3], ['fPCR', 'BSpline', 'Fourier'])
# plt.title('Comparison of MSEs from FLRs using different Basis Expansions.')
# plt.ylabel('MSE')
# plt.show()





