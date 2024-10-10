import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_handler import *
from RegressionModel import *
from utils import *
from sklearn.model_selection import train_test_split
from skfda.ml.regression import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import math

def main():
    
    if sys.argv[1]=="AirQuality":
         
         Kpen=0.1
         if sys.argv[2]=="True":
            X,Y = import_air_qual(multivariate=True)
         else:
            X,Y = import_air_qual(multivariate=False)
         
    elif sys.argv[1]=="ApplianceEnergy":
        Kpen=0.5
        if sys.argv[2]=="True":
            X,Y = import_energy_data(multivariate=True,dim=sys.argv[3])
        else:
            X,Y = import_energy_data(multivariate=False,dim="")

            

    
    models = ["fPCA","BSpline","Fourier"]
    opt_nbasis=[]
    for model in models:
        opt_nbasis.append(nbasis_cross_validation(basis_exp_type=model,X=X,Y=Y))
    
    mses = [[] for _ in range(4)] 
    
    for i, model_type in enumerate(models):
        for j in range(20):  
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33,random_state=j)
            model_ = BasisExpansionFLR(basis_exp_type=model_type, nbasis=opt_nbasis[i])
            model_.fit(X_train, Y_train)
            mse = model_.evaluate(X_test, Y_test)
            mses[i].append(mse) 

    
    for k in range(20):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33,random_state=k+10)
        TO_selector=SignatureOrderSelection(X_train.shape[2]+1,Kpen=Kpen)
        trunc_order = TO_selector.get_hatm(X_train,Y_train, Kpen_values=np.e ** np.linspace(-3, 2, num=200))
        print(trunc_order)
        model = SignatureLinearModel(trunc_order)
        model.fit(X_train,Y_train)
        mses[3].append(model.evaluate(X_test,Y_test))
    
    MSEs = [mses[0], mses[1], mses[2], mses[3]]
    plot_MSEs(MSEs=MSEs,d=X.shape[2])
    
    
    



if __name__ == "__main__":
    main()