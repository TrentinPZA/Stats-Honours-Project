import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src import data_handler, RegressionModel,utils
from data_handler import *
from RegressionModel import *
from utils import *
from sklearn.model_selection import train_test_split
from skfda.ml.regression import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    

    X,Y = import_air_qual(multivariate=True)
    # geom_interp()
       
    models = ["fPCA","BSpline","Fourier"]
    opt_nbasis=[]
    for model in models:
        opt_nbasis.append(nbasis_cross_validation(basis_exp_type=model,X=X,Y=Y))
    
    mses = [[] for _ in range(4)] 
    
    for i, model_type in enumerate(models):
        for j in range(20):  
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
            model_ = BasisExpansionFLR(basis_exp_type=model_type, nbasis=opt_nbasis[i])
            model_.fit(X_train, Y_train)
            mse = model_.evaluate(X_test, Y_test)
            mses[i].append(mse) 

    
    for k in range(20):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
        model = SignatureLinearModel()
        model.fit(X_train,Y_train)
        mses[3].append(model.evaluate(X_test,Y_test))
    
    data = [mses[0], mses[1], mses[2], mses[3]]
    labels = ['fPCR', 'BSpline', 'Fourier', "SLM"]
    sns.boxplot(data=data)
    plt.xticks([0, 1, 2, 3], labels)
    plt.title('Comparison of MSEs of different models (FLR and SLM)')
    plt.ylabel('MSE')
    plt.show()
    

    
    



if __name__ == "__main__":
    main()