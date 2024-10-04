import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV, Ridge
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import Fourier, BSpline, VectorValued
from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.ml.regression import LinearRegression
import iisignature as isig


class RegressionModel:
    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        raise NotImplementedError("Method must be implemented by child classes")

    def predict(self, X):
        raise NotImplementedError("Method must be implemented by child classes")
    
    def evaluate(self, X, Y):
        preds = self.predict(X)
        mse = np.mean((preds - Y) ** 2)
        return mse
    
class BasisExpansionFLR(RegressionModel):
    def __init__(self,basis_exp_type = "Fourier",nbasis=10): #Set default parameters, Fourier and nbasis = 10
        super().__init__()
        self.basis_exp_type = basis_exp_type
        self.nbasis=nbasis
        self.basis_vec_valued = None
        self.model = LinearRegression()

    def create_basis_vec_valued(self,X):
        if self.basis_exp_type == 'Fourier':
            basis_vectors = [Fourier(n_basis=self.nbasis) for _ in range(X.shape[2])] 
        else:
            basis_vectors = [BSpline(n_basis=self.nbasis) for _ in range(X.shape[2])] #At the moment FPCA uses BSPLINE expansion as well
        self.basis_vec_valued = VectorValued(basis_vectors)

    def apply_basis_exp(self,X):
        dim_points = np.linspace(0,1,X.shape[1])
        X_fdg=FDataGrid(X,dim_points)
        X_bexp = X_fdg.to_basis(self.basis_vec_valued)
        if self.basis_exp_type=="fPCA":
           fpca_basis = FPCA(5)
           fpca_basis = fpca_basis.fit(X_bexp)
           X_bexp = fpca_basis.transform(X_bexp)

        return (X_bexp)
    
    def fit(self,X,Y):
        
        self.create_basis_vec_valued(X)
        X_bexp = self.apply_basis_exp(X)
        self.model.fit(X_bexp,Y)

    def predict(self, X):
        return self.model.predict(self.apply_basis_exp(X))
    
def nbasis_cross_validation(basis_exp_type,X,Y):
    if basis_exp_type == "Fourier" or basis_exp_type == "BSpline":
        basis_nums = [8,9,10,11,12,13]
    elif basis_exp_type == "fPCA":
        basis_nums = [7]

    basis_cv_mses = []

    for num_basis in basis_nums:
        kfold_cv = KFold(n_splits=5)
        current_mse = []

        for train_indices,test_indices in kfold_cv.split(X):
           BRegModel = BasisExpansionFLR(basis_exp_type=basis_exp_type,nbasis=num_basis) 
           BRegModel.fit(X[train_indices],Y[train_indices])
           current_mse.append(BRegModel.evaluate(X[test_indices],Y[test_indices]))  
        basis_cv_mses.append(np.mean(current_mse))

    return(basis_nums[np.argmin(basis_cv_mses)])  


class SignatureLinearModel(RegressionModel):
    def __init__(self): #Set default parameters, Fourier and nbasis = 10
        super().__init__()
        self.model = Ridge()
    
    def add_time_dimension(self,X):
        times = np.tile(np.linspace(0, 1, X.shape[1]), (X.shape[0], 1))#np.linspace makes sequence of N points from a to b (0 to 1 in this case) 
                                                                    #np.tile repeats an array along an axis, in this case we repeat it 9189 times along the first axis
        times = np.expand_dims(times, axis=2) # Add an extra dimension
        Xtime = np.concatenate([X, times], axis=2) #Concatenates to the existing data, adding the time component to each data point in each observation.
        return Xtime

    def calculate_sig(self,X):
        trunc_order = 4 #NEED ALGORITHM TO FIND OPTIMAL TRUNC ORDER 
        arr_sig=[]
        for i in range(np.shape(X)[0]):			
                    arr_sig.append(np.insert(isig.sig(X[i],trunc_order),0,1)) #Calculates the signature and adds the zeroeth term in the front to make it S=[1,4,14,13] for example
        the_sig = np.stack(arr_sig) #Stack them to make an "dataframe" like in R 	
        return the_sig	 

    def fit(self,X,Y):
        Xtime = self.add_time_dimension(X)     
        the_sig = self.calculate_sig(Xtime)

        
        # MIGHT NEED TO ADJUST ALPHAS BASED ON PERFORMANCE ON OTHER DATASET
        alphas = np.linspace(10 ** (-5), 1000, num=100) #In scikit-learn the lambda values are referred to as alpha  
        reg_cv = RidgeCV(alphas=alphas, store_cv_values=True, fit_intercept=False, gcv_mode='svd') #Create a RidgeCV object
        reg_cv.fit(the_sig, Y)#Fit the RidgeCV to determine the best regularisation strength
        alpha = reg_cv.alpha_ #Extract the best regularisation strength
        self.model.alpha=alpha #Assign best regularisation strength to the Ridge regression object we created earlier
        self.model.fit(the_sig,Y) #Fit Ridge regression using the best regularisation strength
    
    def predict(self, X):
        X = self.calculate_sig(self.add_time_dimension(X))
        return(self.model.predict(X))
        