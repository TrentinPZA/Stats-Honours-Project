import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV, Ridge
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import Fourier, BSpline, VectorValued
from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.ml.regression import LinearRegression
import math
import iisignature as isig
import seaborn as sns
import matplotlib.pyplot as plt

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
        elif self.basis_exp_type=='BSpline':
            basis_vectors = [BSpline(n_basis=self.nbasis) for _ in range(X.shape[2])]
        elif self.basis_exp_type=="fPCA":
            basis_vectors = [BSpline(n_basis=7) for _ in range(X.shape[2])] # Use BSplines to smooth the functional covariates in order to eliminate some of the noise and try extract meaningful patterns
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
        basis_nums = [4,5,6,7,8,9,10,11,12,13]
    elif basis_exp_type == "fPCA":
        basis_nums = [1,2,3,4,5]

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
    def __init__(self,m): #Set default parameters, Fourier and nbasis = 10
        super().__init__()
        self.model = Ridge(normalize=False, fit_intercept=False, solver='svd')
        self.m=m
        
    
    def add_time_dimension(self,X):
        times = np.tile(np.linspace(0, 1, X.shape[1]), (X.shape[0], 1))#np.linspace makes sequence of N points from a to b (0 to 1 in this case) 
                                                                    #np.tile repeats an array along an axis, in this case we repeat it 9189 times along the first axis
        times = np.expand_dims(times, axis=2) # Add an extra dimension
        Xtime = np.concatenate([X, times], axis=2) #Concatenates to the existing data, adding the time component to each data point in each observation.
        return Xtime

    def calculate_sig(self,X):
        trunc_order = self.m #NEED ALGORITHM TO FIND OPTIMAL TRUNC ORDER 
        arr_sig=[]
    
        if trunc_order == 0:
            the_sig=np.full((np.shape(X)[0],1),1)
        else:
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
        self.model.alpha_=alpha #Assign best regularisation strength to the Ridge regression object we created earlier
        self.model.fit(the_sig,Y) #Fit Ridge regression using the best regularisation strength
    
    def predict(self, X):
        X = self.calculate_sig(self.add_time_dimension(X))
        return(self.model.predict(X))
        
class SignatureOrderSelection(object):
    """Estimation of the order of truncation of the signature

    Parameters
    ----------
    d: int
        Dimension of the space of the paths X, from which an output Y is learned.

    rho: float
        Parameter of the penalization: power of 1/n. It should satisfy : 0<rho<1/2.

    Kpen: float, default=none
        Constant in front of the penalization, it has to be a positive number.

    alpha: float, default=None
        Regularization parameter in the Ridge regression.

    max_features: int,
        Maximal size of coefficients considered.

    Attributes
    ----------
    max_k: int,
        Maximal value of signature truncation to keep the number of features below max_features.

    """
    def __init__(self, d, rho=0.4, Kpen=None, alpha=None, max_features=10 ** 3):
        self.d = d
        self.rho = rho
        self.Kpen = Kpen
        self.alpha = alpha

        self.max_features = max_features
        self.max_k = math.floor((math.log(self.max_features * (d - 1) + 1) / math.log(d)) - 1)

    def fit_alpha(self, X, Y):
        """ Find alpha by cross validation with signatures truncated at order 1.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        Returns
        -------
            alpha: float
                Regularization parameter
        """
        sigreg = SignatureLinearModel(1)
        sigreg.fit(X, Y)
        self.alpha = sigreg.model.alpha_
        # print(sigreg.model.alpha_)
        return self.alpha

    def get_penalization(self, n, k, Kpen):
        """Returns the penalization function used in the estimator of the truncation order, that is,
        pen_n(k)=Kpen*sqrt((d^(k+1)-1)/(d-1))/n^rho.

        Parameters
        ----------
        n: int
            Number of samples.

        k: int
            Truncation order of the signature.

        Kpen: float, default=1
            Constant in front of the penalization, it has to be strictly positive.

        Returns
        -------
        pen_n(k):float
            The penalization pen_n(k)
        """
        if k == 0:
            size_sig = 1
        else:
            size_sig = isig.siglength(self.d, k) + 1

        return Kpen * n ** (-self.rho) * math.sqrt(size_sig)

    def slope_heuristic(self, X, Y, Kpen_values, savefig=False):
        """Implements the slope heuristic to select a value for Kpen, the
        unknown constant in front of the penalization. To this end, hatm is
        computed for several values of Kpen, and these values are then plotted
        against Kpen.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        Kpen_values: array, shape (n_K)
            An array of potential values for Kpen. It must be calibrated so that any value of hatm between 1 and max_k
            is obtained with values of Kpen in Kpen_values.

        savefig: boolean, default=False
            If True, saves the slope heuristics plot of hatm against Kpen.

        Returns
        -------
        hatm: array, shape (n_K)
            The estimator hatm obtained for each value of Kpen in Kpen_values.
        """
        if self.alpha is None:
            self.fit_alpha(X, Y)
        hatm = np.zeros(len(Kpen_values))
        loss = np.zeros(self.max_k + 1)
        for j in range(self.max_k + 1):
            sigReg = SignatureLinearModel(j)
            sigReg.fit(X, Y)
            loss[j] = sigReg.evaluate(X, Y)

        for i in range(len(Kpen_values)):
            # print(i)
            pen = np.zeros(self.max_k + 1)
            for j in range(self.max_k + 1):
                pen[j] = self.get_penalization(Y.shape[0], j, Kpen_values[i])
            hatm[i] = np.argmin(loss + pen)

        palette = sns.color_palette('colorblind')
        fig, ax = plt.subplots()
        jump = 1
        for i in range(self.max_k + 1):
            if i in hatm:
                xmin = Kpen_values[hatm == i][0]
                xmax = Kpen_values[hatm == i][-1]
                ax.hlines(i, xmin, xmax, colors='b')
                if i != 0:
                    ax.vlines(xmax, i, i - jump, linestyles='dashed', colors=palette[0])
                jump = 1
            else:
                jump += 1
        ax.set(xlabel=r'$K_{pen}$', ylabel=r'$\hat{m}$')
        plt.show()
        return hatm

    def get_hatm(self, X, Y, Kpen_values=np.linspace(10 ** (-6), 1, num=200), plot=False, savefig=False):
        """Computes the estimator of the truncation order by minimizing the sum of hatL and the penalization, over
        values of k from 1 to max_k.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        Kpen_values: array, shape (n_K)
            An array of potential values for Kpen. It must be calibrated so that any value of hatm between 1 and max_k
            is obtained with values of Kpen in Kpen_values.

        plot: boolean, default=False
            If True, plots the functions k->hatL(k), k->pen(k) and k-> hatL
            (k)+pen(k). The latter is minimized at hatm.

        savefig: boolean, default=False
            If True, saves the slope heuristics plot of hatm against Kpen.

        Returns
        -------
        hatm: int
            The estimator of the truncation order
        """
        if not self.Kpen:
            self.slope_heuristic(X, Y, Kpen_values, savefig=savefig)
            Kpen = 1 #Chosen based off Fermys work
        else:
            Kpen = self.Kpen
        objective = np.zeros(self.max_k + 1)
        loss = np.zeros(self.max_k + 1)
        pen = np.zeros(self.max_k + 1)
        for i in range(self.max_k + 1):
            sigReg = SignatureLinearModel(i)
            sigReg.fit(X, Y)
            loss[i] = sigReg.evaluate(X, Y)
            pen[i] = self.get_penalization(Y.shape[0], i, Kpen)
            objective[i] = loss[i] + pen[i]
        hatm = np.argmin(objective)

        
        return hatm