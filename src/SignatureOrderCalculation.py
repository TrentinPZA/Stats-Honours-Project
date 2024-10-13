
from RegressionModel import SignatureLinearModel
import seaborn as sns
import math
import iisignature as isig
import numpy as np
import matplotlib as plt

#Author: Adeline Fermanian
#Year: 2022
#This class implements the signature truncation order estimation method, proposed by Fermanian
#Minor changes were made to ensure compatibility between this class and other classes. All credit for the creation of this class to Adeline Fermanian.

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
            Kpen = float(input("Enter slope heuristic constant Kpen: "))
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

        if plot:
            palette = sns.color_palette('colorblind')
            plt.plot(np.arange(self.max_k + 1), loss, label=r"$\widehat{L}_n(m)$", color=palette[0])
            plt.plot(np.arange(self.max_k + 1), pen, label=r"$pen_n(m)$", color=palette[1], linestyle='dashed')
            plt.plot(np.arange(self.max_k + 1), objective, label=r"$\widehat{L}_n(m) + pen_n(m)$",
                     color=palette[2], linestyle='dotted')
            plt.legend()
            plt.xlabel(r'$m$')
        
        return hatm