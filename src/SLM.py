# import pandas as pd
# import numpy as np
# import skfda
# from skfda.representation.grid import FDataGrid
# from skfda.representation.basis import Fourier,VectorValued,Monomial,BSpline
# from skfda.ml.regression import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
# from skfda.preprocessing.dim_reduction.projection import FPCA
# from data_handler import import_air_qual
# from sklearn.linear_model import RidgeCV, Ridge
# import iisignature as isig



# def add_time_dimension(X):
#     times = np.tile(np.linspace(0, 1, X.shape[1]), (X.shape[0], 1))#np.linspace makes sequence of N points from a to b (0 to 1 in this case) 
#                                                                 #np.tile repeats an array along an axis, in this case we repeat it 9189 times along the first axis
#     times = np.expand_dims(times, axis=2) # Add an extra dimension
#     Xtime = np.concatenate([X, times], axis=2) #Concatenates to the existing data, adding the time component to each data point in each observation.
#     return Xtime



# #What Ferm did
#                                                                             #+1 because siglenth gives length without 0th term.
# # sigX = np.zeros((np.shape(Xtime)[0], isig.siglength(dimension, trunc_order) + 1)) #Setting up the signatures for each observation, therefore np.shape(Xtime)[0]=9189 and 
# #                                                                                   #isig.siglength(dimension, trunc_order) + 1 = 1 +2^1+2^2+2^3   (dim^order) and trunc order =3
# # sigX[:,0] = 1


# def calculate_sig(X):
#     trunc_order = 4 #NEED ALGORITHM TO FIND OPTIMAL TRUNC ORDER 
#     arr_sig=[]
#     for i in range(np.shape(X)[0]):			
#                 arr_sig.append(np.insert(isig.sig(X[i],trunc_order),0,1)) #Calculates the signature and adds the zeroeth term in the front to make it S=[1,4,14,13] for example
#     the_sig = np.stack(arr_sig) #Stack them to make an "dataframe" like in R 	
#     return the_sig		

# X,Y = import_air_qual(multivariate=True)
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

# Xtrain = add_time_dimension(Xtrain)
# Xtest = add_time_dimension(Xtest)
# Ridge_reg = Ridge(normalize=False, fit_intercept=False, solver='svd') #Create a ridge regression object

# train_sig = calculate_sig(Xtrain)
# test_sig = calculate_sig(Xtest)

# alphas = np.linspace(10 ** (-5), 1000, num=100) #In scikit-learn the lambda values are referred to as alpha


# reg_cv = RidgeCV(alphas=alphas, store_cv_values=True, fit_intercept=False, gcv_mode='svd') #Create a RidgeCV object
# reg_cv.fit(train_sig, Ytrain)#Fit the RidgeCV to determine the best regularisation strength
# alpha = reg_cv.alpha_ #Extract the best regularisation strength
# Ridge_reg.alpha=alpha #Assign best regularisation strength to the Ridge regression object we created earlier
# Ridge_reg.fit(train_sig,Ytrain) #Fit Ridge regression using the best regularisation strength

# preds = Ridge_reg.predict(test_sig) #Make predictions using test set

# # print(np.mean((preds-Ytest) ** 2))