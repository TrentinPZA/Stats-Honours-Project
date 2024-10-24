import matplotlib.pyplot as plt
import pandas as pd
import joblib
from joblib import Parallel, delayed
import os.path
import tarfile
import random
import time
import urllib.request
from skfda.representation.grid import FDataGrid
import matplotlib.pyplot as plt
import numpy as np

import MNIST_funcs
import skfda

# Load the data from the file
file_path = 'mnist_data/sequences/testimg-3-points.txt'
data = pd.read_csv(file_path)
DATA_DIRECTORY = "mnist_data"
N_CPU = joblib.cpu_count()
train_points, train_inputs, train_targets = MNIST_funcs.mnist_train_data(
    data_directory=DATA_DIRECTORY, cpu_number=N_CPU
)
test_points, test_inputs, test_targets = MNIST_funcs.mnist_test_data(
    data_directory=DATA_DIRECTORY, cpu_number=N_CPU
)
index = 50145
# print(train_points[index])

print("The number is "+str(train_targets[index]))

data = pd.DataFrame(train_points[index],columns=['x','y'])

# data=data[:-1]
# print(data)
length = len(data)
sequence = list(range(1, length + 1))
seq = np.linspace(0,1,length)
fig, ax = plt.subplots()
    # sns.set_style("darkgrid")
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.subplots_adjust(left=0.5, bottom=0.15)

# fdg = FDataGrid(data['y'],seq)
# fdg.plot(color="red")

# Plot the points

# plt.scatter(data['x'], -1*data['y'], color="green",s=125)
plt.scatter(seq, data['y'], color="red")
plt.xlabel('t', fontsize=32)
plt.ylabel('Y(t)', fontsize=32)
# plt.gca().set_aspect()



plt.tick_params(axis='both', which='major', labelsize=30)
# plt.savefig('mnistY.png', bbox_inches='tight')
plt.show()





# colour_arr = ["orange","blue","red","yellow","green","magenta","black","cyan"]
# for i in range(0,3000):
#     if train_targets[i]==9:
        
#         data = pd.DataFrame(train_points[i],columns=['x','y'])
#         filtered_data = data[(data['x'] >= 0) & (-1 * data['y'] <= 0)]
#         randy= random.choice(colour_arr)
#         plt.plot(filtered_data['x'], -1 * filtered_data['y'], color=randy, alpha=0.15)
#         # plt.plot(data['x'],-1*data['y'], color=randy,alpha=0.15)
#         # Adding labels and title
# plt.xlabel('X(t)')
# plt.ylabel('Y(t)')
# plt.xlim(left=0)
# plt.ylim(top=0)
# plt.xlabel('X(t)', fontsize=32)
# plt.ylabel('Y(t)', fontsize=32)
# plt.tick_params(axis='both', which='major', labelsize=30)


# # Show the plot
# plt.grid(True)
# plt.show()      
  



