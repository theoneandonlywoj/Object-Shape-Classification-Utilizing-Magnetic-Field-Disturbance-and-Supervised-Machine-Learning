import tflearn
import time
import numpy as np
from tqdm import tqdm

# Data loading and preprocessing

from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data(one_hot = False)
X = X.reshape(len(X), 32 * 32 * 3)
testX = testX.reshape(len(testX), 32 * 32 * 3)
X = X / 255
testX = testX / 255

percentage_validate = 20

from sklearn.neighbors import KNeighborsClassifier
optimization_range = 100
score = np.empty([optimization_range, 2])

#k_variable = [100, 1000, 2000, 3000, 4000, 5000, 6000, 20000]
#score = np.empty([len(k_variable), 2])

for iteration in tqdm(range(1, optimization_range + 1)):
#for iteration in tqdm(range(1, len(k_variable) + 1)):
    kNN = KNeighborsClassifier(n_neighbors = iteration, algorithm = 'brute')
    kNN.fit(X, Y)
    accuracy = kNN.score(testX, testY)
    
    score[iteration - 1, 0] = iteration
    score[iteration - 1, 1] = accuracy
    
'''
start = time.time()
kNN.fit(X, Y) 
print("kNN.fit time: ", time.time() - start)

start = time.time()
print(kNN.score(testX, testY))
print("kNN.score time: ", time.time() - start)
'''

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.scatter(score[:, 0], score[:, 1])


# In[ ]:



