from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random


'''file = 'Data_Classification_1.csv'
read_file = pd.read_csv(file)
data = read_file.as_matrix()
row, column = data.shape
X = data[:, 0:column-1]
Y = data[:,n-1]
Y = Y.reshape(-1, 1)'''
X = [[0,0],[0,1],[1,0],[1,1]]
X = np.matrix(X)
m, n = X.shape
K = int(input("Please input the value of k : "))
centroid = np.zeros((K, n))

#choose random number from a matrix without duplicate value

centroid = X[np.random.choice(np.arange(len(X)), K, replace=False)]
print(type(centroid))
print(centroid)
c_index = np.zeros(m)
copy = np.zeros(m)
Iter = 0;
MaxIter = int(input("Please input the value of Max Iteration : "))
while Iter <= MaxIter:
    p = 0
    for i in range(0, m):
        distance = np.zeros(K)
        for j in range(0, K):
            for k in range(0, n):
                distance[j] += (X[i, k]-centroid[j, k]) ** 2
        c_index[p] = np.argmin(distance)
        p += 1
    centroid = np.asarray([X[c_index == k].mean(axis=0) for k in range(K)])
    centroid = np.matrix(centroid)
    print(centroid)

    if np.array_equal(c_index, copy):
        break
    copy = c_index
    Iter += 1




