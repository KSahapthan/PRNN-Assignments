# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:59:04 2024
@author: sahap
"""

import warnings
warnings.filterwarnings("error")

import numpy as np
import matplotlib.pyplot as plt

#Logistic Regressor for Classification of KMNIST Images

#total data avilable = 14000
#Lets take 0.65 of it for training = 9100
A = 14000 #total data
B = 9100 #training data
C = A-B #test data
d = 10 #feature vector
f = 1 #label vector dimension
k = 2 #number of classes
#Empirical risk R(h) = cross-entropy loss
#Assume h(x_j) and (y_j) are (k,1) arrays

#Let h_i(x) = exp(-M_i^tx)/sum[exp(-M_j^tx)]
#i will vary from 0 to k-1
#We accomodate for constant term by adding a column of ones
#M_i have shapes (d+1,1)
#These are the soft max functions
#h(x) = (h_0(x),....h_k-1(x))^t

file_path = r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 1\Data given\binary_classification_data_group_8_train.txt"

#data_array_F contains entire feature space data (A,d)
#data_array_f contains the training data features (B datapoints)

data_array = np.loadtxt(file_path,skiprows=1)
data_array_F = data_array[:,0:d] #(A,d)
data_array_f = data_array_F[0:B] #(B,d)

#Training data
data_array_L = data_array[:,[d]] #(A,1)
data_array_l = data_array_L[0:B] #(B,1)

#Test data
test_data_f = data_array_F[B:]
test_data_l = data_array_L[B:]
data_array_f_a = np.c_[data_array_f,np.ones(B)]
test_data_f_a = np.c_[test_data_f,np.ones(C)]

#Each element of shape (26,)
#initialisation 
RChoice = [np.random.uniform(0.3,0.6) for _ in range(20)]
LM = np.random.choice(RChoice, size = (k,d+1))
#We just give a random value to start with

#For convenience we define h_function
#Dividing by 1/B is crucial for GD to not shoot up
def h_func(x:np.ndarray):
    #x must be (d+1,)
    #return h(x) as an array
    H = np.exp(-(LM @ x.reshape((-1,1))))
    return (H/np.sum(H))
    #H is an array of (10,1)

#This sums over all datapoints to get the final gradient descent matrix
def gradient():
    D = np.zeros((k,d+1)) #Gradient Matrix
    for s in range(B):
        x1 = data_array_f_a[s]
        y1 = int(data_array_l[s])
        z1 = h_func(x1)
        for r in range(k):
            new = float(z1[r])
            if r == y1:
                D[r] += ((-1) * (new-1) * x1)
            else:
                D[r] += ((-1) * (new) * x1)
    return D/B #can be performed only for np arrays
    #There is a 1/B term in the Loss
    #Else GD becomes very large
    #returns the final gradient descent matrix as (k,d+1)
  
#------------------------#
#Now lets start the actual gradient descent and find LM
g = 100
#g is no of times we want to carry out the gradient descent
#We perform the Gradientdescent
lr = 0.5
ER_list = []
Accuracy_train_list = []
Accuracy_test_list = []
for n in range(g):
    sum = 0
    sum1 = 0
    sum2 = 0
    GD = gradient()
    LM -= ((lr) * GD)

    #Calculate training accuracy
    for s in range(B):
        x1 = data_array_f_a[s]
        y1 = int(data_array_l[s])
        class_prob = list(h_func(x1).reshape((-1,)))
        yp = np.argmax(class_prob)
        sum += (-np.log(float(class_prob[y1])))
        if (yp != y1):
            sum1 += 1
    #Calculate Testing Accurcay
    for s in range(C):
        x1 = test_data_f_a[s]
        y1 = int(test_data_l[s])
        class_prob = list(h_func(x1).reshape((-1,)))
        yp = np.argmax(class_prob)
        if (yp != y1):
            sum2 += 1
    Accuracy_train_list.append((B-sum1)*100/B)
    Accuracy_test_list.append((C-sum2)*100/C)
    ER_list.append(sum/B)
    print(n)

#Plotting
iterations = range(len(ER_list))
plt.scatter(iterations, ER_list, label='ER')
plt.title('ER over Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('ER')
plt.legend()
plt.show()

#Plotting
iterations = range(len(ER_list))
plt.scatter(iterations, Accuracy_train_list, label='Training')
plt.scatter(iterations, Accuracy_test_list, label='Testing')
plt.title('Accuracy over Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#22 parameters in the model
