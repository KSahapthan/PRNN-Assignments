# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:31:24 2024

@author: saha
"""
#PRNN ASGN 1
#Grp No 8

#PROBLEM 1 
#MULTILINEAR REGRESSION

#total data avilable = 14000
#Lets take 0.65 of it for training = 9100
A = 14000 #total data
B = 9100 #training data
C = A-B #test data
d = 10 #feature vector
f = 3 #label vector

import numpy as np
import matplotlib.pyplot as plt
file_path = r"C:\Users\sahap\OneDrive\Desktop\regression_data_multilinear_group_8_train.txt"

#data_array_F contains entire feature space data
#data_array_f contains the training data features (B datapoints)
raw_data = np.loadtxt(file_path,skiprows=1)
data_array_F = raw_data[:,0:d]
data_array_f = data_array_F[0:B]

#data_array_L contains all the Labels
#data_array_l contains the training labels

data_array_L = raw_data[:,d:d+f]
data_array_l = data_array_L[0:B]

#data_array_f.shape=(B,d)
#There are B data points as rows
#There are d columns
#data_array_f[0].shape=(d,)
#data_array[0][1] gives us a single numerical value

#Suppose we are using the linear regression with MSE and solving ERM
#Then the optimal solution is already known to us in form of 
#W=(XtX)^-1XtY

#So we can split the problem into 3 categories for each of the coordinates
#h(x)=w^t.(x)+w0
#Our case W can be a  matrix and we can directly find it
#We can accomodate the bias by using W as a (d+1) x 1 vector
#Append 1 to columns of data

X = np.c_[data_array_f,np.ones(B)]
#We first add a column of ones to accomodate bias
Y = data_array_l
X_t = X.transpose()
Z = np.matmul(X_t,X)
Z_i = np.linalg.inv(Z)
W = (Z_i) @ (X_t) @ Y #(d+1,f)
W_t = W.transpose()

#Test data is already appended with 1
test_data_f = np.c_[data_array_F[B:],np.ones(C)] #(C,d+1)
test_data_l = data_array_L[B:] #(C,f)

#Y axis will be predicted value
#X axis will be the true value
R = (test_data_f) @ (W) #(C,d+1)
#R is basically the predicted values on the test_data set

#Finding line of best fit is again a regression problem!
regression_coeff = [[0,0] for _ in range(f)]
#First one is intercept
#Second one is the slope
for s in range(f):
    Y_f = R[:,s].reshape((C,1))
    X_f = np.c_[test_data_l[:,s].reshape((C,1)),np.ones(C)] #(C,2)
    X_t_f = X_f.transpose()
    Z_f = np.matmul(X_t_f,X_f)
    Z_i_f = np.linalg.inv(Z_f)
    W_f = (Z_i_f) @ (X_t_f) @ Y_f
    #W is (2,1)
    regression_coeff[s][0] = W_f[1][0]
    regression_coeff[s][1] = W_f[0][0]  

#Now lets compute the metrics
PC = [0,0,0] #Pearson Corr Coeff
MSE = [0,0,0] #Mean Squared Error
MAE = [0,0,0] #Mean Absolute Error
for s in range(f):
    PC[s] = np.corrcoef(test_data_l[:,s],R[:,s])[0][1]
    MSE[s] = np.mean((test_data_l[:,s] - R[:,s]) ** 2)
    MAE[s] = np.mean(np.abs((test_data_l[:,s] - R[:,s])))
    
# Create a single figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for i, ax in enumerate(axes):
    # Plotting
    ax.plot(test_data_l[:, [i]], R[:, [i]], marker='o', label=f'Predicted Y{i + 1}')
    line_of_best_fit_y = np.dot(np.column_stack((np.ones_like(test_data_l[:, [i]]), test_data_l[:, [i]])), regression_coeff[i])
    ax.plot(test_data_l[:, [i]], line_of_best_fit_y, color='green', label='Line of Best Fit')
    ax.plot(test_data_l[:, [i]], test_data_l[:, [i]], linestyle='--', color='red', label='y=x')
    ax.set_xlabel(f'Actual Y{i + 1}')
    ax.set_ylabel(f'Predicted Y{i + 1}')
    ax.set_title(f'Linear Regression for Y{i + 1}')
    ax.legend()

    # Add annotations
    text = f"MSE: {MSE[i]:.2f}\nMAE: {MAE[i]:.2f}\nPearson: {PC[i]:.2f}"
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

#EXTRA1
#Analysing our Model
#Let us consider the Empirical risk associated with our model
def Empirical_Risk1(M1:np.ndarray): 
    #M1 will be the parameters (W2)
    sum = 0
    for i in range(B):
        x = data_array_f[i] #(d,)
        x_a = np.append(x,np.ones(1,),axis =0) #(d+1,)
        x_b = x_a.reshape((d+1,1))
        Q = (M1.transpose()) @ (x_b) #(f,1)
        Y = data_array_l[i].reshape((f,1)) #(f,1)
        sum += np.linalg.norm(Q-Y)**2
    return sum/B

def True_Risk1(M2:np.ndarray):
    #M2 will be the parameters (W2)
    sum = 0
    for i in range(C):
        x_a = test_data_f[i] #(d+1,)
        x_b = x_a.reshape((d+1,1))
        Q1 = (M2.transpose()) @ (x_b) #(f,1)
        Q2 = test_data_l[i].reshape((f,1)) #(f,1)
        sum += np.linalg.norm(Q2-Q1)**2
    return sum/C

#Empirical_Risk1(W) = 24.380
#True_Risk1(W) = 24.501

#So Empirical_Risk and True_Risk almost seems to be same
#But now can we reduce the True_Risk even further by Regularisation?
#Lets try l2 regularisation
#We already know the solution ...JUST add lamdaI 
def Regularizer(v):
    LambdaI = v * np.identity(d+1) 
    Z2 = np.linalg.inv(Z + LambdaI)
    W2 = (Z2) @ (X_t) @ Y #(d+1,f)
    return W2

#Now we again check the empirical and true risk for different values of lambda and plot a graph
ER_list = []
TR_list = []
for i in range(-200,200,10):
    W2 = Regularizer(i)
    R_ER = Empirical_Risk1(W2)
    R_TR = True_Risk1(W2)
    ER_list.append(R_ER)
    TR_list.append(R_TR)
    
# Create a scatter plot
iterations = range(-200,200,10)
plt.scatter(iterations, ER_list, label='ER')
plt.scatter(iterations, TR_list, label='TR', marker='^') 
 # You can customize the marker type if needed
# Customize the plot
plt.title('Scatter Plot of ER and TR vs Regularisation Constant')
plt.xlabel('Regularisation Constants')
plt.ylabel('Values')
plt.legend()
# Show the plot
plt.show()
    
#We can see that the true risk is minimised at around lamda = -25
#while the empirical risk is minimised at lamda = 0!!
#Regularised ER = Empirical_Risk1(Regularizer(-25)) = 24.389
#Regularised True = True_Risk1(Regularizer(-25)) = 24.492
#Though there is a difference in this case its very minute

#EXTRA2 
#We will use gradient descent for Lasso Regression
def sign(M:np.ndarray):
    Out = np.zeros(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i][j] > 0:
                Out[i][j] = 1
            if M[i][j] <= 0:
                Out[i][j] = -1
    return Out

def lasso_gradient(w:np.ndarray,v): #v is the lambda
    GD = np.zeros((d+1,f))
    for s in range(B):
        x = data_array_f[s] #(d,)
        x_a = np.append(x,np.ones(1,),axis =0) #(d+1,)
        x_b = x_a.reshape((d+1,1))
        Q3 = (w.transpose()) @ (x_b) #(f,1)
        y2 = data_array_l[s].reshape((f,1))
        Q4 = Q3 - y2
        Q5 = (x_b) @ (Q4.transpose()) #(d+1,f)
        GD += Q5     
    GD *= 2   
    GD += (v*sign(w))
    return GD/B #Gradient Descent matrix

#Lets start with the W we got earlier
#Let v denote the regularisation constant

Lasso_W = W #(d+1,f)

def lasso_loss(v):
    A1 = Empirical_Risk1(Lasso_W)
    A2 = 0
    for i in range(d+1):
        for j in range(f):
            A2 += np.abs(Lasso_W[i][j])
    return (A1) + (v*A2/B)

LL = [] #Lasso_Loss wrt regularised lasso minimum parameter
TLL = [] #True Risk associated with regularised lasso minimum parameter
ELL = [] #Empirical Risk associated with regularised lasso minimum parameter
for t in range(0,300,10):
    #Iterating over different regularisation constant
    Lasso_W = W
    lasso_loss_list = []
    lasso_loss_list.append(lasso_loss(t))
    #Gradient steps for the particular lamda
    v = t
    n = 100
    for i in range(n):
        Lasso_W = Lasso_W - (0.25 * lasso_gradient(Lasso_W,v))
        lasso_loss_list.append(lasso_loss(v))
    LL.append(lasso_loss_list[-1])
    TLL.append(True_Risk1(Lasso_W))
    ELL.append(Empirical_Risk1(Lasso_W))
    
    # # Create a scatter plot
    # iterations = range(n+1)
    # plt.scatter(iterations, lasso_loss_list, label='Lasso_Loss')
    # # Customize the plot
    # plt.title('Scatter Plot of Lasso_Loss vs Gradient Steps')
    # plt.xlabel('Gradient Steps')
    # plt.ylabel('Lasso_Loss')
    # plt.legend()
    # # Show the plot
    # plt.show()
    
# Create a scatter plot
iterations = range(0,300,10)
plt.scatter(iterations, LL, label='Lasso_Loss')
plt.scatter(iterations, TLL, label='True_Risk')
plt.scatter(iterations, ELL, label='Empirical_Risk')
# Customize the plot
plt.title('Scatter Plot of Lasso_Loss vs Regularisation Constant')
plt.xlabel('Regularisation Constant')
plt.ylabel('Lasso_Loss')
plt.legend()
# Show the plot
plt.show()


        
        
        
    
    
    





        




