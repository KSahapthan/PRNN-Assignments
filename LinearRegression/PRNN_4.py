# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:14:51 2024

@author: sahap
"""
import warnings
warnings.filterwarnings("error")

#PRNN ASGN 1
#Grp No 8

#PROBLEM 4 
#BINARY CLASSIFICATION

#Same codes for PROBLEM 4 just replace the intial variables

#total data avilable = 1400
#Lets take 0.65 of it for training = 9100
A = 14000 #total data
B = 9100 #training data
C = A-B #test data
d = 10 #feature vector
f = 1 #label vector dimension
k = 2 #number of classes
import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
import random

file_path = r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 1\Data given\binary_classification_data_group_8_train.txt"

#data_array_F contains entire feature space data (A,d)
#data_array_f contains the training data features (B datapoints)

data_array = np.loadtxt(file_path,skiprows=1)
data_array_F = data_array[:,0:d] #(A,d)
data_array_f = data_array_F[0:B] #(B,d)

#data_array_L contains all the Labels
#data_array_l contains the training labels

#labels are like 4.0,8.0
#data_array_l[7][0] = 4.0

#Training data
data_array_L = data_array[:,[d]] #(A,1)
data_array_l = data_array_L[0:B] #(B,1)
 
#Test data
test_data_f = data_array_F[B:]
test_data_l = data_array_L[B:]

# To implement the following
# Bayes’ classifiers with 0-1 loss assuming
# 1) Normal
# 2) There is no 2
# 3) and GMMs (with diagonal covariances) as class-conditional densities
# For GMMs, code up the EM algorithm,
# 4) Non Parametric estimation 
# Parzen window with 2 different kernels
# 5) K-nearest neighbours with 2 different distance metrics
# 6) Linear classifier 0 vs R

# lets define some accuracy metrics
def Acc_metric(q:np.ndarray):
    #q is the CM
    TP =[0]*k
    TN =[0]*k
    FP =[0]*k
    FN =[0]*k
    Precision = [0]*k
    Recall = [0]*k
    F1 = [0]*k
    for i in range(k):
        TP[i] = q[i,i]
        #diagonals give true positives
        TN[i] = np.sum(q) - np.sum(q[i,:]) - np.sum(q[:,i]) + q[i,i] 
        #true negative is sum of all elements except those in that row/column
        FP[i] = np.sum(q[i,:])-q[i,i]
        #All sums in ith row except the diagonal
        FN[i] = np.sum(q[:,i])-q[i,i]
        #All sums in ith column except diagonal
        Precision[i] = TP[i]/(TP[i]+FP[i])
        Recall[i] = TP[i]/(TP[i]+FN[i])
        #just defined above
        F1[i] = 2*Precision[i]*Recall[i]/(Precision[i]+Recall[i])
        #F1 is just harmonic mean of precision and recall
    return (TP,TN,FP,FN,F1)

#Implementation 1

#Assuming a Normal class conditional density
#We already know what the MLE Estimate would be
#First we actually have to group the data into various classes
#Or we can use if condition while iterating over the data and accordingly calculate

l=[[np.zeros((d,)), 0, np.zeros((d,d))] for _ in range(k)]
#List comprehension is very important to create independent copies
# as using *10 would create copies
#with the same reference and modifying one would change all others as well!

#len(l)=10 ; len(l[0])=3 ; l[0][1]=0 ;l[0][0].shape=(d,) same as data_array_f[2]
#l contains a list of lists with each one for a particular class 
#Contains the running sum of data vectors,count,potential covariance matrix
#The count/35000 would sort of act as prior probabilities of class!

for i in range(0,B):
    x = data_array_f[i]
    y = int(data_array_l[i][0])
    l[y-1][0] += (x)
    l[y-1][1] += 1
#Now we have data wrt each of the classes in form of l
#Lets simply store the mean of class k in l[k-1][0] itself
for j in range(0,k):
    l[j][0] /= l[j][1]
#Now to estimate the covariance (we can do only after estimating mean)
for i in range(0,B):
    x = data_array_f[i]
    y = int(data_array_l[i][0])
    e = (x-l[y-1][0]).reshape((d,1))
    c = e @ (np.transpose(e))
    #e is x-mu with the appropriate class
    l[y-1][2] += c
#Now we simply store the covariance after dividing by the count
for j in range(0,k):
    l[j][2] /= l[j][1]

#Now the underlying class conditional densities have been estimated 
#Now we can define a likelihood function calculator and then implement
#a Bayes Classifier

def likelihood_calc1(t: np.ndarray, m :np.ndarray ,c :np.ndarray):
    # input t is of form (d,)
    t1 = t.reshape((d,1))
    m1 = m.reshape((d,1))
    #We first convert the 1d array into a (d,1) array and then operate
    det_cov = np.linalg.det(c)
    d1 = ((det_cov)**(-0.5))*((2*(np.pi))**(-d/2))
    d2 = np.reshape((t1-m1),(1,d))
    d3 = np.matmul(d2,np.linalg.inv(c))
    d4 = np.matmul(d3,t1-m1)
    #d4 is a single valued (1,1) array
    #To access it we must give [0,0]!
    k1 = np.exp((-1/2)*(d4[0,0]))
    result_Prob = d1 * k1
    return result_Prob
    # returns the "actual probability" for a gaussian

def Bayes_class1(x: np.ndarray):
    # x would generally be (d,)
    Log_like = [0 for _ in range(k)]
    # We cannot avoid computing all the Log_like = p(x|y).p(y))
    # Sum of Log_like will not be 0 since we havent normalised
    # But that doesnt matter for the Bayes Classifier as denominator is common for all
    for j in range(k):
        Log_like[j] = np.log((likelihood_calc1(x,l[j][0],l[j][2]))*(l[j][1])/B)
    max_index = Log_like.index(max(Log_like))
    return (max_index + 1)
    # returns the class

#Lets test our model on the test data 
def loss1():
    Loss=0 ; CM = np.zeros((k,k))
    #In multiclass classification, the confusion matrix is an extension of the binary classification confusion matrix. 
    #It is a square matrix where each row represents the instances in a predicted class
    #column represents the instances in an actual class. 
    #diagonal of the matrix -correctly classified
    #off-diagonal elements -misclassifications.
    for r in range(0,C):
        X = test_data_f[r]
        Y = int(test_data_l[r])
        P = Bayes_class1(X)
        if P == Y:
            CM[Y-1][Y-1] += 1
        if P != Y:
            Loss+=1
            CM[P-1][Y-1] += 1        
    Accuracy = (C-Loss)*100/C
    return (Accuracy,CM)

# R1 = loss1()
# Accuracy1 = R1[0]
# CM1 = R1[1]
# metrics_computed1 = Acc_metric(CM1)

#Accuracy1 is  34.55%!
#At least better than random guessing which leads to 50% accuracy

def loss_training():
    Loss=0
    for r in range(0,B):
        X = data_array_f[r]
        Y = int(data_array_l[r][0])
        P = Bayes_class1(X)
        if P != Y:
            Loss+=1
    Accuracy = (B-Loss)*100/B
    return (Accuracy)

#Implementation 3

"""Class conditional densities are assumed to be GMM
First we have to code up the EM Algorithm and vary the hyper parameter M
Then for each of the hyperparameter we must find the other parameters
Then evaluate the log likelihood/accuracy metric then finally fix upon one of them!"""

#First lets split up the data_array_f into the different classes
data_list = [[] for _ in range(k)]
for i in range(B):
    j = int(data_array_l[i][0])
    data_list[j-1].append(data_array_f[i])

b = [0 for _ in range(k)]
"""b is going to store the estimated parameters from the GMM for each of the classes
Estimated parameters for a GMM are vector_parameters; Each element of b is a dict """

def class_data(c: int):
    return np.array(data_list[c])
    # returns N x d array
    # here d=25 and N is #of labels in that class

def likelihood_calc3(t: np.ndarray, m :np.ndarray ,c :np.ndarray):
    # input t is of form (d,)
    t1 = t.reshape((1,d))
    m1 = m.reshape((1,d))
    #We first convert the 1d array into a (1,d) array and then operate
    det_cov = np.linalg.det(c)
    d1 = ((det_cov)**(-0.5))*((2*(np.pi))**(-d/2))
    d2 = np.reshape((t1-m1),(d,1))
    d3 = np.matmul(t1-m1,np.linalg.inv(c))
    d4 = np.matmul(d3,d2)
    #d4 is a single valued (1,1) array
    v = np.exp((-1/2)*(d4[0,0]))
    result_Prob = (d1)*v
    return result_Prob
    #return "exact actual probability" given a particular gaussian

"""Also define another likelihood since GMM likelihood is different
Its sum of likelihoods of each component of GMM"""
def lg(R,S):
    """This function takes in the data array S; Parameters of GMM is taken as R
    Calculates the net (l(theta)) in terms of sum of logs of p
    For GMM p is itself a sum over each Gaussian Component
    Bascically the total net likelihood"""
    P=0
    for i in range(0,len(S)):
        t = S[i] #(d,)
        sum = 0
        for j in range(0,len(R)):
            a = R[f"v_{j}"][0]
            b = R[f"v_{j}"][1]
            c = R[f"v_{j}"][2]
            X = a*likelihood_calc3(t, b, c)
            sum += X
        P += np.log(sum)
        #default base is e
    return P

#Also define Bayes classifier function as Bayes_class3
def Bayes_class3(x: np.ndarray,bb):
    #bb is the b list we want to give as input
    # x would generally be (d,)
    Log_like = [0 for _ in range(k)]
    for j in range(k):
        Log_like[j] = lg(bb[j],[x]) + np.log(l[j][1]/B)
    max_index = Log_like.index(max(Log_like))
    return (max_index + 1)

def GMM_update(m,g): #Updates b thats all
    #m is number of HyperParameters
    #g is # of iterations in the EM algorithm
    for q in range(k):
        #Lets vary M for each of class conditional density fixing #steps
        Final_Likelihood = []
        temp_vector_par = []
        for s in range(m,m+1):
            data_array_c = class_data(q)
            N = len(data_array_c)
            #So now we have the data in array format size is (N,d)
            #data_array[0,0] gives first component of first data
            #data_array[0] gives the v_1 vector
            #Each data has shape (d,) as 1d array
            #Now we do not yet know the # of clusterings
            #So initially we should give cluster value
            #Clustering to initialize parameters
            #First lets try to use K-Means to Cluster the data

            # Specify the number of clusters (ck)
            ck = s
            # Perform K-Means clustering
            centroids, _ = kmeans(data_array_c, ck)
            """Unpacking the centroids and distortion!
            This is a (ck,d) array ; We can directly access it to extract out the initial mean!
            Assign each data point to a cluster"""
            clusters, _ = vq(data_array_c, centroids)
            """This is a 1-d array of (N,) of [0/1] ; Centroids[0] has mean of 1st clustering
            Its number is given as 0 in clusters! ; Covariance have to be computed
            Calculate covariance for each cluster directly using if inside loop"""
            covariances = [np.cov((data_array_c[clusters==i]).T) for i in range(ck)]
            """Each element is (d,d) Covariance Matrix """

            #We have to enumerate to extract/loop
            M = ck 
            """Hypervariable same as # of clusters
            Create an empty dictionary to store the parameters as vectors"""
            vector_parameters = {}
            # Loop to create and store vector parameters
            for i in range(0, M):
                # Create the variable name dynamically (e.g., v_1, v_2, ...)
                variable_name = f"v_{i}" 
                """Variable name is a string ; GMM coeff"""
                count = 0
                for z in clusters:
                    if z==(i):
                        count += 1
                A_j = count/N
                M_j = centroids[i]
                # mean vector of size (d,) a 1d array
                C_j = covariances[i] # Covariance matrix
                vector = [A_j,M_j,C_j] # parameter for the gaussian it is a list
                # Store the vector in the dictionary
                vector_parameters[variable_name] = vector
            """Now we have a set of initial parameter θ.
            post prob depend on data point"""
            post_prob = np.zeros((N,M))

            #Now lets write the E step to compute the posterior probabilities
            def E_step():
                for i in range(0,N):
                    t = data_array_c[i]
                    L = []
                    for j in range(0,M):
                        a_j = vector_parameters[f"v_{j}"][0]
                        m_j = vector_parameters[f"v_{j}"][1]
                        c_j = vector_parameters[f"v_{j}"][2]
                        Likelihood = likelihood_calc3(t,m_j,c_j)
                        V = a_j*Likelihood
                        L.append(V)
                    D = sum(L)
                    for j in range(0,M):
                        post_prob[i,j] = (L[j]/D)

            #Now in the M step the above posterior probabilities will be used
            #Now we have to modify the parameters
            def M_step():
                for i in range(0,M):
                    #fist we access post_prob and find N_i
                    N_i = 0
                    for j in range(0,N):
                        N_i += post_prob[j,i]
                    new_ai = N_i/N #New a_i
                    #Next we want to find the new mean_i and cov_i simultaneously
                    #But only after finding mean we can find covariance!
                    #Now we need both the data and the post_prob
                    m_i = np.zeros((d,))
                    c_i = np.zeros((d,d))
                    for j in range(0,N):
                        p = post_prob[j,i]
                        t = data_array_c[j]
                        w_v = p * t
                        m_i += (w_v)
                    new_mi = m_i/N_i #(d,)
                    #Now using the new_mi lets find the new_ci
                    for j in range(0,N):
                        p = post_prob[j,i]
                        t = data_array_c[j]
                        x = t - new_mi
                        c_v = p * (np.matmul(np.reshape(x,(d,1)),x.reshape((1,d))))
                        c_i += c_v
                    new_ci = (c_i)/(N_i)
                    #Now lets update the parameters
                    vector_parameters[f"v_{i}"][0] = new_ai
                    vector_parameters[f"v_{i}"][1] = new_mi
                    vector_parameters[f"v_{i}"][2] = new_ci

            # Initialize an empty list to store values of p
            values_of_p = []
            for t in range(0,g):
                #Compute posterior prob and store them
                E_step()
                #Update the parameters
                M_step()
                #Find the likelihood
                values_of_p.append(lg(vector_parameters,data_array_c))

            #Plotting Likelihood vs EM steps
            iterations = range(1, len(values_of_p) + 1)
            plt.scatter(iterations, values_of_p, label='p values')
            plt.title('Scatter Plot of p values over Iterations')
            plt.xlabel('Number of Iterations')
            plt.ylabel('p values')
            plt.legend()
            plt.show()

            temp_vector_par.append(vector_parameters)
            Final_Likelihood.append(values_of_p[-1])

        #Plotting Likelihood at end of EM vs Hyperparamter M
        iterations = range(m,m+1)
        plt.scatter(iterations, Final_Likelihood, label='p values')
        plt.title('Max_Likelihood over hyperparameters for the class f"{q}"')
        plt.xlabel('Number of HyperParameters')
        plt.ylabel('Max_Likelihood')
        plt.legend()
        plt.show()

        max_index = Final_Likelihood.index(max(Final_Likelihood))
        b[q] = temp_vector_par[max_index]
        #The original list gets updated with the class conditional GMM parameters Corresponding to max_likelihood

#Even after saturating for 25 iteration it rose up to and likelihood increased beyond 40 as well

#Lets test it using training data set
def loss3(m,g):
    #almost same as loss1()
    Loss=0 ; CM = np.zeros((k,k))
    GMM_update(m,g)
    for r in range(0,C):
        X = test_data_f[r]
        Y = int(test_data_l[r])
        P = Bayes_class3(X,b)
        if P == Y:
            CM[Y-1][Y-1] += 1
        if P != Y:
            Loss+=1
            CM[P-1][Y-1] += 1
    Accuracy = (C-Loss)*100/C
    return (Accuracy,CM)

# R3 = loss3(5,40)
# Accuracy3 = R3[0]
# CM3 = R3[1]
# metrics_computed3 = Acc_metric(CM3)

#So we can also vary the Hyper Paramter M keeping g fixed

def loss_training3():
    Loss=0
    for r in range(0,B):
        X = data_array_f[r]
        Y = int(data_array_l[r][0])
        P = Bayes_class3(X,b)
        if P != Y:
            Loss+=1
    Accuracy = (B-Loss)*100/B
    return (Accuracy)

A_test_list = []
A_train_list = []
for s in range(10,15):
    A_test_list.append(loss3(s,40)[0])
    A_train_list.append(loss_training3())
    
#Plotting Accuracy vs Hyperparamter M
iterations = range(2,len(A_test_list) + 2)
plt.scatter(iterations, A_test_list, label='Testing Accuracy')
plt.scatter(iterations, A_train_list, label='Training Accuracy')
plt.title('Accuracy vs  hyperparameters"')
plt.xlabel('Number of HyperParameters')
plt.ylabel('Accuracy')
plt.legend()
plt.show() 
    
#Implementation 4

#Again we estimate the classconditional densities nonparametrically
#Like K-nearest we once again integrate bayes classifier with the estimation

hn = 1.5 #This is the window with but we must be able to vary it
def kernel(u:np.ndarray,z):
    #z for the kernel type
    if z==1:
        #Cubical discrete kernel
        #hn is sidelength
        #return phi(u)
        #u is (d,)
        R = 1
        for s in range(len(u)):
            if np.abs(u[s]) > 0.5:
                R=0
                break
        return R   
    if z==2:
        #Lets make this a gaussian kernel
        #Supposing that its a d-dim gaussian kernel
        #Directly invoke likelihood calc1
        return likelihood_calc1(u,np.zeros((d,)),np.identity(d))
#Both the kernels are valid since
#They are nonnegative and integrate to 1  
def Bayes_class4(x:np.ndarray,z):
    #x will directly come from test_data_f (d,)
    p=[0 for _ in range(k)]
    #p will capture net "kernel likelihood" class k_i
    #p(x) = kn(x)/nV
    #We have to iterate over all the data points and find kn(x)
    #with appropriate weightages based on distance depending on the kernel
    for s in range(k):
        kn = 0
        n = len(data_list[s])
        for r in range(n):
            kn += kernel((x-data_list[s][r])/hn,z)  
        p[s] = kn/(hn**d)
        #denominator can be ignored if its uniform for all classes
        #class priors already counted [kn(x)/n*hn]*n
    return (p.index(max(p))+1)

def loss4(z): #z is the kernel to be used
    Loss = 0; CM = np.zeros((k,k))
    C = 50 #local variable since computationally intensive
    for r in range(0,C):
        X = test_data_f[r]
        Y = int(test_data_l[r])
        P = Bayes_class4(X,z)
        if P == Y:
            CM[Y-1][Y-1] += 1
        if P != Y:
            Loss+=1
            CM[P-1][Y-1] += 1        
    Accuracy = (C-Loss)*100/C
    return (Accuracy,CM)

# R4 = loss4(2)
# Accuracy4 = R4[0]
# CM4 = R4[1]
# metrics_computed4 = Acc_metric(CM4)
# For 50 test samples Accuracy4 = 38% very low

#Implementation 5

from scipy import spatial 
#Just to compute norms thats all
#To do is first of all define a distance metric
#Use that metric to get K-nearest neighbours and solve it
knn = 4
def dmetric(x:np.ndarray,y:np.ndarray,z):
    if z==1:
        #standard euclidean distance
        return np.linalg.norm(x-y)
    elif z==2:
        #standard manhattan distance
        return np.linalg.norm(x-y,ord=1)
    elif z==3:
        #cosine distance
        return spatial.distance.cosine(x,y)

#Again same logic first to estimate class conditional densities for all classes
#But this time we dont really have to estimate at all points
#Practically we cant at all points since that requires infinte iterations
#The point where we need to estimate is actually the test_data points 
#The neighbours belong to training data points
#So here its directly to the testing no training

def Bayes_class5(x:np.ndarray,Y:np.ndarray,Z:np.ndarray,z:int):
    #z is for the metric to be used
    #x will directly come from test_data_f
    #Y is the underlying training dataset i.e data_array_f
    #Z is data_array_l
    p=[0 for _ in range(k)]
    #We will use built in function to compute max_k 
    dist_array = np.array([-(dmetric(x,Y[q],z)) for q in range(len(Y))])
    #of shape (len(Y),)
    k_indices = np.argpartition(dist_array, -knn)[-knn:]
    #This contains indices of knn nearest neighbours
    for s in range(knn):
        label = int(Z[k_indices[s]][0])
        p[label-1] += 1
    #We have to account for class priors as well
    #Class priors always come from original data_array_f
    #They are already stored in l
    for s in range(k):
        p[s] *= (l[s][1])/B
    return (p.index(max(p))+1)

#Testing time as usual
def loss5(z):
    #z is for dmetric to be used
    Loss = 0 ; CM = np.zeros((k,k))
    C = 350
    for r in range(0,C):
        X = test_data_f[r]
        Y = int(test_data_l[r])
        P = Bayes_class5(X,data_array_f,data_array_l,z)
        if P == Y:
            CM[Y-1][Y-1] += 1
        if P != Y:
            Loss+=1
            CM[P-1][Y-1] += 1        
    Accuracy = (C-Loss)*100/C
    return (Accuracy,CM)

# R5 = loss5(1)
# Accuracy5 = R5[0]
# CM5 = R5[1]
# metrics_computed5 = Acc_metric(CM5)
# For 350 test samples Accuracy5 = 37.7% very low
            

#Implementation 6 (One vs R)

#We augment the feature space by adding column of ones to (B x d+1) vector
#For each class C_k ,  C_k vs rest will be a binary classifier
#Its linear discriminant function will be determned by a modified label space
#And apply formula for optimal solution for least squared error
#Then once we get the (k-1) discriminant functions 
#We assign a point to Ck if all (k-1) return negative(meaning not Ci for i=1,..k-1)
#If two of them return positive its a ambiguous region and we assign randomly
#If only one of them return positive we simply assign it as the class

#There is a very crucial subtle point
#We got the linear discriminant function g_k of class k vs rest
#g_k(x) is either positive or negtaive [we ignore on the decision boundary]
#Which among these actually correspond to class k?
#It actually seems like the way we have gone on to derive the equations with the one-hot vector
#given value 1 if its in class k g_k(x)>0 alwasys corresponds to x in class k itself

#Lets adopt the convention that if g_k(x) > 0 belong to class k
#But we have to check if this is true...if its not have to just invert g_k(x)

data_array_f_a = np.c_[data_array_f,np.ones(B)]
#Augmented data_array_f is (B x (d+1))
#We can directly create Y_m_ = (B x k) where k is no of classes
#We will get W as (d+1,k) directly
Y_m = np.zeros((B,k))
#Modified label space
for r in range(B):
    column = int(data_array_l[r][0])
    Y_m[r][column -1] = 1
    #Sort of 1 hot vector       
X_t = data_array_f_a.transpose()
Z = np.matmul(X_t,data_array_f_a)
Z_i = np.linalg.inv(Z)
Wb = (Z_i) @ (X_t) @ (Y_m) #(d+1,k)
Wa = Wb[:,0:k-1] #(d+1,k-1)
Wa_t = Wa.transpose()
Wb_t = Wb.transpose()
def Bayes_class6a(x:np.ndarray):
    #x will directly come from test_data_f
    x_a = np.append(x,np.ones(1,), axis=0) #(d+1,)
    G = Wa_t @ (x_a.reshape((d+1,1))) #(k-1,1)
    Positive = 0
    out = 0
    for s in range(k-1):
        if (G[s][0] > 0) and Positive == 0:
            Positive += 1
            out = s
        elif (G[s][0] > 0) and Positive != 0:
            return (random.randint(1,k))
            #x has fallen into ambiguous region
    if Positive == 0:
        return k
    else:
        return (out + 1)

#Bayes Class6b is already like the tradidtional best routed K-class classifier
#But accuracy is not as much as we want
def Bayes_class6b(x:np.ndarray):
    #x will directly come from test_data_f
    #This time T will comprise of class Ck as well (B,k) itself
    x_a = np.append(x,np.ones(1,), axis=0)
    G = Wb_t @ (x_a.reshape((d+1,1))) #(k,1)
    return (G.argmax(axis=0)[0]+1)
    #returning (d+1,)
#Testing time as usual
def loss6a():
    Loss = 0; CM = np.zeros((k,k))
    for r in range(0,C):
        X = test_data_f[r]
        Y = int(test_data_l[r])
        P = Bayes_class6a(X)
        if P == Y:
            CM[Y-1][Y-1] += 1
        if P != Y:
            Loss += 1
            CM[P-1][Y-1] += 1        
    Accuracy = (C-Loss)*100/C
    return (Accuracy,CM)
def loss6b():
    Loss=0 ; CM = np.zeros((k,k))
    for r in range(0,C):
        X = test_data_f[r]
        Y = int(test_data_l[r])
        P = Bayes_class6b(X)
        if P == Y:
            CM[Y-1][Y-1] += 1
        if P != Y:
            Loss += 1
            CM[P-1][Y-1] += 1        
    Accuracy = (C-Loss)*100/C
    return (Accuracy,CM)

# R6a = loss6a()
# Accuracy6a = R6a[0]
# CM6a = R6a[1]
# metrics_computed6 = Acc_metric(CM6a)
# R6b = loss6b()
# Accuracy6b = R6b[0]
# CM6b = R6b[1]
# metrics_computed6b = Acc_metric(CM6b)
# For 25000 test samples Accuracy6a = 9.28% and Accuracy6b = 23.916% very low

#What is happenning is in almost all cases x is landing in ambiguous region for 6a
#Multiple classes actually giving rise to positive g_k(x)
#Since dim is 25 this happens almost for all x
#Even taking the max{g_k(x)} doesnt help much
    
#Lets start with some extra implementations  

#EXTRA 1
#Logistic Regressor for Classification using ERM and Gradient Descent

#Rmpirical risk R(h) = sum j=1 to B ||h(x_j)-(y_j)||**2_2
#Assume h(x_j) and (y_j) are (k,1)

#Let h_i(x) = exp(-M_i^tx)/sum[exp(-M_j^tx)]
#i will vary from 0 to k-1
#We accomodate for constant term by adding a column of ones
#M_i have shapes (d+1,1)
#These are the soft max functions
#h(x) = (h_0(x),....h_k-1(x))^t
#Let subscripts of h be denoted as t
#We are going to use Squared Error Loss
#We directly choose the class with max h_i(x)

X1 = data_array_f_a #(B,d+1)
Y1 = data_array_l #(B,1)

#initialisation 
RChoice = [random.uniform(0.3,0.8) for _ in range(20)]
LM = np.random.choice(RChoice, size = (k,d+1))
#We just give a random value to start with

#For convenience we define a exp_func
def exp_func(x:np.ndarray,M:np.ndarray):
    # x will be (d+1,)
    # M would be (d+1,)
    try:
        val = np.exp(-((M.reshape((1,d+1)) @ x.reshape((d+1,1)))[0][0]))
        return val
    except RuntimeWarning:
        return 0
        
#For convenience we define h_function

#np.exp(700) is roughly the maximum that can be computed
#np.exp(700) = 1.01 * 10^304
#Same limit would be for negative as well which is just the reciprocal

#In our data all the x_i are in [0,2]
#Mt.x is something like 25(each) < 700
#So each component must be less than 28!
#So each element of LM must be < 6
#Dividing by 1/B is crucial for GD to not shoot up

def h_func(x:np.ndarray):
    #x must be (d+1,)
    #return h(x) as a list
    h = [0 for i in range(k)]
    sum = 0
    for s in range(k):
        #LW[s,:] would give a (d+1,)
        D = exp_func(x,LM[s,:])
        h[s] = D
        sum += D
    for j in range(k):
        h[j] = h[j]/sum
    return h
    #h is a list at all costs!

#Partial derivative of the Risk wrt the parameter M_i
#is given by 2*(sum j=1 to B)((sum t=0 to k-1)(h_t(x_j)-(y_j_t)*d/dM_i[h_t(x_j)]))

#The inner sum on fixing a data point can be written as (k,d+1).transpose() @ (k,1) where
#the 1st part is obtained by stacking derivatives dht/M_i (as columns side by side)
#the second part is simply (h(x_j)-y_j)

#This derivative ultimately is a (d+1,1) as we want after summing over j
#To simplify this expression we define 2 other functions

#We compute the entire gradient intelligently at one go at a fixed point 

#This is for computing partial derivative wrt h
def partial_deriv_h(x0:np.ndarray):
    #returns PD_list which has PD for all M_i and as a bonus h(x0) as well
    #x0 is the point being evaluated at muse be (d+1,)
    #i denotes the M_i (0 to k-1)
    PD_list =[0 for _ in range(k)]
    u = h_func(x0) #its as list
    for i in range(k):
        #returns a tuple with a list with each element (k,d+1)
        PD = np.zeros((k,d+1)) 
        for s in range(k): 
            if s != i:
                PD[s] = x0 * u[s] * u[i]
            else:
                PD[s] = x0 * u[s] * (1-u[s])
        PD_list[i] = PD
        
    PD_list.append(u)
    return PD_list #length k+1
    
#This computes the net final derivative wrt all M_i for a fixed datapoint x0
def partial_deriv_j(x0:np.ndarray,y0:np.ndarray):
    #x0 is the datapoint and y0 is the label in one hot form
    Derivative = np.zeros((k,d+1))
    Info = partial_deriv_h(x0)
    for i in range(k):
        Derivative[i] = (Info[i].transpose() @ (np.array((Info[k]-y0))).reshape((k,1))).reshape((d+1,))
        #Above is (d+1,)
    return Derivative 
    #returns (k,d+1)
    #But it has evaluated it at the point x0 only

#This sums over all datapoints to get the final gradient descent matrix
def gradient():
    GD = np.zeros((k,d+1)) #Gradient descent Matrix
    for s in range(B):
        GD += 2 * partial_deriv_j(X1[s],Y1[s])
    return GD/B #can be performed only for np arrays
    #There is a 1/B term in the empirical risk so that must stick to the derivative as well
    #Else GD becomes very large
    #returns the final gradient descent matrix as (k,d+1)
  
#Now lets start the actual gradient descent and find LM
def Empirical_Risk():
    sum = 0
    for i in range(B):
        Y = [0]*k
        Y[int(data_array_l[i])-1] = 1
        sum += (dmetric(np.array(h_func(data_array_f_a[i])).reshape((k,)),(np.array(Y)).reshape((k,)),1))**2
    return sum/B
N = 20
#N is no of times we want to carry out the gradient descent
def step(M:np.ndarray):
    #M is the GD matrix
    return 0.04

#We perform the Gradientdescent
Risk = [0]*(N+1)
Risk[0] = Empirical_Risk()

#break point
for n in range(1,N+1):
        GD = gradient()
        stepsize = step(GD)
        print(n)
        LM -= stepsize*GD
        Risk[n] = Empirical_Risk()
        if n>1 and (Risk[n] > Risk[n-1]):
            break
        
# Create a scatter plot
iterations = range(1, len(Risk) + 1)
plt.scatter(iterations, Risk, label='ER')
# Customize the plot
plt.title('ER over Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('ER')
plt.legend()
# Show the plot
plt.show()

#Note that the number on the top left denotes 2 things (scale and offset)
        
def Bayes_classE1(x:np.ndarray):
    #x is (d,)
    x_a = np.append(x,np.ones(1,), axis=0) #(d+1,)
    T = h_func(x_a) #a list
    return (T.index(max(T)) + 1)

def lossE1():
    Loss=0 ; CM = np.zeros((k,k))
    for r in range(0,C):
        X = test_data_f[r]
        Y = int(test_data_l[r][0])
        P = Bayes_classE1(X)
        if P == Y:
            CM[Y-1][Y-1] += 1
        if P != Y:
            Loss += 1
            CM[P-1][Y-1] += 1        
    Accuracy = (C-Loss)*100/C
    return (Accuracy,CM)      
    
# RE1 = lossE1()
# AccuracyE1 = RE1[0]
# CME1 = RE1[1]
# metrics_computedE1 = Acc_metric(CME1)
# For 25000 test samples AccuracyE1 = 9.69

def ER_test():
    sum = 0
    for i in range(C):
        Y = [0]*k
        Y[(int(test_data_l[i][0]))-1] = 1
        x_a = np.append(test_data_f[i],np.ones(1,),axis=0)
        sum += (dmetric(np.array(h_func(x_a)).reshape((k,)),(np.array(Y)).reshape((k,)),1))**2
    return sum/C
    
#print(ER_test) is 0.9000067590943314
#print(Risk[-1]) is 0.9000082428217324


    