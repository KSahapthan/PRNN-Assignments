# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:36:45 2024

@author: sahap
"""
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, i_d, o_d, h_l, neurons, activ_f, loss_f, o_act_f):
        #h_l count number of adjustable parameters
        #Actual no of "hidden layers" = h_l - 1
        #Neurons would be a list of length (h_l -1) having info about output dimension in each of the "hidden layer"
        #loss_f is the loss function we want to use
        #activ_f is the "Non-Linear-h function" we want to use in all the transformations except the last one
        #o_act_f denotes the "output activation function" we want to use finally

        self.i_d = i_d #input dimension
        self.o_d = o_d #output dimension
        self.h_l = h_l
        self.neurons = neurons
        self.loss_f = loss_f
        self.activ_f = activ_f
        self.out_activ_f = o_act_f

        #Lets create space for storing the parameters in all layers
        self.weights = [0] * self.h_l
        #We club the biasses and weights

        #Initialising the parameters
        #Assuming h_l >= 2
        #We use Xavier initialization
        self.weights[0] = (np.random.randn(self.i_d + 1, self.neurons[0]) / np.sqrt(self.i_d + 1))
        #(i_d + 1,M1)
        for i in range(1, self.h_l - 1):
            self.weights[i] = (np.random.randn(self.neurons[i-1] + 1, self.neurons[i]) / np.sqrt(self.neurons[i-1] + 1))
        self.weights[self.h_l - 1] = (np.random.randn(self.neurons[-1] + 1, self.o_d) / np.sqrt(self.neurons[-1] + 1))

    def activation_func(self,a:np.ndarray):
        #Input is a which is a vector (a1,a2,..aj)
        #Acts component wise but we can do the calculation together
        if self.activ_f == 'Logistic Sigmoid':
            return np.tanh(a)
        if self.activ_f == 'Sigmoid':
            return 1/(1+np.exp(-a))
        if self.activ_f == 'Relu':
            return a * (a > 0)
        if self.activ_f == 'Swish':
            return (a*1/(1+np.exp(-a)))
            #Return shape is same as input shape
            #This would be the z!

    def output_activ_func(self,y:np.ndarray):
        if self.out_activ_f == 'linear':
            return y
        if self.out_activ_f == "Softmax":
            A = np.zeros((y.shape[0],y.shape[1]))
            for s in range(A.shape[0]):
                y2 = y[s]
                A[s] = (np.exp(-y2) / np.sum(np.exp(-y2), axis=0))
            return A
            #Assuming classes are labelled (0,1,...k-1)
            #return shape is same as input shape
            #For the last outputs namely "y" we dont carry out "the h func".

    #Forward pass
    #While doing forward pass we have to store the activations of all the hidden and output units
    #This will help us to compute the derivatives easily later on
    def forward_pass(self, x:np.ndarray):
        B1 = x.shape[0]
        #x is of shape(B,d+1) is entire data
        Act_list = [0] * self.h_l
        #except the input part all hidden layers + output contribute to activation
        #a is referring to the output before activation
        b = np.copy(x) # (B,d+1)
        for i in range(self.h_l - 1):
            #iterating over each layer except last layer
            a = (b @ self.weights[i]) #(B,M_{i+1})
            z = self.activation_func(a) #(B,M_{i+1}) its the final output of that layer
            #We update for next iteration and store the activation output which is a vector[including append 1]
            b = np.c_[z,np.ones(B1)] #(B,M_{i+1}+1)
            Act_list[i] = a #(B,M_{i+1})
        y = (b @ self.weights[self.h_l - 1]) #(B,o_d)
        #We act to get the final output of the neural network
        #There is no nonlinear transformation
        Act_list[self.h_l - 1] = y
        return Act_list
    #Entire info of the forward pass is returned including the final output

    # derivative of the activation function
    def activation_derivative(self, a):
        #essentially inputting the "a" vector (i think its (f,1))
        if self.activ_f == "Logistic Sigmoid":
            return (1 - (np.tanh(a))**2)
        if self.activ_f == "Sigmoid":
            return (1)/(1+np.exp(a))*(1-(1/(1+np.exp(a))))
        if self.activ_f == "Relu":
            return 1 * (a>0)
        if self.activ_f == "Swish":
            return (a*1/(1+np.exp(-a))) + (1/(1+np.exp(-a))) * (1 - (a*1/(1+np.exp(-a))))

    #Lets write the code for Error-Back-Propagation to compute the gradient
    def backpropagation(self, x:np.ndarray, y:np.ndarray):
        #x is entire data set (B,d+1) and y is (B,1)
        #y from the training set is needed only to compute the output delta
        info = self.forward_pass(x)
        Derivative = [0 for i in range(self.h_l)]
        deltas = [np.zeros((B,self.neurons[i])) for i in range(self.h_l - 1)] #for all hidden layers except last
        deltas.append(np.zeros((B,self.o_d))) #for the last one

        #First we compute delta for all the layers and all the parameters
        # calculate delta for output layer
        info2 = (self.output_activ_func(info[-1])) #After acting the output_activation_func
        #input is (B,o_d) also same shape
        for r in range(self.o_d):
            info3 = info2[:,r]
            #Take all rows and columns r
            for s in range(B):
                if int(y[s]) == r:
                    deltas[-1][s][r] = (-(info3[s]-1))
                else:
                    deltas[-1][s][r] = (-(info3[s]))
            #dk = "corresponding derivative for that loss"

        # calculate delta for all the hidden layers
        for i in reversed(range(self.h_l - 1)):
            info1 = self.activation_derivative(info[i])
            #(B,??) #h'(a) as an array
            for j in range(self.neurons[i]): #accounted for bias as well
                for s1 in range(B):
                    deltas[i][s1][j] = (np.dot(self.weights[i+1][j],deltas[i+1][s1]) * (float(info1[s1][j])))
        #calculate the final derivative
        #Assume h_l >=2
        deriv_0 = np.zeros((self.i_d+1,self.neurons[0]))
        for r in range(self.i_d+1):
            for s in range(self.neurons[0]):
                deriv_0[r][s] = np.dot(x[:,r],deltas[0][:,s])
        Derivative[0] = deriv_0
        for i in range(1,self.h_l-1):
                #Derivative at the ith layer must be a matrix of shape (M_{i}+1,M_{i+1}) corresponding to
                #that adjustable parameter
                #input dimension M_{i} is actually neurons[i-1]
                #output dimension M_{i+1} is neurons[i]
                deriv = np.zeros((self.neurons[i-1]+1,self.neurons[i]))
                for r in range(self.neurons[i-1]+1):
                    if r != self.neurons[i-1]:
                        for s in range(self.neurons[i]):
                            deriv[r][s] = np.dot(info[i-1][:,r],deltas[i][:,s])
                    else:
                        for s in range(self.neurons[i]):
                            deriv[r][s] = np.dot(np.ones((B,)),deltas[i][:,s]) #we have appended 1
                Derivative[i] = deriv
        deriv_1 = np.zeros((self.neurons[-1]+1,self.o_d,))
        for r in range(self.neurons[-1]+1):
            if r != self.neurons[-1]:
                for s in range(self.o_d):
                    deriv_1[r][s] = np.dot(info[-2][:,r],deltas[-1][:,s])
            else:
                for s in range(self.o_d):
                    deriv_1[r][s] = np.dot(np.ones((B,)),deltas[-1][:,s]) #we have appended 1
        Derivative[-1] = deriv_1
        return (Derivative,info[-1],deltas[0])
    
import warnings
warnings.filterwarnings("error")

# Code for CNN 
# It has to extract out the features of input and return some reduced dimension output
class CNN:
    def __init__(self, i_d, o_d, c_l, activ_f, loss_f, o_act_f, f_count, f_sizes, padding, stride):
        #i_d  and o_d will be (a,b,c)
        #f_sizes will be (f1,f2,f3)
        #Everything will be 3D triplets
        #c_l count number of adjustable Convolution Layers
        #loss_f is the loss function we want to use
        #activ_f is the "Non-Linear-h function" we want to use in all the transformations except the last one
        #o_act_f denotes the "output activation function" we want to use finally
        
        self.i_d = i_d #input dimension 
        self.o_d = o_d #output dimension
        self.c_l = c_l 
        self.loss_f = loss_f
        self.activ_f = activ_f
        self.out_activ_f = o_act_f
        self.f_count = f_count #list of number of filters in each layer
        self.f_sizes = f_sizes #list of fliter sizes in each layer
        self.padding = padding #list of padding in each layer
        self.stride = stride #list of stride in each layer
        
        #Lets create space for storing the parameters in all layers
        self.filters = [0] * self.c_l
        #Each layer has a list of filters (each filter is a 3D array)
        self.biases = [0] * self.c_l
        #Each filter in a layer has just one number for bias (Each element is a list)
        
        #Initialising the parameters
        for i in range(self.c_l):
            temp_f = []
            temp_b = []
            for j in range(f_count[i]):
                temp_f.append(np.random.randn(self.f_sizes[i][0],self.f_sizes[i][1],self.f_sizes[i][2])/10)
                #All filters in a layer have same size
                #Each filter is stored as an array
                temp_b.append(np.random.randn()/10)
                #Each filters bias is just a number
            self.filters[i] = temp_f
            self.biases[i] = temp_b
            
    def activation_func(self,a:np.ndarray):
        #Input is a which is a vector (a1,a2,..aj)
        #Acts component wise but we can do the calculation together
        if self.activ_f == 'Logistic Sigmoid':
            return np.tanh(a)
        if self.activ_f == 'Sigmoid':
            return 1/(1+np.exp(-a))
        if self.activ_f == 'Relu':
            return a * (a > 0)
        if self.activ_f == 'Swish':
            return (a*1/(1+np.exp(-a)))
            #Return shape is same as input shape
            #This would be the z!
            
    def output_activ_func(self,y:np.ndarray):
        if self.out_activ_f == 'linear':
            return y
        if self.out_activ_f == "Softmax":
            return (np.exp(-y) / np.sum(np.exp(-y), axis=0))
            #Assuming classes are labelled (0,1,...k-1)
            #return shape is same as input shape
            #For the last outputs namely "y" we dont carry out "the h func".
    
    # derivative of the activation function
    def activation_derivative(self, a):
        #essentially inputting the "a" vector (i think its (f,1))
        if self.activ_f == "Logistic Sigmoid":
            return (1 - (np.tanh(a))**2)
        if self.activ_f == "Sigmoid":
            return (1)/(1+np.exp(a))*(1-(1/(1+np.exp(a))))
        if self.activ_f == "Relu":
            return 1 * (a>0)
        if self.activ_f == "Swish":
            return (a*1/(1+np.exp(-a))) + (1/(1+np.exp(-a))) * (1 - (a*1/(1+np.exp(-a))))

    #First lets define the Convolve operation/Cross-Correlation bw 2 3D matrices
    def convolve3D(self,X1 :np.ndarray, X2 :np.ndarray,st,b):
        #Let X1 be (B,1,28,28)
        #b is the bias term
        #s will be the stride in each dimension a triplet
        #s[0] skip columns and s[1] skip rows and then s[2] is = 1 always
        #for now we assume that padding has been done apriori
        #Array X1 and Filter X2 match in the 3D
        #Note that in Numpy (2,4,3) is interpreted as 2 arrays of size (4,3)
        #3D is the first element makes it easy for us
        
        #We take say (1,28,28) and A filter (1,5,5) to give (24,24)
        #Then say 2 such filters
        #We finally get (2,16,16)
        #But this convolve3D only gives (16,16)
        ht = X1.shape[2]; wt = X1.shape[3]
        k1 = X2.shape[1]; k2 = X2.shape[2]
        Out_dim = [int(((ht-k1)/st[1]) + 1) , int(((wt-k2)/st[0]) + 1)]
        Output = np.zeros((X1.shape[0],Out_dim[0],Out_dim[1]))
        
        for i in range(0,Out_dim[0]):
            for j in range(0,Out_dim[1]):
                #np.multiply does the Hadamard product
                #We then take np.sum(resulting array)
                temp = np.multiply(X1[:,:,(st[1]*i):((st[1]*i)+k1),(st[0]*j):((st[0]*j)+k2)],X2)
                for s in range(X1.shape[0]):
                    Output[s][i][j] = np.sum(temp[s])
        return (Output + b)
        #returns final output array (input * filter) (a 2D matrix)
    
    #Forward pass of the CNN(only) for a single input image
    def forward_pass(self,X):
        #input X is a 3D array (B,1,28,28)
        Hidden_Outputs = [X] #Stores after activation
        Hidden_Outputs1 = [X] #Stores before activation
        #all outputs are directly stored as 3D arrays
        for s in range(self.c_l):
            f_map1 = self.convolve3D(Hidden_Outputs[-1], self.filters[s][0], self.stride[s], self.biases[s][0])
            #(B,24,24)
            temp_output = np.zeros((X.shape[0],self.f_count[s],f_map1.shape[1],f_map1.shape[2]))
            temp_output[:,0,:,:] = f_map1
            #(2,24,24)
            for r in range(1,self.f_count[s]):
                f_map = self.convolve3D(Hidden_Outputs[-1], self.filters[s][r], self.stride[s], self.biases[s][r])
                temp_output[:,r,:,:] = f_map
            #(2,24,24)
            Hidden_Outputs1.append(temp_output)
            Hidden_Outputs.append(self.activation_func(temp_output))
            #We append the ouput
  
        return (Hidden_Outputs[1:],Hidden_Outputs1[1:])
        #Each list of length c_l
        #returns entire information of the forward pass of CNN only
        
    def backpropagation(self, x:np.ndarray, y:np.ndarray):
        #x would be (B,1,28,28) y is an aeeay as well
        info_CNN = self.forward_pass(x) #Tuple
        (B4,u,v,w) = info_CNN[0][-1].shape
        #First we need to get the deltas/derivatives wrt the MLP layer
        #Hidden_Outputs[-1] must be flattened and passed into a MLP
        #This is the fully connected layer
        MLP_input = np.c_[info_CNN[0][-1].reshape((B4,u*v*w)),np.ones(B4)]

        #(6,3,3) becomes (B2,54)
        info_MLP = NN.backpropagation(MLP_input,y)
        #This contains all information about the forward pass of MLP
        #a tuple with (derivative,info[-1],deltas[0]) from MLP
        #deltas[0] is a list of length 10
        
        Derivative_filters = [0 for i in range(self.c_l)]
        Derivative_biases = [0 for i in range(self.c_l)]
        #Derivative_filters[0] will be a list containing derivative wrt each filter as an array
        #[some (1,3,3) array,another (1,3,3) array] for 2 such filters of dim (1,3,3) in 0 meaning first layer
        #Derivative_biases[0] will be [1,2] for 2 filters in 1st layer
        
        deltas = [0] * self.c_l
        #Each element of deltas stores all deltas for that layer wrt "Output of that layer"
        #deltas[0] will be array (2,24,24) corresponding to output dimension of first layer
        
        #Lets compute delta for last layer of CNN
        #deltas[-1] will be of shape (6,3,3)
        deltas[-1] = np.zeros((B4,self.o_d[0],self.o_d[1],self.o_d[2]))
        A1 = info_MLP[2]
        #A1 which is deltas[0] of MLP is a list of length 4
        for f in range(self.o_d[0]):
            for i_ in range(self.o_d[1]):
                for j_ in range(self.o_d[2]):
                    #NN.weights[0] is an array
                    index = (f * self.o_d[1] * self.o_d[2]) + (i_ * self.o_d[2]) + (j_)
                    A2 = [NN.weights[0][index,r] for r in range(A1.shape[1])]
                    deltas[-1][:,f,i_,j_] = np.dot(A1,A2)
                    
        
        #Compute deltas for other layers 
        for l_ in reversed(range(self.c_l-1)):
            A1 = deltas[(l_ + 1)] #an array
            A3 = self.activation_derivative(info_CNN[1][(l_ + 1)])
            
            (B3,a0,b0,c0) = info_CNN[0][l_].shape
            (B3,a1,b1,c1) = info_CNN[0][(l_ + 1)].shape
            deltas[l_] = np.zeros((B3,a0,b0,c0))
            for f in range(a0):
                α = f
                for i_ in range(b0):
                    for j_ in range(c0):
                        A2 = np.zeros((a1,b1,c1))
                        #Below loop is to compute A2
                        for kk in range(a1):
                            for ii in range(b1):
                                for jj in range(c1):
                                    β = (i_ - (self.stride[l_ + 1][1])*ii)
                                    γ = (j_ - (self.stride[l_ + 1][0])*jj)
                                    if (β in range(self.f_sizes[l_ + 1][1])) and (γ in range(self.f_sizes[l_ + 1][2])):
                                        A2[kk,ii,jj] = self.filters[l_ + 1][kk][α][β][γ]
                                    else:
                                        A2[kk,ii,jj] = 0
                        A4 = np.multiply(A1,A3)
                        A5 = np.multiply(A4,A2)
                        #We are using broadcasting
                        t0 = np.sum(A5,axis=3) #(B3,?)
                        t1 = np.sum(t0,axis=2) #(B3,??)
                        t2 = np.sum(t1,axis=1) #(B3,)
                        break
                        deltas[l_][:,f,i_,j_] = t2
        
        #Now we compute derivative wrt filters and biases
        
        #derivative wrt filter
        for l in range(self.c_l):
            #l iterates over layer
            temp_d = [0] * self.f_count[l]
            temp_d_f = np.zeros((self.f_sizes[l][0],self.f_sizes[l][1],self.f_sizes[l][2]))
            for f in range(self.f_count[l]):
                #f iterates over number of filters in that layer
                B1 = deltas[l][:,f] 
                B2 = self.activation_derivative(info_CNN[1][l][:,f])
                C1 = np.multiply(B1,B2)
                for k in range(self.f_sizes[l][0]):
                    #k iterates over 3D               
                    for i in range(self.f_sizes[l][1]):
                        #i iterates over 1D/rows
                        for j in range(self.f_sizes[l][2]):
                            #j iterates over 2D/columns
                            #C2 has to be constructed for each fixed k,i,j
                            C2 = np.zeros((B4,B2.shape[1],B2.shape[2]))
                            for r in range(B2.shape[1]):
                                for s in range(B2.shape[2]):
                                    if l!=0:
                                        C2[:,r,s] = info_CNN[0][l-1][:,k,(self.stride[l][1]*r)+i,(self.stride[l][0]*s)+j]
                                    else:
                                        C2[:,r,s] = x[:,k,(self.stride[l][1]*r)+i,(self.stride[l][0]*s)+j]
                            temp_d_f[k,i,j] = np.sum(np.multiply(C1,C2))  
                temp_d[f] = temp_d_f/B4
                #We already divide it by N
            Derivative_filters[l] = temp_d
            
        #derivative wrt biases
        for l_ in range(self.c_l):
            #l iterates over layer
            Derivative_biases[l_] = [0] * self.f_count[l_]
            for f in range(self.f_count[l_]):
                # f iterates over filter
                D1 = deltas[l_][:,f] 
                D2 = self.activation_derivative(info_CNN[1][l_][:,f])
                Derivative_biases[l_][f] = np.sum(np.multiply(D1,D2))/B4
                #We already divide it by N
              
        #All derivatives are now computed
        return (Derivative_filters,Derivative_biases,info_MLP[1],info_MLP[0],MLP_input)
        # tuple is (D_f_CNN,D_b_CNN,final output of MLP,D_MLP)
        # D_MLP is a list containing arrays


#total training data = 60000
A = 70000 #total data
B = 6000 #training data
C = 1000 #test data
d = 784 #feature vector
f = 1 #label vector
k = 10 #10 classes 0 to 9 numbers like kanji
d1 = [1,28,28]
d2 = [6,3,3] #feature vector
d3 = 54

train_imgs = np.load(r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 2\Data given\kmnist-train-imgs.npz")['arr_0'][:B].reshape((B,1,28,28))/255
train_labels = np.load(r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 2\Data given\kmnist-train-labels.npz")['arr_0'][:B].reshape((B,1))
test_imgs = np.load(r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 2\Data given\kmnist-test-imgs.npz")['arr_0'][:C].reshape((C,1,28,28))/255
test_labels = np.load(r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 2\Data given\kmnist-test-labels.npz")['arr_0'][:C].reshape((C,1))

HL = 3
NEURON = [10,10]
NN = NeuralNetwork(d3,k,HL,NEURON,"Sigmoid","Cross-Entropy","Softmax")

CL = 3
FS = [[1,6,6],[2,5,5],[3,4,4]]
FC = [2,3,6]
STRIDE = [[2,2,1],[1,1,1],[2,2,1]]
CNN = CNN(d1,d2,CL,"Logistic Sigmoid","Cross-Entropy","Softmax",FC,FS,2,STRIDE)

learning_rate = 0.5

CE_list = []
Accuracy_train_list = []
Accuracy_test_list = [] 

def gradient_descent(g,lr):
    for dummy in range(g):
        sum = 0
        sum1 = 0
        sum2 = 0
        
        P = CNN.backpropagation(train_imgs,train_labels)
        Total_Derivative_CNN_f = P[0]  
        Total_Derivative_CNN_b = P[1]
        Total_Derivative_MLP = [x/B for x in P[3]]
        YP = CNN.output_activ_func(P[2]) 
        for s in range(B):
            prediction = np.argmax(YP[s])
            y1 = int(train_labels[s])
            if prediction != y1:
                sum1 += 1
            sum += (-np.log(YP[s][y1]))

        #Lets write a code for simultaneously getting the testing accurcay as well
        CNN_yp = CNN.forward_pass(test_imgs)[0][-1]
        MLP_input = np.c_[CNN_yp.reshape((C,d3)),np.ones(C)]
        #(6,3,3) becomes (54,)
        MLP_yp = NN.forward_pass(MLP_input)[-1]
        yp = CNN.output_activ_func(MLP_yp) 
        #returns class_prob (k,1)
        for s in range(C):
            prediction = np.argmax(yp[s])
            y1 = int(test_labels[s])
            if prediction != y1:
                sum2 += 1
                
        #Actual gradient updation step
        for s1 in range(NN.h_l):
            NN.weights[s1] -= ((lr) * (Total_Derivative_MLP[s1]))
        for s2 in range(CNN.c_l):
            for f in range(CNN.f_count[s2]):
                CNN.filters[s2][f] -= (lr * Total_Derivative_CNN_f[s2][f])
                CNN.biases[s2][f] -= (lr * Total_Derivative_CNN_b[s2][f])
                
        CE_list.append(sum/B)
        print(CE_list[-1])
        Accuracy_train_list.append((B-sum1)*100/B)
        print(Accuracy_train_list[-1])
        Accuracy_test_list.append((C-sum2)*100/C)
        print(dummy)

#-----------------------#
gradient_descent(500,learning_rate)
#Plotting
iterations = range(len(CE_list))
plt.scatter(iterations, CE_list, label='CE')
plt.title('CE vs Gradient Steps')
plt.xlabel('Gradient iterations')
plt.ylabel('CE')
plt.legend()
plt.show()

iterations = range(len(CE_list))
plt.scatter(iterations, Accuracy_test_list, label='TestingAccuracy')
plt.scatter(iterations, Accuracy_train_list, label='TrainingAccuracy')
plt.title('Accuracy vs Gradient Steps')
plt.xlabel('Gradient iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#1 data point taking 2.4 sec

# CNN_model = CNN.filters
# MLP_model = NN.weights

