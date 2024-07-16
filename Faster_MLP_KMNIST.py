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
        self.weights[0] = (np.random.randn(self.i_d + 1, self.neurons[0]) / np.sqrt(self.i_d + 1)*0.1)
        #(i_d + 1,M1)
        for i in range(1, self.h_l - 1):
            self.weights[i] = (np.random.randn(self.neurons[i-1] + 1, self.neurons[i]) / np.sqrt(self.neurons[i-1] + 1)*0.1)
        self.weights[self.h_l - 1] = (np.random.randn(self.neurons[-1] + 1, self.o_d) / np.sqrt(self.neurons[-1] + 1)*0.1)

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
        return (Derivative,info[-1])
    
    

#PRNN ASGN 1
#Grp No 8

#PROBLEM 5
#MULTICLASS CLASSIFICATION

#Same codes for PROBLEM 4 just replace the intial variables
train_imgs = np.load(r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 2\Data given\kmnist-train-imgs.npz")['arr_0']
train_labels = np.load(r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 2\Data given\kmnist-train-labels.npz")['arr_0']
test_imgs = np.load(r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 2\Data given\kmnist-test-imgs.npz")['arr_0']
test_labels = np.load(r"C:\Users\sahap\OneDrive\Desktop\iisc materials\Mathematics\sem 4\PRNN\Asgns\Asgn 2\Data given\kmnist-test-labels.npz")['arr_0']

#total training data = 60000
A = 70000 #total data
B = 60000 #training data
C = 10000 #test data
d = 784
f = 1 #label vector
k = 10 #10 classes 0 to 9 numbers like kanji

#data_array_F contains entire feature space data (A,d)
#data_array_f contains the training data features (B datapoints)

data_array_f = train_imgs.reshape((B,784)) #(B,d)
data_array_l = train_labels.reshape((B,1)) #(B,1)

#Test data
test_data_f = test_imgs.reshape((C,784))
test_data_l = test_labels.reshape((C,1))
data_array_f_a = np.c_[data_array_f,np.ones(B)]
test_data_f_a = np.c_[test_data_f,np.ones(C)]

HL = 3
NEURON = [4,5]

NN = NeuralNetwork(d,k,HL,NEURON,"Swish","Cross-Entropy","Softmax")
learning_rate = 0.0000000001

#-------------------------#
CE_list = []
Accuracy_train_list = []
Accuracy_test_list = []
def gradient_descent(g):
    for dummy in range(g):
        sum = 0
        sum1 = 0
        sum2 = 0
        P = NN.backpropagation(data_array_f_a,data_array_l)
        YP = NN.output_activ_func(P[1])
        for s in range(B):
            prediction = np.argmax(YP[s])
            y1 = int(data_array_l[s])
            if prediction != y1:
                sum1 += 1
            sum += (-np.log(float(YP[s][y1])))
        Total_Derivative = [t/B for t in P[0]]
        #Lets write a code for simultaneously getting the testing accurcay as well
        P = NN.forward_pass(test_data_f_a)
        YP = NN.output_activ_func(P[-1])
        for s in range(C):
            prediction = np.argmax(YP[s])
            y1 = int(test_data_l[s])
            if prediction != y1:
                sum2 += 1
        for s in range(NN.h_l):
            NN.weights[s] -= (learning_rate * Total_Derivative[s])
        CE_list.append(sum/B)
        Accuracy_train_list.append((B-sum1)*100/B)
        Accuracy_test_list.append((C-sum2)*100/C)
        print(dummy)

gradient_descent(200)
#Plotting
iterations = range(len(CE_list))
plt.scatter(iterations, CE_list, label='ER')
plt.title('CE vs Gradient Steps')
plt.xlabel('Gradient iterations')
plt.ylabel('CE')
plt.legend()
plt.show()

iterations = range(len(CE_list))
plt.scatter(iterations, Accuracy_train_list, label='TrainingAccuracy')
plt.scatter(iterations, Accuracy_test_list, label='TestingAccuracy')
plt.title('Accuracy vs Gradient Steps')
plt.xlabel('Gradient iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


