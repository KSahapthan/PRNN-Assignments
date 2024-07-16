import scipy
import numpy as np
import matplotlib.pyplot as plt

#We use MLP from previous code for FFNN(from Faster_CNN)
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
        y4 = (b @ self.weights[self.h_l - 1]) #(B,o_d)
        #We act to get the final output of the neural network
        #There is no nonlinear transformation
        Act_list[self.h_l - 1] = y4
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
    def backpropagation(self, x:np.ndarray, y:np.ndarray,BB):
        #BB for batchsize
        #x is entire data set (BB,d+1) and y is (BB,1)
        #y from the training set is needed only to compute the output delta
        info = self.forward_pass(x)
        Derivative = [0 for i in range(self.h_l)]
        deltas = [np.zeros((BB,self.neurons[i])) for i in range(self.h_l - 1)] #for all hidden layers except last
        deltas.append(np.zeros((BB,self.o_d))) #for the last one

        #First we compute delta for all the layers and all the parameters
        # calculate delta for output layer
        info2 = (self.output_activ_func(info[-1])) #After acting the output_activation_func
        #input is (B,o_d) also same shape
        for r in range(self.o_d):
            info3 = info2[:,r]
            #Take all rows and columns r
            for s in range(BB):
                if int(y[s][0]) == r:
                    deltas[-1][s][r] = (-(info3[s]-1))
                else:
                    deltas[-1][s][r] = (-(info3[s]))
            #dk = "corresponding derivative for that loss"

        # calculate delta for all the hidden layers
        for i in reversed(range(self.h_l - 1)):
            info1 = self.activation_derivative(info[i])
            #(B,??) #h'(a) as an array
            for j in range(self.neurons[i]): #accounted for bias as well
                for s1 in range(BB):
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
                            deriv[r][s] = np.dot(np.ones((BB,)),deltas[i][:,s]) #we have appended 1
                Derivative[i] = deriv
        deriv_1 = np.zeros((self.neurons[-1]+1,self.o_d,))
        for r in range(self.neurons[-1]+1):
            if r != self.neurons[-1]:
                for s in range(self.o_d):
                    deriv_1[r][s] = np.dot(info[-2][:,r],deltas[-1][:,s])
            else:
                for s in range(self.o_d):
                    deriv_1[r][s] = np.dot(np.ones((BB,)),deltas[-1][:,s]) #we have appended 1
        Derivative[-1] = deriv_1
        return (Derivative,info[-1],deltas[0])

#----------------------------------------#
#First lets make a self attention block as a class with a single-head
#We give an array as input(sentence)
#X = input = (N0,n,d) 
#N0 is #SENTENCES and n is LONGEST SENTENCE IN DATASET
#d depends on RICHNESS OF VOCABULARY REQD/EMBEDDING USED
#Each input will be a 3D array with n denoting no of inputs(words)
#Remaining 1D component is dimension of each input in array form
#Parameters to be initialised are the W_Q,W_K,W_V 

class self_attention_block:
    #Token Length is same as the input dimensions
    #(Same as length of longest sentence) = n
    def __init__(self,Q_shape,K_shape,V_shape):
        self.Q_shape = Q_shape #(d,dk)
        self.K_shape = K_shape #(d,dk)
        self.V_shape = V_shape #(d,dk)
        #initialisation of the QKV Matrices
        self.W_Q = np.random.randn(Q_shape[0],Q_shape[1])/np.sqrt(np.prod(Q_shape))
        self.W_K = np.random.randn(K_shape[0],K_shape[1])/np.sqrt(np.prod(K_shape))
        self.W_V = np.random.randn(V_shape[0],V_shape[1])/np.sqrt(np.prod(V_shape))

    def SA_score(self,X):
        #Function to calculate SA_score
        #X is the input of entire dataset
        #Dimensions of input is already predetermined by Q,V,K
        (N0,n,d) = X.shape
        #QKV vectors are all same dimensions.Let it be dk
        #Compute QKV vectors
        Q = (X @ self.W_Q) #(N0,n,dk)
        K = (X @ self.W_K) #(N0,n,dk)
        V = (X @ self.W_V) #(N0,n,dk)
        dk = Q.shape[2]
        #dk is dimension of key vectors of each word in each sentence
        y2 = ((Q @ np.transpose(K,(0,2,1)))/np.sqrt(dk)) #(N0,n,n)
        y1 = np.exp(y2) #(N0,n,n) 
        #We normalise to make row sums 1 for each datapoint
        for i in range(N0):
            for j in range(n):    
                y1[i,j] /= (np.sum(y1[i,j]))
                
        z = (y1 @ V) #(N0,n,dk)
        return (z,Q,K,V,y1)
        #z has the SA score to be passed into the FFNN
        #We normalise by n and pass into MLP only once as A CONTEXT VECTOR
        #To get the corresponding output class for that sentence

class Encoder:
    #For 1 single encoder 
    def __init__(self,FFNN_data,SA_size):
        #SA_size is just a tuple of (Q_shape,K_shape,V_shape)
        #FFNN_data is (i_d, o_d, h_l, neurons, activ_f, loss_f, o_act_f)
        self.SA = self_attention_block(*SA_size)
        self.FFNN = NeuralNetwork(*FFNN_data)
       
class Transformer:
    def __init__(self,FFNN_data,SA_size,T_o_d):
        #o_d is that of the last FFNN/MLP
        #FFNN_data and SA_size are both tuples
        #We accordingly create as many Encoder classes
        self.E = Encoder(FFNN_data,SA_size)
        #only 1 encoder
        self.T_o_d = T_o_d
    
    def T_forward(self,X:np.ndarray,Y:np.ndarray,indicator:int,bs):
        #We do the forward pass FOR ALL SENTENCES    
        inp = X #(N0,n,d)
        n = X.shape[1]
        N0 = X.shape[0]
        Outputs_SA = []
        Outputs_MLP = []
        if indicator == 1:
            #We do Back prop as well
            SA_info = self.E.SA.SA_score(inp)
            z = SA_info[0]
            #Output of SA of Encoder of that layer
            #This has to be passed into MLP of Encoder 
            #(treating as data points) and z is (N0,n,dk)
            #Input to MLP must be (N0,dk+1) so we have to iterate
            h = np.sum(z,axis=1)/n #(N0,dk)
            z1 = np.c_[h,np.ones(N0)]
            #z1 will be directly passed into FFNN
            #We pass input and true labels of each sentence
            h1 = self.E.FFNN.backpropagation(z1,Y,bs)
            #h1 contains (Derivative,info[-1],deltas[0])
            Outputs_SA.append(SA_info) 
            Outputs_MLP.append(h1)
            #We append a tuple
            return (Outputs_SA,Outputs_MLP)
        
        if indicator == 0:
            #Only forward pass for testing/validation
            SA_info = self.E.SA.SA_score(inp)
            z = SA_info[0]
            h = np.sum(z,axis=1)/n #(N0,dk)
            z1 = np.c_[h,np.ones(N0)]
            h1 = self.E.FFNN.forward_pass(z1)
            #We append a tuple
            return (h1)
            
    def T_backprop(self,X:np.ndarray,Y:np.ndarray,bs):
        (N0,n,d) = X.shape
        #First step is to compute the forward pass
        #For MLP forward pass and backprop are done together as in the code
        f_info = self.T_forward(X,Y,1,bs)
             
        #Now we have to use chain rule to derive gradient for SA_block
        Q = f_info[0][-1][1] #(N0,n,dk)
        K = f_info[0][-1][2] #(N0,n,dk)
        V = f_info[0][-1][3] #(N0,n,dk)
        H = f_info[0][-1][4] #(N0,n,n) #softmax intermediate ouput 
        dk = Q.shape[2]
        gamma = 1/np.sqrt(dk)
        
        temp_link = (f_info[1][-1][2] @ (self.E.FFNN.weights[0]).T)[:,0:-1] #(N0,dk)
        Z1 = np.zeros((N0,n,dk))
        Z1[:,:,:] = temp_link.reshape((N0,1,dk))
        H1 = np.multiply(H,Z1 @ np.transpose(V,(0,2,1))) #(N0,n,n)
        H2 = np.zeros((N0,n,1)) #(N0,n,1)
        for i in range(n):
            for j in range(N0):
                H2[j,i] = np.sum(H1[j,i])
        
        #Apply formula derived painstakingly
        #Calculate final derivative wrt ALL DATAPOINTS/ALL SENTENCES/INCORPORATE N0
        #Derivative wrt W_V
        d_V = (np.transpose(X,(0,2,1))) @ (np.transpose(H,(0,2,1))) @ (Z1)
        #Derivative wrt W_K
        d_K = gamma * (np.transpose(X,(0,2,1)) @ np.transpose(H1,(0,2,1)) @ (Q - (H @ Q)))
        #Derivative wrt W_Q
        d_Q = gamma * (np.transpose(X,(0,2,1)) @ (H1 - np.multiply(H,H2)) @ K)
        return(d_Q,d_K,d_V,f_info[1][-1][0],f_info[1][-1][1])
        #FFNN Derivative is returned as a list as usual
        #Returns total derivatives of both SA and FFNN and final output of FFNN
        #Final ouptut is #(N0,T_o_d)

#total training data = 22710
A = 26178 #total data
B = 22710 #training data
C = 3468 #test data

#-------------------------------------------------------------#
#Converting data into processable form
from sklearn.decomposition import PCA
pca = PCA(n_components = 30)
import os
import cv2
f_path = r"C:\Users\sahap\OneDrive\Desktop\IISc materials\Extra Lectures Learnt\PRNN\Asgns\Asgn 3\Animal Images\raw-img"
subfolders = os.listdir(f_path)
#list of strings (10)
Common_Image_size = (1,30) #finally after PCA
data_array = []
#data_array[0] = ((10,10,3),0)
animal = 0
img_count = [4862, 2623, 1446, 2112, 3098, 1668, 1866, 1820, 4821, 1862]
for folder in subfolders:
    folder_path = os.path.join(f_path,folder)
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    temp_img = np.zeros((img_count[animal],67500))
    j = 0
    for image_path in image_paths:
        temp_img[j] = cv2.resize(cv2.imread(image_path),(150,150)).reshape((1,-1))
        j += 1
    temp_img /= 255
    pca_img = (pca.fit(temp_img)).transform(temp_img)
    print(pca_img.shape)
    #image paths is list of image names
    #image path is one image
    #each element of images in a tuple  ((225,300,3),0) [BGR and labels]
    # Normalize pixel values (divide by 255)
    data_array.append(pca_img)
    print("yes")
    animal += 1

data_array1 = []
for i in range(10):
    for j in range(img_count[i]):
        data_array1.append((data_array[i][j].reshape((1,30)),i))
        
np.random.shuffle(data_array1)
np.random.shuffle(data_array1)
#len(data_array1) = 26178
image_labels = [i[1] for i in data_array1]
image_array = [i[0] for i in data_array1]
#Each image of shape (1,30)
#[4862, 2623, 1446, 2112, 3098, 1668, 1866, 1820, 4821, 1862]
#Total images = 26718
#subfolders contains name of each image
#image_array contains all images
#We have to split into training and testing

working_image = np.array(image_array).reshape((A,30,1))
train_imgs = working_image[:B]
train_labels = np.array(image_labels[:B]).reshape((-1,1))
test_imgs = working_image[B:B+C]
test_labels = np.array(image_labels[B:B+C]).reshape((-1,1))

Random_shuffle_count = [0] * 10
for s in image_labels[:B]:
    Random_shuffle_count[s] += 1
#[4231, 2273, 1243, 1824, 2714, 1434, 1611, 1578, 4176, 1626]
    
X1 = train_imgs
X1 /= (np.max(np.abs(X1)))
X2 = test_imgs
X2 /= (np.max(np.abs(X2)))
##------------------------------------------------##
d = 1 
n = 30 #after PCA
dk = 30
i_d = dk #i_d of MLP = dk of SA
k = 10 #o_d of MLP corresponds to #classes
HL = 3
Neurons = [15,15]
TFMER = Transformer((i_d,k,HL,Neurons,"Relu","Cross-Entropy","Softmax"),((d,dk),(d,dk),(d,dk)),k)
lr = 5
batch_size = 250
no_of_batches = int(B/batch_size)

#---------------------#
CE_list = []
Accuracy_train_list = []
Accuracy_test_list = []
CM = np.zeros((12,12))
#CM has correct columns and predicted rows
Adjustment_factor = np.array([i/(B+C) for i in img_count])
#Adjustment_factor = np.ones((12,))/12

def gradient_descent(g,epoch,bs):
    #epochs are essentially number of gradient steps
    epoch_count = 0
    for _ in range(epoch):
        strt = 0
        stp = bs
        for dummy in range(g):
            #g iterates over batches
            #X1 and label_train is passed
            #X1 is numpy matrix
            #label_train is a list
            P = TFMER.T_backprop(X1[strt:stp],train_labels[strt:stp],bs)
            y = P[4] #(bs,o_d)
            (bs,o_d) = y.shape
            Total_Derivative_MLP = [x/bs for x in P[3]]
            Total_Derivative_WQ = np.sum(P[0],axis = 0)/bs
            Total_Derivative_WK = np.sum(P[1],axis = 0)/bs
            Total_Derivative_WV = np.sum(P[2],axis = 0)/bs  
            
            #Actual gradient updation step
            TFMER.E.SA.W_Q -= (lr * Total_Derivative_WQ)
            TFMER.E.SA.W_K -= (lr * Total_Derivative_WK)
            TFMER.E.SA.W_V -= (lr * Total_Derivative_WV)
            for s2 in range(HL):
                TFMER.E.FFNN.weights[s2] -= (lr * Total_Derivative_MLP[s2]) 
            strt += bs
            stp += bs
      
        #Lets write a code for simultaneously getting the testing accurcay as well
        #Testing_Accuracy after 1 epoch
        sum2 = 0
        P = TFMER.T_forward(X2,test_labels,0,C)
        YP = np.multiply(TFMER.E.FFNN.output_activ_func(P[-1]),Adjustment_factor)
        #We have to make some adjustments for the labels posterior distribution
        for s in range(C):
            prediction = np.argmax(YP[s])
            y1 = int(test_labels[s][0])
            if prediction != y1:
                sum2 += 1
        #Training_Accuracy after 1 epoch
        sum = 0
        sum1 = 0
        P = TFMER.T_forward(X1,train_labels,0,B)
        YP = np.multiply(TFMER.E.FFNN.output_activ_func(P[-1]),Adjustment_factor)
        for s in range(B):
            prediction = np.argmax(YP[s])
            y1 = int(train_labels[s][0])
            if prediction != y1:
                sum1 += 1 
                CM[prediction][y1] += 1  
            else:
                CM[y1][y1] += 1
            sum += (-np.log(float(YP[s][y1])))

        Accuracy_test_list.append((C-sum2)*100/C)
        Accuracy_train_list.append((B-sum1)*100/B)
        CE_list.append(sum/B)
        epoch_count += 1
        print("Epoch count:",epoch_count)
        print("Train_Accuracy:",Accuracy_train_list[-1])
        print("Test_Accuracy:",Accuracy_test_list[-1])
        print("CE:",CE_list[-1])

#-----------------------#
#Testing and PLotting
epochs = 5
gradient_descent(no_of_batches,epochs,batch_size)

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

fig, ax = plt.subplots()
min_val, max_val = 0, 12
ax.matshow(CM, cmap=plt.cm.Blues)
for i in range(12):
    for j in range(12):
        c = CM[j,i]
        ax.text(i, j, str(c), va='center', ha='center')



        
        

    

        


        

