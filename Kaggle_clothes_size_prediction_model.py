#clothes size prediction
import pandas as pd
import numpy as np

def load_dataset():
    
    #read data from csv file using pandas
    data_temp = pd.read_csv(r'C:\coding\kaggle_clothes_size_prediction\final_test.csv')
 
    data_temp = data_temp[~pd.isnull(data_temp).any(axis=1)]
    
    #convert this panda dataframe to a numpy array
    data = data_temp.to_numpy()
    
    X = np.delete(data,3,axis =1)
    Y = np.delete(data,[0,1,2],axis = 1)
    
    x_train = np.delete(X,np.s_[95787:],axis = 0)
    y_train = np.delete(Y,np.s_[95787:],axis = 0)
    x_test = np.delete(X,np.s_[:95787],axis = 0)
    y_test = np.delete(Y,np.s_[:95787],axis = 0)

    return x_train.T , y_train.T , x_test.T , y_test.T

#Activation functions
#tanh(x) for 1st layer and the hidden layers
def tanh(Z):
    A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    return A

#softmax for the last layer as we need to classify for a single label from multiple classes
def Softmax(Z):
    shift_Z = Z - np.max(Z,axis = 0,keepdims = True)              
    exps = np.exp(shift_Z)
    A = exps / np.sum(exps,axis = 0,keepdims = True)
    return A

#initialize parameters w and b
def initialize_parameters(layers_dim):
    L = len(layers_dim);
    parameters = {};
    for l in range (1,L):
        parameters['W'+str(l)] = np.random.randn(layers_dim[l],layers_dim[l-1]) #/ np.sqrt(layers_dim[l-1]);
        parameters['b'+str(l)] = np.zeros((layers_dim[l],1));

    return parameters

#we need to convert y into one hot for classification
def convert_to_one_hot(y,C):
    Y=np.eye(C)[y].T
    
    return Y.reshape((C,y.shape[1]))                  

def convert_to_int(y):
    L=y.shape[1]
    for l in range(0,L):
        if(y[0][l]=='XXS'):
            y[0][l]=0
        elif(y[0][l]=='S'):
            y[0][l]=1
        elif(y[0][l]=='M'):
            y[0][l]=2
        elif(y[0][l]=='L'):
            y[0][l]=3
        elif(y[0][l]=='XL'):
            y[0][l]=4
        elif(y[0][l]=='XXL'):
            y[0][l]=5
        elif(y[0][l]=='XXXL'):
            y[0][l]=6

    return y.astype(int)

def linear_forward(A_prev,w,b,activation):
    Z = np.dot(w,A_prev)+b;
    if(activation == 'tanh'):
        A = tanh(Z);
    elif(activation == 'Softmax'):
        A = Softmax(Z);

    linear_cache = (Z,w,b)
    activation_cache = A_prev            
    cache = (linear_cache,activation_cache)
    
    return A,cache

def linear_L_forward(x_train,parameters):
    x_train = x_train.astype(np.float)
    L = len(parameters)//2;                  
    A_prev = x_train;
    caches = []
    for l in range(1,L):
        A,cache = linear_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],'tanh')
        A_prev = A
        caches.append(cache)

    AL,cache = linear_forward(A_prev,parameters['W'+str(L)],parameters['b'+str(L)],'Softmax')
    caches.append(cache)
    
    return AL,caches

def cost(Y,AL):
    m = Y.shape[1];
    cost_ = -np.sum(Y*np.log(AL+1e-8))             
    return cost_/m

def back_sigmoid(AL,dAL):
    dZ = []
    m = AL.shape[1]
    c = AL.shape[0]            #no. of classes
    I = np.eye(c)               #Identity matrix of size (c,c)

    for l in range(0,m):
        AL_temp = AL[:,l]
        AL_temp = AL_temp.reshape((c,1))                
        dAL_temp = dAL[:,l]
        dAL_temp = dAL_temp.reshape((c,1))
        dZ_temp = np.dot(dAL_temp.T , AL_temp*(I - AL_temp.T))

        dZ.append(dZ_temp)

    dZ = np.array(dZ)                    
    dZ = dZ.T
    dZ = dZ.reshape(AL.shape[0],dAL.shape[1])
  
    
    return dZ

def back_tanh(A,dA):
    temp = 1-(A*A)                 #this is dA/dZ

    dZ = dA*temp

    return dZ

def linear_backward(m,A,W,dA,A_prev,activation):
    
    if(activation == 'back_sigmoid'):
        dZ = back_sigmoid(A,dA)

    elif(activation == 'back_tanh'):
        dZ = back_tanh(A,dA)
    
    dW = np.dot(dZ/m,(A_prev.T)/m)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)

    dA_prev = np.dot(W.T,dZ)

    return dW,db,dA_prev

def linear_L_backward(Y,AL,caches):
    L = len(caches)
    m = Y.shape[1]
    grads = {}

    #for last layer
    linear_cache,activation_cache = caches[L-1]
    Z,W,b = linear_cache
    A_prev = activation_cache
    dAL = -Y/(AL+1e-8)
    grads['dW'+str(L)],grads['db'+str(L)],dA_prev = linear_backward(m,AL,W,dAL,A_prev,'back_sigmoid')

    for l in range(L-2,-1,-1):
        linear_cache,activation_cache = caches[l]
        Z,W,b = linear_cache
        dA  = dA_prev                       
        A = A_prev                           
        A_prev = activation_cache
        
        grads['dW'+str(l+1)],grads['db'+str(l+1)],dA_prev = linear_backward(m,A,W,dA,A_prev,'back_tanh')

    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2             
    for l in range (1,L+1):
        parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate*grads['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate*grads['db'+str(l)]

    return 

def accuracy(Y,AL):
    m = Y.shape[1]
    count = 0;
    Y=Y.astype(int)
    
    for l in range(0,m):
        if((Y[:,l]==AL[:,l]).all()):
            count+=1;

    return (count/m)*100
    
def model(X_train,Y_train):
    layers_dim = [3,30,15,7]
    learning_rate = 0.07
    parameters = initialize_parameters(layers_dim)
    Y_train = convert_to_int(Y_train)
    Y_train = convert_to_one_hot(Y_train,7)
    
    for epoch in range(1,501):
        AL,caches = linear_L_forward(X_train,parameters)
        
        grads = linear_L_backward(Y_train,AL,caches)
        update_parameters(parameters,grads,learning_rate)
        
        AL = (AL == AL.max(axis=0,keepdims=True)).astype(int)
        cost_ = cost(Y_train,AL)
        if(epoch%100 == 0):
            print('cost = ',cost_,'training accuracy after ',epoch,' epochs = ',accuracy(Y_train,AL),'\n')
            
    return parameters

x_train,y_train,x_test,y_test = load_dataset()
parameters = model(x_train/500,y_train)
AL,caches = linear_L_forward(x_test,parameters)
y_test = convert_to_int(y_test)
y_test = convert_to_one_hot(y_test,7)
AL = (AL == AL.max(axis=0,keepdims=True)).astype(int)
print('test accuracy = ',accuracy(y_test,AL),'\n')

    
    
    
    
    
    
    


    

