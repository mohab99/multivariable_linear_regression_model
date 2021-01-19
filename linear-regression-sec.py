#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all pachage will be needed during assignment
import numpy as np
# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd


# In[2]:


def get_training_data(file_path):
    data=pd.read_table(file_path,sep=',',header=None)
    Y_data=data.iloc[:,-1].to_numpy().reshape(-1,1)
    X_data=data.iloc[:,:-1].to_numpy()
    return X_data,Y_data


# In[3]:


#test
X,Y = get_training_data('ex1data2.txt')
X.shape
X1 = X[:,0].reshape(-1,1)
X2 = X[:,1].reshape(-1,1)
print(X.shape, "\n")


# ### expected output
# **(47,2)**

# In[4]:


fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(X1, X2, Y)
pyplot.show()


# In[5]:


def initial_parameters(number_features):
    w=np.zeros((number_features ,1))
    b=np.zeros((1,1))
    parameters = {"w":w,
                  "b":b}  
    return parameters


# In[6]:


#test
par=initial_parameters(5)
assert par["w"].shape == (5,1)
print(par["w"])


# ### expected_output 
# **[[0.]
#  [0.]
#  [0.]
#  [0.]
#  [0.]]**

# In[7]:


x_max = X.max(axis=0)
x_min = X.min(axis=0)
def normalize (X):
    X_norm = (X-x_min) /(x_max-x_min)
    mu = np.mean(X, axis=0)
    sigma = np.std(X,axis=0)
    
    return X_norm,mu,sigma


# In[8]:


#test
_,mu,_=normalize(X)
print(mu)
Xnorm = normalize(X)[0]


# ### expexted output
# [[2000.68085106    3.17021277]]
# 

# In[9]:


def compute_cost(X,Y,parameters):
    
    W=parameters["w"]
    b=parameters["b"]
    Y_hat=(np.matmul(X, W))+b
    m=Y.shape[0]
    cost=np.sum((Y_hat-Y)**2,axis=0)/(2*m)

    return float(cost)


# In[10]:


#test
cost=compute_cost(X,Y,{"w":np.ones((2,1)),"b":3})
print(cost)


# ### expected output
# **64827520487.180855**

# In[11]:


def update_parameters(X,Y,parameters,learning_rate):
                      
    W=parameters["w"]
    b=parameters["b"]
    Y_hat=(np.matmul(X, W)+b)
    m=X.shape[0]
    dW=np.sum((Y_hat-Y)*X,axis=0,keepdims=True)/(m)
    db=np.sum((Y_hat-Y),axis=0,keepdims=True)/(m)    
    W=(W-learning_rate*dW)[0:1,:].T
    b=b-learning_rate*db
    parameters = {"w":W,"b":b} 
    
    return parameters


# In[12]:


#test
par=update_parameters(X,Y,{"w":np.ones((2,1)),"b":3},.01)
print(par["w"].shape)
print("x",X.shape)

print(par.values())


# ### expected output
# **(1,2)**
# 
# **dict_values([array([[7595757.13702128,   11137.73553191]), array([[3387.05808511]])])**

# In[13]:


#cost history is a list contain cost every 100 iteration
def linear_regression(data_file,learning_rate=.1,iteration=1500,print_cost=False):

    X,Y =get_training_data(data_file)
    parameters=initial_parameters(X.shape[1])
    cost_history=[]
    Xnorm = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    for i in range(0,iteration+1):
        parameters=update_parameters(Xnorm, Y, parameters, learning_rate)
        if print_cost and i%200==0:
            cost=compute_cost(Xnorm, Y, parameters)
            cost_history.append(cost)
            print("cost after iteration {} is {}".format(i,cost))
    
    return parameters,cost_history


# In[14]:


#test
parameters,cost_history=linear_regression('ex1data2.txt',learning_rate=0.1,iteration=2000,print_cost=True)
#the cost doesn't have the exact expected value after each number of iteration 
#but they both reach the same minimum value eventually "4.4769..."


# - Cost after iteration 0: 6.737190
# - Cost after iteration 100: 5.476363
# - Cost after iteration 200: 5.173635
# - Cost after iteration 300: 4.962606
# - Cost after iteration 400: 4.815501

# In[15]:


pyplot.plot([x for x in range(0,2001,200)],cost_history, '-')


# In[16]:


def predict (x,parameters):
    
    w=parameters["w"]
    b=parameters["b"]
    x= x/x_max
    y=np.matmul(x,w)+b  
    
    return y


# In[17]:

        #ftSqr, rooms
y=predict([4900, 3],parameters)
y #price

