import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import time


# The linear regression model

class LinearRegression():

    def __init__(self):
        self.x = None
        self.y = None

        """
        These variables are being used for the linear regression neural network
        like the weights and bais or intercept 
        """
        self.parameter_cache = []
        self.weight_matrix = None
        self.intercept = None
        self.dcostdm = None
        self.dcostdc = None

    """
    name: setUp

    This function helps put the data in the proper array form 
    so when it executes the the linear regression equation 
    runs the data
    """
    def setUp(self, X, Y):
        self.x = np.array(X).reshape(-1,1)
        self.y = np.array(Y).reshape(-1,1)
        print(self.x.shape)
        x_shape = self.x.shape
        num_var = x_shape[1]

        return (x_shape, num_var)

    """
    name: train

    This function is used to train the linear regression 
    model with the data the user provided
    """
    def train(self, X, Y, iterations = 50):
        (x_shape, num_var) = self.setUp(X, Y)

        self.weight_matrix = np.random.normal(0, 1, (num_var, 1))
        self.intercept = np.random.rand(1)

        for i in range(iterations):

            #partial derivative of cost w.r.t the weights
            self.dcostdm = np.sum(np.multiply(((np.matmul(self.x,self.weight_matrix)+self.intercept)-self.y),self.x))*2/x_shape[0]
            
            #partial derivative of cost w.r.t the intercept
            self.dcostdc = np.sum(((np.matmul(self.x,self.weight_matrix)+self.intercept)-self.y))*2/x_shape[0] 

            #updating the weights with the calculated gradients
            self.weight_matrix -= 0.1*self.dcostdm

            #updating the weights with the calculated gradients
            self.intercept -= 0.1*self.dcostdc

        self.parameter_cache.append(np.array((self.weight_matrix,self.intercept)))
        return self.weight_matrix,self.intercept,self.parameter_cache
    
    """
    name: predict

    This function is used to predict the weight of a person with the given height
    """
    def predict(self, X):
        prediction = np.matmul(np.array(X).reshape(-1,1), self.weight_matrix) + self.intercept
        return prediction




