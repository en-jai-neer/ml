# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:13:22 2020

@author: JAI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],5)
        self.bias1      = np.ones([1 ,5])
        self.weights2   = np.random.rand(5,1)
        self.bias2      = np.ones([1,1])
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.layer1     = np.zeros([self.input.shape[0], 5])
        self.alpha      = 0.00025
        self.z1         = 0
        self.z2         = 0
    
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        return self.sigmoid(Z)*(1-self.sigmoid(Z))

    def relu(self, Z):
        return np.maximum(0,Z)
    
    def relu_derivative(self, Z):
        for i, row in enumerate(Z):
            for j, val in enumerate(row):
                if val>0:
                    Z[i][j] = 1
                else:
                    Z[i][j] = 0
        return Z
            

    def feedforward(self):
        self.z1 = np.dot(self.input, self.weights1)
        for i, row in enumerate(self.z1):
            row = row + self.bias1
        self.layer1 = self.relu(self.z1)
        self.z2 = self.bias2 + np.dot(self.layer1, self.weights2)
        self.output = self.sigmoid(self.z2)
    
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_output = 2*(self.y - self.output) * self.sigmoid_derivative(self.z2)
        d_weights2 = np.dot(self.layer1.T, d_output)
        d_layer1 = np.dot(d_output, self.weights2.T) * self.relu_derivative(self.z1)
        d_weights1 = np.dot(self.input.T, d_layer1)

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += self.alpha*d_weights1
        self.weights2 += self.alpha*d_weights2
        self.bias1 += self.alpha*np.sum(d_layer1, axis=0)
        self.bias2 += self.alpha*np.sum(d_output, axis=0)        

    
    def checkAccuracy(self, X_test, Y_test):
        ff1 = np.dot(X_test, self.weights1)
        for i, row in enumerate(ff1):
            row = row + self.bias1
        hidden1 = self.relu(ff1)
        Y_pred = self.sigmoid( self.bias2 + np.dot(hidden1, self.weights2))
        for i, val in enumerate(Y_pred):
            if(val>=0.5):
                Y_pred[i]=1
            else:
                Y_pred[i]=0
        
        Y_pred = Y_pred.reshape(Y_pred.shape[0])
        actual = Y_test.reshape(Y_test.shape[0])

        #Computing the Confusion Matrix
        K = len(np.unique(actual)) # Number of classes 
        confusion_matrix = np.zeros((K, K))

        for i in range(len(actual)):
            x = actual[i]
            y = Y_pred[i].astype(int)
            confusion_matrix[x][y] += 1
        print(confusion_matrix)
        sse = np.sum((Y_pred-actual)**2)
        print('The Squared Sum of Errors is - ',sse)
        precision = confusion_matrix[1][1]/(confusion_matrix[0][1] + confusion_matrix[1][1])
        recall = confusion_matrix[1][1]/(confusion_matrix[1][0] + confusion_matrix[1][1])
        print('The Precision is - ', precision)
        print('The Recall is - ', recall)
        f_score = 2*precision*recall/(precision+recall)
        print('The F1-score is - ',f_score)
        print('Accuracy of NN using relu on test set: {:.3f}'.format((Y_test.shape[0]-sse)/Y_test.shape[0]*100))
        return sse
    
def loadData():
    data = pd.read_csv('housepricedata.csv')
    Y = data.loc[:, 'AboveMedianPrice']
    X = data.loc[:, :'GarageArea']
    for colName in X.columns:
        X[colName] = (X[colName] - X[colName].mean())/(X[colName].std())
    return X, Y
        
if __name__ == "__main__":
    X, Y = loadData()
    X_train=X.sample(frac=0.8,random_state=3) #random state is a seed value
    X_test=X.drop(X_train.index)
    Y_train=Y.sample(frac=0.8,random_state=3) 
    Y_test=Y.drop(Y_train.index)
    X_train = X_train.values
    Y_train = Y_train.values.reshape(Y_train.shape[0],1)
    X_test = X_test.values
    Y_test = Y_test.values.reshape(Y_test.shape[0],1)
    nn = NeuralNetwork(X_train, Y_train)
    x_axis = []
    y_axis = []
    for i in range(800):
        nn.feedforward()
        nn.backprop()
        y = nn.checkAccuracy(X_test, Y_test)
        x_axis.append(i)
        y_axis.append(y)
    plt.plot(x_axis, y_axis)
    plt.xlabel('Iterations')  
    plt.ylabel('Loss')  
    plt.title('Loss Function with 1 Hidden Layer')
    plt.show
    plt.savefig('Gaussian.jpg', bbox_inches='tight', dpi=300, quality=80, optimize=True, progressive=True)


    