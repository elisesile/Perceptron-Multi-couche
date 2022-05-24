#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
# -----------------------------------------------------------------------------
import numpy as np

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)
    

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # print(self.layers[3])

        # Return error
        return (error**2).sum()



def readNumber(train_imgs):
    
    for i in range(len(train_imgs)) :
        train_imgs[i] = float(train_imgs[i])
    img = np.reshape(train_imgs,(28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
    

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import os
    import random

    def normalize(content):
        
        for i in range(len(content)):
            if content[i] != 0.0:
                content[i] = content[i]/255
                
        return content
                

    def learn(network,samples, epochs=60000, lrate=.001, momentum=0.01):
        # Train
        erreur = []
        
        for i in range(epochs):
            
            network.propagate_forward( samples['input'][i] )
            a = network.propagate_backward( samples['output'][i], lrate, momentum )
            erreur.append(a/784)
                
        #Renvoi une liste contenant la liste des erreurs effectuées à chaque itération    
        return erreur


    def test(network, testsamples, epochs = 30) :
        
        for i in range(epochs):
            b = []
            x = random.randint(0, len(testsamples))
            #Affichage pour chaque test de l'image d'entrée et de l'image de sortie
            readNumber(testsamples["input"][x])
            readNumber(network.propagate_forward(testsamples["input"][x]))
            for neuron in network.layers[3]:
                b.append(neuron)
            #Affichage pour chaque test de l'activation des neurones de la couche du milieu
            plt.plot(b)
            plt.show()
    
# -------------------------------------------------------------------------------    
    
    network = MLP(784,250,125,10,125,250,784)

#Creation des échantillons pour l'entrainement 

    files = os.listdir("./images")
    samples = np.zeros(len(files), dtype=[('input',  float, 784), ('output', float, 784)])

    
    for i in range(len(files)) :
        content = np.array(open("images/"+files[i],"r").read().split(","))
        content = content.astype(float)
        content = normalize(content)
        samples['input'][i] = content
        samples['output'][i] = content  
    
#Creation des échantillons pour la phase de test
        
    files = os.listdir("./images_test")
    testsamples = np.zeros(len(files), dtype=[('input',  float, 784), ('output', float, 784)])
        
    for i in range(len(files)) :
        content = np.array(open("images_test/"+files[i],"r").read().split(","))
        content = content.astype(float)
        content = normalize(content)
        testsamples['input'][i] = content
        testsamples['output'][i] = content

#Entrainement (peut être très long)
        
    print("training now\n")

    x = learn(network,samples)
    
#Test (beaucoup moins long)
    
    print("testing now\n")
    
    test(network,testsamples)    
    
    
#Affichage de l'évolution de l'erreur au cours des entrainements  
    
    y=list(range(len(x)))

    plt.plot(y,x,color='b',lw=1)
    plt.show()
