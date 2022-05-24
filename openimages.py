# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:50:45 2020

@author: elise
"""
import numpy as np
import matplotlib.pyplot as plt

def saveFile():
    train_data = np.loadtxt("mnist_test.csv", delimiter=",")
    
    for i in range(len(train_data)) :
        file = open("images_test/image"+str(int(train_data[i][0]))+str(i)+".txt","w+")
        for j in range(1,len(train_data[i])) :
            file.write(str(int(train_data[i][j])))
            if j != len(train_data[i])-1 : 
                file.write(",")
        
        file.close()

def readNumber(file):
    
    fichier = open(file,"r").read()
    train_imgs = fichier.split(",")
    for i in range(len(train_imgs)) :
        train_imgs[i] = float(train_imgs[i])
    img = np.reshape(train_imgs,(28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
    
    
saveFile()
