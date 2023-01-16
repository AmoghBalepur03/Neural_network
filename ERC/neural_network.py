from keras.datasets import mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from basic_functions import *

def load_mnist_dataset():

  # load data from tensorflow framework
  ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data() 
  data = np.vstack([trainData, testData])       #(70000,28,28)   
  labels = np.hstack([trainLabels, testLabels]) #(70000,)
  return (data, labels)

def initailize():
  Wt1 = np.random.uniform(-0.5,0.5,[784,16])   # Random vales between (-0.5,0.5) 
  Bias1 = np.zeros([1,16])                     
  Wt2 = np.random.uniform(-0.5,0.5,[16,16])     
  Bias2= np.zeros([1,16])                      #initialized bias to zero
  Wt3 = np.random.uniform(-0.5,0.5,[16,10])
  Bias3= np.zeros([1,10])

  return Wt1,Bias1,Wt2,Bias2,Wt3,Bias3


def Layers(img):
  global Wt1,Wt2,Wt3,Bias1,Bias2,Bias3 
  while True:
    try:  
      inputLayer = np.empty([784,1])            #creating empty numpy array
      for i in range(len(img)) :
        temp = np.zeros([28,])
        for a in range(i):
          temp[a]=img[i][a]                   
        inputLayer[28*i:28*(i+1) , 0] = temp 
      inputLayer=inputLayer.reshape([1,784])    #reshaping to get required shape
      Z1 = inputLayer.dot(Wt1) + Bias1
      H1Layer = sigmoid(Z1)
      Z2 = H1Layer.dot(Wt2)+ Bias2
      H2Layer  = sigmoid(Z2)
      Z3 = H2Layer.dot(Wt3)+ Bias3
      outputLayer = sigmoid(Z3)
      break
    except:
      pass
    

  return inputLayer,Z1,H1Layer,Z2,H2Layer,Z3,outputLayer,Wt1,Bias1,Wt2,Bias2,Wt3,Bias3

data,labels=load_mnist_dataset()
Wt1,Bias1,Wt2,Bias2,Wt3,Bias3 = initailize()



  
  





  




  
    

  



