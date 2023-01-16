import numpy as np
from neural_network import *
from basic_functions import *

def costFunction(outputLayer,Output):
    outputLayer = np.array(outputLayer).reshape([1,10])
    requiredOutput = np.zeros([1,10])
    requiredOutput[0,Output] = 1         #the neuron which needs to get activated

    costSqRt=(outputLayer - requiredOutput)      
    costFn = (costSqRt**2)/20
    cost = np.sum(costFn)
    return cost ,costFn ,costSqRt


def Derivatives(i):
    global inputLayer,Z1,H1Layer,Z2,H2Layer,Z3,outputLayer,Wt1,Bias1,Wt2,Bias2,Wt3,Bias3,cost,costFn,costSqRt,data,labels
    inputLayer,Z1,H1Layer,Z2,H2Layer,Z3,outputLayer,Wt1,Bias1,Wt2,Bias2,Wt3,Bias3 = Layers(data[i])
    cost,costFn,costSqRt=costFunction(outputLayer,labels[i])
    dOutputLayer = (costSqRt)/10
    dWt3 = H2Layer.reshape([16,1]).dot((dSigmoid(Z3)*(dOutputLayer)))
    dBias3=dOutputLayer*dSigmoid(Z3)
    dH2Layer = (dOutputLayer*dSigmoid(Z3)).dot(Wt3.reshape(10,16))
    dWt2 = ((dH2Layer*dSigmoid(Z2)).reshape([16,1])).dot(H1Layer)
    dBias2 =dH2Layer*dSigmoid(Z2)
    dH1Layer = (dH2Layer*dSigmoid(Z2)).dot(Wt2)
    dWt1 = (inputLayer.reshape([784,1])).dot(dH1Layer*dSigmoid(Z1))
    dBias1=dH1Layer*dSigmoid(Z1)   


    return(dWt3,dBias3,dWt2,dBias2,dWt1,dBias1)


def train(epochs,alpha):
    global inputLayer,Z1,H1Layer,Z2,H2Layer,Z3,outputLayer,Wt1,Bias1,Wt2,Bias2,Wt3,Bias3,cost,costFn,costSqRt
    for i in range(epochs):
        dWt3,dBias3,dWt2,dBias2,dWt1,dBias1 = Derivatives(i)
        Wt1 = Wt1 - alpha*dWt1
        Bias1 = Bias1 - alpha*dBias1
        Wt2 = Wt2 - alpha*dWt2
        Bias2 = Bias2 - alpha*dBias2
        Wt3 = Wt3 - alpha*dWt3
        Bias3 = Bias3 - alpha*dBias3
        if i % 10 ==0 :
            print(i , cost)

    final_wt1,final_b1,final_wt2,final_b2,final_wt3,final_b3 = Wt1,Bias1,Wt2,Bias2,Wt3,Bias3
    return final_wt1,final_b1,final_wt2,final_b2,final_wt3,final_b3

epochs = 10000
alpha = 0.025
final_wt1,final_b1,final_wt2,final_b2,final_wt3,final_b3 = train(epochs,alpha)














        
    



