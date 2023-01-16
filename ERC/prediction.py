import numpy as np
from basic_functions import *
from neural_network import *
from backpropogation import *

def predict(img,final_wt1,final_b1,final_wt2,final_b2,final_wt3,final_b3):
    input= np.empty([784,1])
    for i in range(len(img)) :
        temp = np.zeros([28,])
        for a in range(i):
          temp[a]=img[i][a]
        input[28*i:28*(i+1) , 0] = temp 
    input=input.reshape([1,784])
    Z1_p = inputLayer.dot(final_wt1) + final_b1
    H1 = sigmoid(Z1_p)
    Z2_p = H1.dot(final_wt2)+ final_b2
    H2  = sigmoid(Z2_p)
    Z3_p = H2.dot(final_wt3)+ final_b3
    output = sigmoid(Z3_p)
    prediction = np.argmax(output, 0)    # the index of maximum of the output will be the number 

    return prediction

image = int(input('Enter any number between 0 and 59999'))
image = data[image]
prediction  = predict(image,final_wt1,final_b1,final_wt2,final_b2,final_wt3,final_b3)

plt.imshow(image)
plt.title(prediction)
plt.show()