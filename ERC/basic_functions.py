import numpy as np
import cv2

# Function return value between 0 and 1 for all element of matrix
def sigmoid(z):
  z= 1/(1+np.exp(-z))

# Function return derivative value of sigmoid function for all element of matrix
def dSigmoid(z):
  # σ'(x) = σ(x)[1-σ(x)]
  a=sigmoid(z)*(1-sigmoid(z))


def image_resize(path):
  #resizes any image to 28,28 pixels
  
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)   
  w, h = img.shape
  if h > 28 or w > 28:
    (tH, tW) = img.shape
    dX = int(max(0, 28 - tW) / 2.0)
    dY = int(max(0, 28 - tH) / 2.0)

    img = cv2.copyMakeBorder(img, top=dY, bottom=dY,
          left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
          value=(0, 0, 0))
          #Creates border for the image of size 28,28
    img = cv2.resize(img, (28, 28))

  w, h = img.shape

  if w < 28:
      add_zeros = np.ones((28-w, h))*255
      img = np.concatenate((img, add_zeros))

  if h < 28:
      add_zeros = np.ones((28, 28-h))*255
      img = np.concatenate((img, add_zeros), axis=1)
  cv2.resize(img , (28,28))
  return img
