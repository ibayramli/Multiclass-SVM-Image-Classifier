import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Outputs:
  - Loss 
  - Gradient Matrix
  """ 
  num_train = X.shape[0]
  loss = 0.0

  # initialize the gradient as a zero DxC
  dW = np.zeros(W.shape) 

  # NxC matrix representing class scores for each image
  score_matrix = X.dot(W) 
  correct_class_vector = score_matrix[np.arange(score_matrix.shape[0]), y]

  # NxC matrix representing loss incurred from each class for all N images
  loss_matrix = np.maximum(0, (score_matrix.transpose() - correct_class_vector + 1).transpose())
  loss_matrix[np.arange(loss_matrix.shape[0]), y] = 0
  loss = np.sum(loss_matrix) / loss_matrix.shape[0] + reg * np.sum(W * W)
  
  loss_binary_matrix = np.where(loss_matrix == 0, 0, 1)
  loss_binary_matrix[np.arange(num_train), y] -= np.sum(loss_binary_matrix, axis=1)
  dW = (X.transpose()).dot(loss_binary_matrix)
  dW = dW/num_train + reg*W

  return loss, dW
