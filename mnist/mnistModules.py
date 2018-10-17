import tensorflow as tf
import numpy as np
from numpy import random as rand
import pandas as pd
import os
import collections
import itertools
import csv
from PIL import Image
import sys
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pickle as pl
from mnist import MNIST

########################
###  Importing Data  ###
########################

def importData(dataDir):

  #####  Initialize Variables  #####
  mndata = MNIST(dataDir)
  dataX_train, dataY_train = mndata.load_training()
  dataX_test, dataY_test = mndata.load_testing()

  dataX_train = np.array(dataX_train, dtype=np.float32) 
  dataY_train = np.array(dataY_train, dtype=np.int32)
  dataX_test  = np.array(dataX_test, dtype=np.float32)
  dataY_test  = np.array(dataY_test, dtype=np.int32)

  dataX_train = np.reshape(dataX_train, (dataX_train.shape[0],28,28,1))
  dataY_train = np.reshape(dataY_train, (-1,1))
  dataX_test  = np.reshape(dataX_test, (dataX_test.shape[0],28,28,1))
  dataY_test  = np.reshape(dataY_test, (-1,1))

  return dataX_train, dataY_train, dataX_test, dataY_test


#######################
#####  Embedding  #####
#######################
"""
Embed input into learned vector representation
"""
class predictionClass(object):

  def __init__(self, embedding_size, Nclasses):
    self.embedding_size = embedding_size
    self.Nclasses = Nclasses

  def predict(self, inputX, isTraining):
    with tf.variable_scope("embedding"):

      W = tf.Variable(tf.zeros([784, self.Nclasses]))
      b = tf.Variable(tf.zeros([self.Nclasses]))

      print(inputX)
      x = tf.reshape(inputX, (-1,784))
      predictions = tf.matmul(x, W) + b

      return predictions

