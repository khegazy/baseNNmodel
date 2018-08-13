import tensorflow as tf
import numpy as np
from numpy import random as rand
import pandas as pd
from StringIO import StringIO
import os
import collections
import itertools
import csv
from PIL import Image
import sys
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mnist import MNIST

########################
###  Importing Data  ###
########################

def importData(dataDir):

  #####  Initialize Variables  #####
  mndata = MNIST(dataDir)
  dataX_train, dataY_train = mndata.load_training()
  dataX_test, dataY_test = mndata.load_testing()

  dataX_train = np.array(dataX_train, dtype=np.float) 
  dataY_train = np.array(dataY_train, dtype=np.int32)
  dataX_test  = np.array(dataX_test, dtype=np.float)
  dataY_test  = np.array(dataY_test, dtype=np.int32)

  dataX_train = np.reshape(dataX_train, (dataX_train.shape[0],28,28,1))
  dataY_train = np.reshape(dataY_train, (-1,1))
  dataX_test  = np.reshape(dataX_test, (dataX_test.shape[0],28,28,1))
  dataY_test  = np.reshape(dataY_test, (-1,1))

  return dataX_train, dataY_train, dataX_test, dataY_test



#############################################################################
def initializeModel(session, model, folderName, expect_exists=False, import_train_history=False):
  print("folder",folderName)

  ckpt = tf.train.get_checkpoint_state(folderName)
  print("ckpt", ckpt)
  #ckpt = tf.train.get_checkpoint_state(folderName + "/" + self.FLAGS.experime    nt_name)
  v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
  print("v2_path", v2_path)
  if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
      print "Reading model parameters from %s" % ckpt.model_checkpoint_path
      model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    if expect_exists:
      raise RuntimeError("ERROR: Cannot find saved checkpoint in %s" % folderName)
    else:
      print("Cannot find saved checkpoint at %s" % folderName)
      session.run(tf.global_variables_initializer())
      print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())

  session.run(tf.local_variables_initializer())

  if import_train_history:
    pass





#####  Embedding  #####
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

      x = tf.reshape(inputX, (-1,784))
      predictions = tf.matmul(x, W) + b

      return predictions

