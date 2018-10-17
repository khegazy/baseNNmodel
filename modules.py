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


def printDataInfo(xTrn, yTrn, xTst, yTst, 
    xTrnDs, xValDs, xTstDs):

  print("\n\n##############################")
  print("#####  Data Information  #####")
  print("##############################\n")
  print("#####  Original Data Structure  #####")
  print("\tTrain X/Y: {} / {}".format(xTrn.shape, yTrn.shape))
  print("\tTest X/Y: {} / {}".format(xTst.shape, yTst.shape))
  print("\n#####  tf.Data Data Structure  #####")
  print("\tTrain: {} / {}".format(xTrnDs.output_shapes, xTrnDs.output_types))
  print("\tValid: {} / {}".format(xValDs.output_shapes, xValDs.output_types))
  print("\tTest: {} / {}".format(xTstDs.output_shapes, xTstDs.output_types))




def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims



#############################################################################
def initializeModel(session, model, folderName, expect_exists=False, import_train_history=False):

  ckpt = tf.train.get_checkpoint_state(folderName)
  v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
  if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)

      # Import history
      model.history = pl.load(open(ckpt.model_checkpoint_path 
          + "-history.pl", "rb"))
  else:
    if expect_exists:
      raise RuntimeError("ERROR: Cannot find saved checkpoint in %s" 
          % folderName)
    else:
      print("Cannot find saved checkpoint at %s" % folderName)
      session.run(tf.global_variables_initializer())
      print('Num params: %d' % sum(v.get_shape().num_elements()\
          for v in tf.trainable_variables()))

  session.run(tf.local_variables_initializer())

  # Initialize iterators
  if model.mode is tf.estimator.ModeKeys.TRAIN:
    model.train_handle = session.run(model.train_iter.string_handle())
    model.valid_handle = session.run(model.valid_iter.string_handle())
    model.test_handle  = session.run(model.test_iter.string_handle())

    session.run(model.train_iter.initializer)
    session.run(model.valid_iter.initializer)
    session.run(model.test_iter.initializer)






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

      print(inputX)
      x = tf.reshape(inputX, (-1,784))
      predictions = tf.matmul(x, W) + b

      return predictions

