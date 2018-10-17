import tensorflow as tf
from tensorflow.python.framework import function
from modules import *
import numpy as np
import matplotlib.pyplot as plt
import pickle as pl
import collections
import time
import logging

@function.Defun(tf.float32, tf.float32)
def stable_norm_grad(x, dy):
  #return dy*(x/(tf.norm(x, axis=-1)))
  tdy = tf.tile(tf.expand_dims(dy,1), [1,2])
  txn = tf.tile(tf.expand_dims(tf.norm(x, axis=-1), 1), [1,2]) + 1e-5
  return tdy*(x/txn)

@function.Defun(tf.float32, grad_func=stable_norm_grad)
def stable_norm(x):
  return tf.norm(x, axis=-1)

class baseModel():
  """
  baseModel is a parent class to neural networks that encapsulates logic,
  functions, and variables common to most neural networks. Networks 
  inhereting this class must define the virtual functions below:
    def __init__(self, FLAGS, parameters, ...):
      baseModel.__init__(self, FLAGS, parameters)
    def initialize_placeHolders(self):
      Function to initialize place holders, only run during __init__
    def model(self):
      Network architecture is made here, evaluates features
      and returns labals/predictions.
  """

  def __init__(self, FLAGS, parameters):
    """
    Initialize
      FLAGS:
        tf.app.flags that contain common model variables and 
        hyperparameters: layer sizes, optimizers, batch sizes, etc. 
      datasets:
        tf.dataset classes holding the training, validation, and test
        data.
      parameters:
        variables that are not common among various NN archetictures,
        but specific to this network.
    """

    #############################
    ##### Setting Variables #####
    #############################

    # FLAGS
    self.FLAGS = FLAGS

    # Mode
    if self.FLAGS.mode is "Train" or self.FLAGS.mode is "train":
      self.mode = tf.estimator.ModeKeys.TRAIN
    elif self.FLAGS.mode is "Eval" or self.FLAGS.mode is "eval":
      self.mode = tf.estimator.ModeKeys.EVAL
    else:
      raise RuntimeError("ERROR: Do not recognize mode " + self.FLAGS.mode)

    # Parameters
    self.Nrowcol = -1
    if "NrowCol" in parameters:
      self.Nrowcol  = parameters["NrowCol"]
    
    self.Nclasses = -1
    if "Nclasses" in parameters:
      self.Nclasses = parameters["Nclasses"]

    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    if self.FLAGS.verbose:
      print("Printing parameters")
      print("Nrowcol: \t{}".format(self.Nrowcol))
      print("Nclasses: \t{}".format(self.Nclasses))


  #############################################################################
  def build_graph(self):
    """
    Build network graph. Called in __init__ of the child class to avoid 
    calling the same part of the graph multiple times within each 
    self.sess.run call (this leads to errors).
    """

    with tf.variable_scope(self.FLAGS.model_name):
      self.initialize_placeHolders()
      self.model()


  #############################################################################
  def initialize_placeHolders(self):
    """
    Initialize all place holders.
    """

    raise RuntimeError("Must define function initialize_placeHolders(self)")


  #############################################################################
  def model(self):
    """
    Neural network architecture that recieves features from 
      self.features, 
    and returns the prediction by to the variable
      self.prediction
    """

    raise RuntimeError("Must define function model(self)")


  #############################################################################
  def evaluate(self, sess, outputs, dSet=None, dataX=None, dataY=None):
    """
    Run the session with inputs from feed_dict and return the outputs listed
    in the list outputs. 
    """

    raise RuntimeError("Must define function evaluate(...)")


  #############################################################################
  def get_predictions(self, sess, dSet=None, dataX=None, dataY=None):
    """
    Get predictions for data provided via the options above.
    """

    raise RuntimeError("Must define function get_predictions(...)")


