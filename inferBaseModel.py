import tensorflow as tf
from tensorflow.python.framework import function
from modules import *
from baseModel import baseModel
import numpy as np
import matplotlib.pyplot as plt
import pickle as pl
import collections
import time
import logging


class inferBaseModel(baseModel):
  """
  basicNNCLASS is a parent class to neural networks that encapsulates all 
  the logic necessary for training general nerual network architectures.
  The graph is built by this class and the tensorflow session is passed 
  to this class. Networks inhereting this class must inheret in the same
  way as the function placeholders below:
    ########################################
    #####  Build Neural Network Class  #####
    ########################################
    class NN(modelClass):
      
      def __init__(self, ...):
        modelCLASS.__init__(self, 
            FLAGS, dataset_train, 
            dataset_valid, 
            dataset_test, 
            parameters)
      def initialize_placeHolders(self):
        Function to initialize place holders, only run during __init__
      def build_graph(self):
        Function containing the network architecture to infer the 
        predictions or logits
      def add_loss(self, preds, Y):
        Function to calculate loss (self.loss) given the predictions 
        and truth values in self.prediction and self.labels.
      def add_accuracy(self, preds, Y):
        Function to calculate accuracy (self.accuracy) given the predictions 
        and truth values in self.prediction and self.labels.

  The model is trained after succusfully building the graph by calling
  NN.train(session)
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
    
    baseModel.__init__(self, FLAGS, parameters)


  #############################################################################
  def evaluate(self, sess, outputs, dSet=None, dataX=None, dataY=None):
    """
    Run the session with inputs from feed_dict and return the outputs listed
    in the list outputs. The following lines are required, but more items 
    can be added to the feed_dict.
    """
    
    raise RuntimeError("Must define function evaluate(...)")

  #############################################################################
  def get_predictions(self, sess, dSet=None, dataX=None, dataY=None):
    
    raise RuntimeError("Must define function get_predictions(...)")

