import sys
sys.path.append("../")

import tensorflow as tf
from tensorflow.python.framework import function
from mnistModules import *
from modules import *
from trainBaseModel import trainBaseModel
from inferBaseModel import inferBaseModel
import numpy as np
import matplotlib.pyplot as plt
import pickle as pl
import collections
import time
import logging


class mnistModel(trainBaseModel, inferBaseModel):

  def __init__(
      self, 
      FLAGS, 
      parameters, 
      dataset_train=None, 
      dataset_valid=None, 
      dataset_test=None):
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

    if (FLAGS.mode is "Train") or (FLAGS.mode is "train"):
      if (dataset_train is None) or (dataset_valid is None) or (dataset_test is None):
        raise RuntimeError("Must give valid train, validation, and test datasets")

      trainBaseModel.__init__(
          self,
          FLAGS, 
          parameters,
          dataset_train, 
          dataset_valid, 
          dataset_test) 

    elif (FLAGS.mode is "Infer") or (FLAGS.mode is "infer"):
      inferBaseModel.__init__(
          self,
          FLAGS,
          parameters)

    else:
      raise RuntimeError("Do not recognize mode " + FLAGS.mode)

    #########################
    #####  Build Graph  #####
    #########################

    self.build_graph()


  #############################################################################
  def initialize_placeHolders(self):
    """
    Initialize all place holders with size variables from self.config
    """

    pass


  #############################################################################
  def model(self):
    """
    Builds the graph from the network specific functions. The graph is built in
    __init__ to avoid calling the same part of the graph multiple times within 
    each self.sess.run call (this leads to errors).
    """

    self.predictionCLS = predictionClass(self.FLAGS.embedding_size, self.Nclasses)
    self.prediction = self.predictionCLS.predict(self.features, self.isTraining)


  #############################################################################
  def add_loss(self):

    with tf.variable_scope("loss"):
      self.loss_sum = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.prediction,
                            labels=self.labels))
      self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.prediction,
                            labels=self.labels))
      tf.summary.scalar("loss", self.loss)


  #############################################################################
  def add_accuracy(self):

    with tf.variable_scope("accuracy"):
      logits = tf.argmax(self.prediction, axis=-1, output_type=tf.int32)
      compare = tf.equal(logits, self.labels)
      self.accuracy_sum = tf.reduce_sum(tf.cast(compare, tf.float32))
      self.accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))


