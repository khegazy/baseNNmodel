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

########################
###  Importing Data  ###
########################

def importData(dataDir):

  #####  Initialize Variables  #####
  imgSize = 512
  rebin   = 256
  dataX = None 
  dataY = None
  quality = {
      "blank"   : 0,
      "noxtal"  : 1,
      "weak"    : 2,
      "good"    : 3,
      "strong"  : 4}

  #####  Import data  #####
  data = pd.read_csv(dataDir + "categories1.txt", 
      delim_whitespace=True, header=None)

  Nsamples = 30 #len(data.loc[:,0])
  dataX = np.zeros((Nsamples, rebin, rebin, 1), dtype=np.float32)
  dataY = np.zeros((Nsamples, 3), dtype=np.float32)
  dataY[:,1:] = np.array((data.loc[:Nsamples-1,2:]).values, dtype=np.float32)
  for i in range(Nsamples):
    imgObject = Image.open(dataDir + "png/" + data.loc[i,0])
    imgArray  = np.reshape(
                  np.array(imgObject.getdata()).astype(np.float32),
                  (rebin,2,rebin,-1))
    dataX[i,:,:,0] = imgArray.mean(-1).mean(1)

    if data.loc[i,1] in quality.keys():
      dataY[i,0] = quality[data.loc[i,1]]
    else:
      print("ERROR: Cannot interpret data quality {}".format(data.loc[i,1]))
      sys.exit()

  return dataX, dataY



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

  def __init__(self, embedding_size, Nquality_classes):
    self.embedding_size = embedding_size
    self.Nquality_classes = Nquality_classes

  def predict(self, inputX, isTraining):
    with tf.variable_scope("embedding"):

      print("inp", inputX.shape.as_list())
      conv1     = tf.contrib.layers.conv2d(inputs=inputX, 
                    num_outputs=32, kernel_size=4)
      print("conv1", conv1.shape.as_list())
      conv1_BN  = tf.contrib.layers.batch_norm(conv1, 
                    is_training=isTraining)
      print("conv1BN", conv1_BN.shape.as_list())
      pool1     = tf.layers.max_pooling2d(inputs=conv1_BN,
                    pool_size=4, strides=2,
                    name="pool1")
      print("pool1", pool1.shape.as_list())
      conv2     = tf.contrib.layers.conv2d(inputs=pool1,
                    num_outputs=64, kernel_size=4)
      print("conv2", conv2.shape.as_list())
      conv2_BN  = tf.contrib.layers.batch_norm(conv2, 
                    is_training=isTraining)
      print("conv2BN", conv2_BN.shape.as_list())
      pool2     = tf.layers.max_pooling2d(inputs=conv2_BN,
                    pool_size=4, strides=2,
                    name="pool2")
      print("pool2", pool2.shape.as_list())

      pool2Shape = pool2.shape.as_list()
      pool2Flat = tf.reshape(pool2, [-1, pool2Shape[1]*pool2Shape[2]*pool2Shape[3]])
      print("flat", pool2Flat.shape.as_list())
      fullConn1 = tf.contrib.layers.fully_connected(
                    inputs=pool2Flat,
                    num_outputs=1024,
                    activation_fn=tf.nn.relu,
                    scope="FC1")
      print("pool2", pool2.shape.as_list())
      embedding = tf.contrib.layers.fully_connected(
                    inputs=fullConn1,
                    num_outputs=self.embedding_size,
                    activation_fn=tf.nn.relu,
                    scope="FC2")
      print("pool2", pool2.shape.as_list())

      qualityLogits = tf.contrib.layers.fully_connected(
                        inputs=embedding,
                        num_outputs=self.Nquality_classes,
                        activation_fn=None)
      print("qual", qualityLogits.shape.as_list())
      flt1           = tf.contrib.layers.fully_connected(
                        inputs=embedding,
                        num_outputs=1,
                        activation_fn=None)
      print("flt1", flt1.shape.as_list())
      flt2           = tf.contrib.layers.fully_connected(
                        inputs=embedding,
                        num_outputs=1,
                        activation_fn=None)
      print("flt2", flt2.shape.as_list())

      return qualityLogits, flt1, flt2

