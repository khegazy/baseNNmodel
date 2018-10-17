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

