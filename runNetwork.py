import tensorflow as tf
import model as mdl
from modules import *
import numpy as np
import random
import sys
#from tensorflow.app.flags import DEFINE_integer as int_flag
#from tf.app.flags import DEFINE_float as flt_flag
#from tf.app.flags import DEFINE_string as str_flag




# High-level options
tf.app.flags.DEFINE_string("modelName", "Basic", "Name of the model")
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "MNIST", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment.")
tf.app.flags.DEFINE_integer("Nepochs", 0, "Number of epochs to train. 0 means train indefinitely.")
tf.app.flags.DEFINE_integer("print_every", 1000, "Print training statues every N batches")
tf.app.flags.DEFINE_integer("eval_every", 1, "Print training statues every N batches")
tf.app.flags.DEFINE_bool("verbose", True, "Print")

# Hyperparameters
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("train_batch_size", 10, "Batch size while training")
tf.app.flags.DEFINE_integer("embedding_size", 256, "Size of vector representation.")

# Optimization
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_string("learning_rate_decay", "None", "Which learning rate decay to use. None means no decay")
tf.app.flags.DEFINE_float("decay_rate", 0, "Learing rate decay when specified above")
tf.app.flags.DEFINE_string("optimizer", "Adam", "Name of optimizer to use")
tf.app.flags.DEFINE_float("beta1", 0.95, "Beta1 parameter for Adam optimizer.")
tf.app.flags.DEFINE_float("beta2", 0.995, "Beta2 parameter for Adam optimizer.")

# Dataset
tf.app.flags.DEFINE_float("trainRatio", 0.70, "Ratio of the data that should be used for training")
tf.app.flags.DEFINE_float("valRatio", 0.15, "ratio of the data that should be used for validation")
tf.app.flags.DEFINE_float("testRatio", 0.15, "ratio of the data that should be used for testing")
tf.app.flags.DEFINE_integer("Nfeatures", 11, "Number of features per event") 
tf.app.flags.DEFINE_integer("eval_batch_size", 64, "Maximum number of samples to evaluate at a time")
tf.app.flags.DEFINE_integer("shuffle_buffer_size", 1000, "Number of samples to shuffle at a time")

# Saving
tf.app.flags.DEFINE_integer("save_every", 5, "Save model parameters every N training steps")
tf.app.flags.DEFINE_string("bestModel_loss_ckpt_path", "./checkpoints/bestLoss", "File name of the saved model with the best loss")
tf.app.flags.DEFINE_string("bestModel_acc_ckpt_path", "./checkpoints/bestAcc", "File name of the saved model with the best accuracy")
tf.app.flags.DEFINE_string("checkpoint_path", "./checkpoints", "File name of the saved checkpoint model")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

FLAGS = tf.app.flags.FLAGS
#os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

# tensorflow config
tfConfig = tf.ConfigProto()
tfConfig.gpu_options.allow_growth = True




##################
###  Get Data  ###
##################

print("Importing Data")

###  Import all the data  ####
dataDir = "data/mnist/"
dataX_train, dataY_train, dataX_test, dataY_test = importData(dataDir)

dataX_train -= np.mean(dataX_train)
dataX_train /= np.var(dataX_train)
dataX_test  -= np.mean(dataX_test)
dataX_test  /= np.var(dataX_test)

if FLAGS.verbose:
  print("Original data X/Y shape:  {}  /  {}"
      .format(dataX_train.shape, dataY_train.shape))

###  Split data  ###
Nevents = dataX_train.shape[0]
indData = int(np.ceil(FLAGS.trainRatio/(FLAGS.trainRatio + FLAGS.valRatio)*Nevents))
inds = [1, 3, 5, 10, 9, 11, 13, 15, 17, 19]

dataset_train = tf.data.Dataset.from_tensor_slices(
                  (dataX_train[:indData,:,:,:], 
                   np.reshape(dataY_train[:indData,:], (-1)))).batch(FLAGS.train_batch_size)
dataset_valid = tf.data.Dataset.from_tensor_slices(
                  (dataX_train[indData:,:,:,:], 
                   np.reshape(dataY_train[indData:,:], (-1)))).batch(FLAGS.eval_batch_size)
dataset_test  = tf.data.Dataset.from_tensor_slices(
                  (dataX_test, np.reshape(dataY_test, (-1)))).batch(FLAGS.eval_batch_size)

printDataInfo(dataX_train, dataY_train, dataX_test, dataY_test,
    dataset_train, dataset_valid, dataset_test)


####################
###  Parameters  ###
####################

parameters = {
    "Nclasses" : 10}


############################
#####  Begin Training  #####
############################

basicNet = mdl.basicNNCLASS(FLAGS, 
              dataset_train, 
              dataset_valid, 
              dataset_test,
              parameters)

with tf.Session(config=tfConfig) as sess:
  # Initialize model with new or restored parameters
  initializeModel(sess, basicNet, FLAGS.checkpoint_path, 
      expect_exists=False)

  basicNet.train(sess)



