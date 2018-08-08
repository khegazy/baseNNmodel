import tensorflow as tf
import model as mdl
from modules import *
import numpy as np
import random
#from tensorflow.app.flags import DEFINE_integer as int_flag
#from tf.app.flags import DEFINE_float as flt_flag
#from tf.app.flags import DEFINE_string as str_flag




# High-level options
tf.app.flags.DEFINE_string("modelName", "SSRL", "Name of the model")
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "diffraction", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment.")
tf.app.flags.DEFINE_integer("Nepochs", 0, "Number of epochs to train. 0 means train indefinitely.")
tf.app.flags.DEFINE_integer("print_every", 1000, "Print training statues every N batches")
tf.app.flags.DEFINE_integer("eval_every", 1, "Print training statues every N batches")
tf.app.flags.DEFINE_bool("verbose", True, "Print")

# Hyperparameters
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use")
tf.app.flags.DEFINE_integer("embedding_size", 512, "Size of vector representation.")

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
dataDir = "data/"
dataX, dataY = importData(dataDir)
dataX -= np.mean(dataX)
dataX /= np.var(dataX)


#if verbose:
#  print("Original data X/Y shape:  {}  /  {}".format(dataX.shape, dataY.shape))

###  Split data  ###
data = {}
Nevents = dataX.shape[0]
randInds = np.arange(Nevents)
random.shuffle(randInds)
ind1 = int(np.ceil(FLAGS.trainRatio*Nevents))
ind2 = int(np.ceil((FLAGS.trainRatio + FLAGS.valRatio)*Nevents))
ind3 = int(np.ceil((1 - FLAGS.testRatio)*Nevents))
data["train_X"]   = dataX[randInds[:ind1],:]
data["val_X"]     = dataX[randInds[ind1:ind2],:]
data["test_X"]    = dataX[randInds[ind2:],:]
data["train_Y"]   = dataY[randInds[:ind1],:]
data["val_Y"]     = dataY[randInds[ind1:ind2],:]
data["test_Y"]    = dataY[randInds[ind2:],:]


diffractionNet = mdl.diffractionCLASS(FLAGS, data)

with tf.Session(config=tfConfig) as sess:
  # Restore previously trained models

  initializeModel(sess, diffractionNet, FLAGS.checkpoint_path, 
      expect_exists=False)
  #diffractionNet.restoreModel(sess, FLAGS.checkpoint_path,
  #    import_train_history=True)

  diffractionNet.train(sess)







