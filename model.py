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

class basicNNCLASS():
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

  def __init__(self, FLAGS, dataset_train, dataset_valid, dataset_test, parameters):
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

    # Data
    with tf.device('/cpu:0'):
      # Variables
      self.features = None
      self.labels   = None
      self.handle       = tf.placeholder(tf.string, shape=[])
      self.train_handle = None
      self.valid_handle = None
      self.test_handle  = None

      self.dataset_train = dataset_train
      self.dataset_valid = dataset_valid
      self.dataset_test  = dataset_test

      self.dataset_train = self.dataset_train.shuffle(
                                buffer_size=self.FLAGS.shuffle_buffer_size)
      self.dataset_valid = self.dataset_valid.shuffle(
                                buffer_size=self.FLAGS.shuffle_buffer_size)
      self.dataset_test  = self.dataset_test.shuffle(
                                buffer_size=self.FLAGS.shuffle_buffer_size)

      self.dataset_train = self.dataset_train.batch(batch_size=self.FLAGS.train_batch_size)
      self.dataset_valid = self.dataset_valid.batch(batch_size=self.FLAGS.eval_batch_size)
      self.dataset_test  = self.dataset_test.batch(batch_size=self.FLAGS.eval_batch_size)

      self.dataset_train = self.dataset_train.prefetch(1)
      self.dataset_valid = self.dataset_valid.prefetch(1)
      self.dataset_test  = self.dataset_test.prefetch(1)

      # Data iterator
      self.train_iter = self.dataset_train.make_initializable_iterator()
      self.valid_iter = self.dataset_valid.make_initializable_iterator()
      self.test_iter  = self.dataset_test.make_initializable_iterator()

      self.iter       = tf.data.Iterator.from_string_handle(
                          self.handle, 
                          self.dataset_train.output_types, 
                          self.dataset_train.output_shapes)

      self.features, self.labels = self.iter.get_next()
      self.batch_size = tf.shape(self.features)[0]

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


    #########################
    #####  Build Graph  #####
    #########################

    with tf.variable_scope(self.FLAGS.model_name):
      self.initialize_placeHolders()
      self.build_graph()
      self.add_loss()
      self.add_accuracy()

    #####  Create Optimization  #####
    with tf.variable_scope("optimize"):
      self.initialize_learning_rate()
      self.initialize_optimization()
    

    #####################################
    #####  History and Checkpoints  #####
    #####################################

    self.hasTrained     = False
    self._lastSaved     = collections.defaultdict(None)
    self.history        = collections.defaultdict(list)
    self.saver          = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
    self.bestLossSaver  = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
    self.bestAccSaver   = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)

    logging.basicConfig(level=logging.INFO)
    log_handler = logging.FileHandler("log.txt")
    logging.getLogger().addHandler(log_handler)

    self.summaries = tf.summary.merge_all()

    self._reset()


  #############################################################################
  def _reset(self):
    """
    Resets the graph before each time it is trained.
    """

    self.history = collections.defaultdict(list)


  #############################################################################
  def initialize_placeHolders(self):
    """
    Initialize all place holders with size variables from self.config
    """

    self.isTraining = tf.placeholder(name="isTraining", dtype=tf.bool)

  
  #############################################################################
  def initialize_learning_rate(self):
    """
    Initialize learning rate based on self.FLAGS.
    """

    if (self.FLAGS.learning_rate_decay is "exponential"):
      self.learning_rate = tf.train.exponential_decay(
                                    self.FLAGS.learning_rate,
                                    self.global_step,
                                    self.FLAGS.decay_steps,
                                    self.FLAGS.decay_rate)
    else :
      self.learning_rate = self.FLAGS.learning_rate
 


  #############################################################################
  def initialize_optimization(self):
    """
    Training step called to update the training variables in order to minimize 
    self.loss with the minimization type of self.solver.
    """

    if self.FLAGS.optimizer == "Adam" :
      self.solver = tf.train.AdamOptimizer(
                          learning_rate = self.learning_rate,
                          beta1         = self.FLAGS.beta1,
                          beta2         = self.FLAGS.beta2)
    else:
      print("ERROR: Cannot handle optimizer type {}!!!".format(self.FLAGS.optimizer))
      raise RuntimeError
 
    # batch normalization in tensorflow requires this extra dependency
    # this is required to update the moving mean and moving variance variables
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
      self.update = self.solver.minimize(self.loss, global_step=self.global_step)


  #############################################################################
  def build_graph(self):
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


  #############################################################################
  def run_train_step(self, sess, summaryWriter):

    # Build feed dictionary
    input_feed = {
      self.handle     : self.train_handle,
      self.isTraining : True}

    # Output feed
    output_feed = [self.loss, 
        self.accuracy, 
        self.global_step, 
        self.summaries, 
        self.update]


    # Run train step
    loss, accuracy, global_step, summaries, _ = sess.run(output_feed, input_feed) 

    # All summaries in the graph are added to Tensorboard
    summaryWriter.add_summary(summaries, global_step)

    return loss, accuracy, global_step


  #############################################################################
  def train(self, sess):
    """
    Train the model by using tf.data class to supply batches. 
    The loss and accuracy is saved in the history for each 
    self.FLAGS.Nepoch: The number of epochs trained over
    self.FLAGS.batch_size: Maximimum size of each minibatch
    self.FLAGS.sample_step: Save and print (if verbose) the loss and 
        accuracy ofter self.FLAGS.sample_step minibatch

    EDIT
      If there are network specific placeholders then the feed_dicts in 
      this function must be changed to accommodate them.
    """  

    logging.info("////////////////////////////")
    logging.info("/////  BEGIN TRAINING  /////")
    logging.info("////////////////////////////")

    # for TensorBoard
    summaryWriter = tf.summary.FileWriter(
        "./checkpoints/", 
        sess.graph)

    # Create handles
    #self.train_handle = sess.run(self.train_iter.string_handle())
    #self.valid_handle = sess.run(self.valid_iter.string_handle())
    #self.test_handle = sess.run(self.test_iter.string_handle())

    # Initialize iterator
    sess.run(self.train_iter.initializer)

    # Print initial model predictions
    emaTrainLoss = self.get_loss(sess, dSet="train")
    emaTrainAccr = self.get_accuracy(sess, dSet="train")
    valLoss   = self.get_loss(sess, dSet="val")
    valAccr   = self.get_accuracy(sess, dSet="val")
    logging.info("Initial training Loss / Accuracy: %f / %f)" % (emaTrainLoss, emaTrainAccr))
    logging.info("Initial validation Loss / Accuracy: %f / %f)" % (valLoss, valAccr))

    randomRatio = 1.0
    epoch = 0
    best_val_loss = None
    best_val_acc  = None


    ######  Loop over epochs  #####
    while (self.FLAGS.Nepochs is 0) or (epoch <= self.FLAGS.Nepochs):
      epoch += 1
      epoch_tic = time.time()

      # Evaluate test and validation data
      trnLoss = self.get_loss(sess, dSet="train")
      trnAccr = self.get_accuracy(sess, dSet="train")
      valLoss = self.get_loss(sess, dSet="val")
      valAccr = self.get_accuracy(sess, dSet="val")

      print_info = "\tTraining %.5f / %.5f \tValidation %.5f / %.5f" %\
          (trnLoss, trnAccr, valLoss, valAccr)
      logging.info("\n\n/////  Begin Epoch {}  /////\n".format(epoch) 
          + print_info)


      # Initialize iterator
      sess.run(self.train_iter.initializer)

      #####  Loop over mini batches  #####
      while True:

        # Perform training step
        try :
          tstep_tic = time.time()
          curLoss, curAccr, global_step = self.run_train_step(sess, summaryWriter)
          tstep_toc = time.time()
          tstep_time = tstep_toc - tstep_tic
        except tf.errors.OutOfRangeError:
          break

        # Update training history parameters
        emaTrainLoss = curLoss*(1-self.FLAGS.train_variable_decay)\
            + emaTrainLoss*self.FLAGS.train_variable_decay 
        emaTrainAccr = curAccr*(1-self.FLAGS.train_variable_decay)\
            + emaTrainAccr*self.FLAGS.train_variable_decay 

        ###  Evaluate model  ###
        if global_step % self.FLAGS.eval_every == 0:

          # Save training data measurements
          self.writeSummary(emaTrainLoss, "train/loss", summaryWriter, global_step)
          self.writeSummary(emaTrainAccr, "train/acc", summaryWriter, global_step)
          self.history["step"].append(global_step)
          self.history["trainLoss"].append(emaTrainLoss)
          self.history["trainAccr"].append(emaTrainAccr)

          # Evaluate validation data
          valLoss = self.get_loss(sess, dSet="val")
          valAccr = self.get_accuracy(sess, dSet="val")

          self.writeSummary(valLoss, "val/loss", summaryWriter, global_step)
          self.writeSummary(valAccr, "val/acc", summaryWriter, global_step)
          self.history["validLoss"].append(valLoss)
          self.history["validAccr"].append(valAccr)

          # Logging results
          print_info = "%i\tTraining %.5f / %.5f \tValidation %.5f / %.5f" %\
              (global_step, emaTrainLoss, emaTrainAccr, valLoss, valAccr)
          logging.info(print_info)

          # plot training progress
          self.plot_results()


        # Save model
        if global_step % self.FLAGS.save_every == 0:
          logging.info("Saving model at iteration {} to {}".format(
              global_step, self.FLAGS.checkpoint_path))
          self.saver.save(sess, 
              self.FLAGS.checkpoint_path + "/" + self.FLAGS.experiment_name, 
              global_step=global_step)
          self.saveTrainingHistory(
              fileName=self.FLAGS.checkpoint_path + "/" + self.FLAGS.experiment_name, 
              global_step=global_step)


          # Evaluate validation data
          valLoss = self.get_loss(sess, dSet="val")
          valAccs = self.get_accuracy(sess, dSet="val")

          # Save best models
          if (best_val_loss is None) or (valLoss < best_val_loss):
            logging.info("Saving best loss model at iteration {} in {}".format(
                    global_step, self.FLAGS.bestModel_loss_ckpt_path))
            best_val_loss = valLoss
            self.bestLossSaver.save(sess, 
                    self.FLAGS.bestModel_loss_ckpt_path + "/" + self.FLAGS.experiment_name, 
                    global_step=global_step)
            self.saveTrainingHistory(
                    fileName=self.FLAGS.bestModel_loss_ckpt_path + "/" + self.FLAGS.experiment_name, 
                    global_step=global_step)
          if (best_val_acc is None) or (valAccs > best_val_acc):
            logging.info("Saving best accuracy model at iteration {} in {}".format(
                    global_step, self.FLAGS.bestModel_acc_ckpt_path))
            best_val_acc = valAccs
            self.bestAccSaver.save(sess, 
                    self.FLAGS.bestModel_acc_ckpt_path + "/" + self.FLAGS.experiment_name, 
                    global_step=global_step)
            self.saveTrainingHistory(
                    fileName=self.FLAGS.bestModel_acc_ckpt_path + "/" + self.FLAGS.experiment_name, 
                    global_step=global_step)


    loss_train = self.get_loss(sess, dSet="train")
    acc_train  = self.get_accuracy(sess, dSet="train")

    loss_val = self.get_loss(sess, dSet="val")
    acc_val  = self.get_accuracy(sess, dSet="val")

    print(loss_train, acc_train)
    if self.FLAGS.verbose:
      print("\n\n")
      print("###########################")
      print("#####  Final Results  #####")
      print("###########################")
      print("\nTraining [ Loss: %f\t Accuracy: %f]" \
          % (loss_train, acc_train))
      print("Validation [ Loss: %f\t Accuracy: %f]" \
          % (loss_val, acc_val))
                              
    self.hasTrained = True
       

  #############################################################################
  def plot_results(self):
    """
    Plot the loss and accuracy history for both the train and validation
    datasets after training
    """


    f1, ax1 = plt.subplots()
    h1, = ax1.plot(self.history["step"], self.history["trainLoss"],\
        "b-", label="Loss - Train")
    h2, = ax1.plot(self.history["step"], self.history["validLoss"],\
        "b.", label="Loss - Validation")

    ax1.set_ylabel("Loss", color = "blue")
    ax1.tick_params("y", color = "blue")
    ax1.yaxis.label.set_color("blue")
    ax1.set_xlabel("Training Steps [{}]".format(self.FLAGS.eval_every))

    ax2 = ax1.twinx()
    h3, = ax2.plot(self.history["step"], self.history["trainAccr"], "r-",\
        label = "Accuracy - Train")
    h4, = ax2.plot(self.history["step"], self.history["validAccr"], "r.",\
        label = "Accuracy - Validation")

    ax2.set_ylabel("Accuracy", color = "red")
    ax2.tick_params("y", color = "red")
    ax2.yaxis.label.set_color("red")

    hds = [h1,h2,h3,h4]
    lbs = [l.get_label() for l in hds]
    ax1.legend(hds, lbs)
    f1.tight_layout()
    plt.savefig("trainingHistory.png")

    plt.close(f1)
    #plt.show()


  #############################################################################
  def evaluate(self, sess, outputs, dSet=None, dataX=None, dataY=None):
    """
    Run the session with inputs from feed_dict and return the outputs listed
    in the list outputs. The following lines are required, but more items 
    can be added to the feed_dict.
    """

    feed_dict = {self.isTraining : False}

    if (dataX is not None) and (dataY is not None):
      feed_dict[self.features] = dataX
      feed_dict[self.labels]   = dataY
      return sess.run(output, feed_dict) 

    if dSet is "train":
      sess.run(self.train_iter.initializer)
      feed_dict[self.handle] = self.train_handle
    elif dSet is "val":
      sess.run(self.valid_iter.initializer)
      feed_dict[self.handle] = self.valid_handle
    elif dSet is "test":
      sess.run(self.test_iter.initializer)
      feed_dict[self.handle] = self.test_handle
    else:
      print("ERROR: Do not recognize dataset {}".format(dSet))
      raise RuntimeError

    runningOutput = []
    while True:
      # Get batch data
      try :
        runningOutput.append(sess.run(output, feed_dict))
      except tf.errors.OutOfRangeError:
        break

    return runningOutput


  #############################################################################
  def get_predictions(self, sess, dSet=None, dataX=None, dataY=None):
    
    feed_dict = {self.isTraining : False}

    if dataX is not None and dataY is not None:
      feed_dict[self.features] = dataX
      feed_dict[self.labels]   = dataY
      return sess.run([self.prediction], feed_dict) 

    if dSet is "train":
      sess.run(self.train_iter.initializer)
      feed_dict[self.handle] = self.train_handle
    elif dSet is "val":
      sess.run(self.valid_iter.initializer)
      feed_dict[self.handle] = self.valid_handle
    elif dSet is "test":
      sess.run(self.test_iter.initializer)
      feed_dict[self.handle] = self.test_handle
    else:
      print("ERROR: Do not recognize dataset {}".format(dSet))
      raise RuntimeError

    if feed_dict[self.handle] is None:
      raise RuntimeError("Cannot evaluate with handle = None")

    try:
      runningPredictions = sess.run([self.prediction], feed_dict)
    except tf.errors.OutOfRangeError:
      print("Error: Cannot get predictions because dataset is empty")
      raise RuntimeError

    while True:
      # Get batch data
      try :
        runningPredictions.concatenate(
                sess.run([self.prediction], feed_dict), 
                axis=0)
      except tf.errors.OutOfRangeError:
        break

    return runningPredictions


  #############################################################################
  def get_accuracy(self, sess, dSet=None, dataX=None, dataY=None, isTraining=False):

    feed_dict = {self.isTraining : False}

    if dataX is not None and dataY is not None:
      feed_dict[self.features] = dataX
      feed_dict[self.labels]   = dataY
      return sess.run([self.accuracy], feed_dict)[0]

    runningAccuracy = 0
    if dSet is "train":
      sess.run(self.train_iter.initializer)
      feed_dict[self.handle] = self.train_handle
    elif dSet is "val":
      sess.run(self.valid_iter.initializer)
      feed_dict[self.handle] = self.valid_handle
    elif dSet is "test":
      sess.run(self.test_iter.initializer)
      feed_dict[self.handle] = self.test_handle
    else:
      print("ERROR: Do not recognize dataset {}".format(dSet))
      raise RuntimeError

    if feed_dict[self.handle] is None:
      raise RuntimeError("Cannot evaluate with handle = None")

    Nsamples    = 0
    runningAccr = 0
    while True:
      try :
        _curAccr, _Nsamples = sess.run([self.accuracy_sum, self.batch_size],\
            feed_dict)
        Nsamples    += _Nsamples
        runningAccr += _curAccr
      except tf.errors.OutOfRangeError:
        break

    return runningAccr/Nsamples


  #############################################################################
  def get_loss(self, sess, dSet=None, dataX=None, dataY=None):

    feed_dict = {self.isTraining : False}

    if dataX is not None and dataY is not None:
      feed_dict[self.features] = dataX
      feed_dict[self.labels]   = dataY
      return sess.run([self.loss], feed_dict)[0]

    if dSet is "train":
      sess.run(self.train_iter.initializer)
      feed_dict[self.handle] = self.train_handle
    elif dSet is "val":
      sess.run(self.valid_iter.initializer)
      feed_dict[self.handle] = self.valid_handle
    elif dSet is "test":
      sess.run(self.test_iter.initializer)
      feed_dict[self.handle] = self.test_handle
    else:
      print("ERROR: Do not recognize dataset {}".format(dSet))
      raise RuntimeError

    if feed_dict[self.handle] is None:
      raise RuntimeError("Cannot evaluate with handle = None")

    Nsamples    = 0
    runningLoss = 0
    while True:
      try :
        _curLoss, _Nsamples = sess.run([self.loss_sum, self.batch_size],\
            feed_dict)
        Nsamples    += _Nsamples
        runningLoss += _curLoss
      except tf.errors.OutOfRangeError:
        break

    return runningLoss/Nsamples

  
  #############################################################################
  def writeSummary(self, value, tag, summaryWriter, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summaryWriter.add_summary(summary, global_step)


  #############################################################################
  def saveTrainingHistory(self, fileName="", global_step=None):
    """
    Save training History
    """

    if global_step is not None: 
      fullName = fileName + "-" + str(global_step) + "-history.pl"
    else:
      fullName = fileName + "-history.pl"

    if fileName in self._lastSaved.keys():
      os.remove(self._lastSaved[str(fileName)])

    self._lastSaved[str(fileName)] = fullName

    pl.dump(self.history, open(fullName, "wb"))


  #############################################################################
  def saveModel(self, fileName):
    """
    Saves the current model as fileName
    """

    if self.saver is None:
      self.saver = tf.train.Saver()
      self.saver.save(self.sess, fileName)
    else:
      self.saver.save(self.sess, fileName)

