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
  modelCLASS is a parent class to neural networks that encapsulates all 
  the logic necessary for training general nerual network archetectures.
  The graph is built by this class and the tensorflow session is held 
  by this class as well. Networks inhereting this class must inheret in
  the following way and define the following functions:
    ########################################
    #####  Build Neural Network Class  #####
    ########################################
    class NN(modelClass):
      
      def __init__(self, inp_config, inp_data, **inp_kwargs):
        modelCLASS.__init__(self, config = inp_config, data = inp_data, kwargs = inp_kwargs)
      def _initialize_placeHolders(self):
        Function to initialize place holders, only rand during __init__
      def predict(self):
        Function containing the network architecture to output the 
        predictions or logits
      def calculate_loss(self, preds, Y):
        Function to calculate loss given the predictions/logits from the 
        output of self.predict and the labels (Y)
      def calculate_accuracy(self, preds, Y):
        Function to calculate the accuracy for the predictions/logits 
        from self.predict against the labels(Y)
         
  Training a network using modelCLASS can be done after the previous 
  functions are defined. Once the class is initialized the graph must 
  first be built using self.build_graph(). After successfull graph 
  construction the network can be trained using self.train(). The 
  loss and accuracy history of the training can be plotted by calling 
  self.plot_results(). An example of training the network is given below.
  
      # Declare Class
      config = configCLASS()
      NN = neuralNetCLASS(inp_config=config)
      # Build Graph
      NN.build_graph()
      # Train
      NN.train()
      NN.plot_results()
  One can run the session to get the following values
    loss, acc, preds = NN.sess.run([NN.loss, NN.accuracy, NN.predictions],
                                   feed_dict = {
                                     self.X_placeHolder : data["type_X"],
                                     self.Y_placeHolder : data["type_Y"],
                                     self.isTraining_placeHolder : False})
  """

  def __init__(self, FLAGS, dataset_train, dataset_valid, dataset_test, parameters):
    """
    Initialize
      config:
        A configure class that contains only variables used to set flags,
        data size, network parameters, parameterize the training, and 
        hold any other input variables into modelCLASS.
      data:
        Data to be trained and validated on. Data is a dictionary of 
        [string : np.ndarray] pairs with the following keys:
          data["train_X"] = training sample features
          data["train_Y"] = training sample labels
          data["val_X"] = validation sample features
          data["val_Y"] = validation sample labels
      kwargs:
        keyword arguments for variables that are often changed, arguments 
        given this way will superceded arguments in the config file.
          NNtype: neural network type, given a type certain variables 
                  are mare accessible
          verbose: enable print statements during training
    """

    #############################
    ##### Setting Variables #####
    #############################

    # FLAGS
    self.FLAGS = FLAGS

    # Data
    self.features = None
    self.labels   = None
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
    self.train_iter = tf.data.Iterator.from_structure(
                        self.dataset_train.output_types, 
                        self.dataset_train.output_shapes)
    self.train_iter_train_op = self.train_iter.make_initializer(self.dataset_train)

    self.eval_iter  = tf.data.Iterator.from_structure(
                        self.dataset_train.output_types, 
                        self.dataset_train.output_shapes)
    self.eval_iter_train_op = self.eval_iter.make_initializer(self.dataset_train)
    self.eval_iter_valid_op = self.eval_iter.make_initializer(self.dataset_valid)
    self.eval_iter_test_op  = self.eval_iter.make_initializer(self.dataset_test)

    # Parameters
    self.Nrowcol = -1
    if "NrowCol" in parameters:
      self.Nrowcol  = parameters["NrowCol"]
    
    self.Nclasses = -1
    if "Nclasses" in parameters:
      self.Nclasses = parameters["Nclasses"]

    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    if self.FLAGS.verbose:
      print "Printing parameters"
      print "Nrowcol: \t{}".format(self.Nrowcol)
      print "Nclasses: \t{}".format(self.Nclasses)


    #########################
    #####  Build Graph  #####
    #########################

    with tf.variable_scope(self.FLAGS.modelName):
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
  def get_data(self, Nsamples, dSet, batchIter=0, shuffleInds=False):

    minInd = Nsamples*batchIter
    maxInd = min([Nsamples*(1+batchIter), self.data[dSet+"_X"].shape[0]])
    if minInd >= self.data[dSet+"_X"].shape[0]:
      print("ERROR: Range of data requested is outside of data size!!!")
      sys.exit()
    
    outDataX = self.data[dSet+"_X"][minInd:maxInd]
    outDataY = self.data[dSet+"_Y"][minInd:maxInd,0]

    return outDataX, outDataY



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
                                    self.Nbatches,
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
    Builds the graph from the network specific functions. The graph is built 
    this to avoid calling the same part of the graph multiple times within 
    each self.sess.run call (this leads to errrors).
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
      compare = tf.equal(logits, self.inputY)
      self.accuracy_sum = tf.reduce_sum(tf.cast(compare, tf.float32))
      self.accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))


  #############################################################################
  def run_train_step(self, sess, summaryWriter):

    # Build feed dictionary
    input_feed = { 
      self.isTraining : True}

    # Output feed
    output_feed = [self.loss, self.global_step, self.summaries, self.update]

    # Run train step
    loss, global_step, summaries, _ = sess.run(output_feed, input_feed) 

    # All summaries in the graph are added to Tensorboard
    summaryWriter.add_summary(summaries, global_step)

    return loss, global_step


  #############################################################################
  def train(self, sess):
    """
    Train the model by looping over epochs and making random batches for 
    each epoch. The loss and accuracy is saved in the history for each 
    self.config.sample_step minibatches.
    self.config.Nepoch: The number of epochs trained over
    self.config.batch_size: Maximimum size of each minibatch
    self.config.sample_step: Save and print (if verbose) the loss and 
        accuracy of each minibatch

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

    # Print initial model predictions
    trainLoss = self.get_loss(sess, dSet="train")
    trainAccs = self.get_accuracy(sess, dSet="train")
    valLoss   = self.get_loss(sess, dSet="val")
    valAccs   = self.get_accuracy(sess, dSet="val")
    logging.info("Initial training Loss / Accuracy: %f / %f)" % (trainLoss, trainAccs))
    logging.info("Initial validation Loss / Accuracy: %f / %f)" % (valLoss, valAccs))

    Nbatches = int(np.ceil(self.data["train_X"].shape[0]/float(self.FLAGS.batch_size)))
    randomRatio = 1.0
    epoch = 0
    best_val_loss = None
    best_val_acc  = None


    ######  Loop over epochs  #####
    while (self.FLAGS.Nepochs is 0) or (epoch <= self.FLAGS.Nepochs):
      epoch += 1
      epoch_tic = time.time()

      # Initialize iterator
      sess.run(self.train_iter_train_op)

      logging.info("\n\n/////  Begin Epoch {}  /////\n".format(epoch))

      #####  Loop over mini batches  #####
      while True:

        # Get batch data
        try :
          sess.run([self.features, self.labels])
        except tf.errors.OutOfRangeError:
          break

        # Perform training step
        tstep_tic = time.time()
        curLoss, global_step = self.run_train_step(sess, summaryWriter)
        tstep_toc = time.time()
        tstep_time = tstep_toc - tstep_tic

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

        ###  Evaluate model  ###
        if global_step % self.FLAGS.eval_every == 0:

          # Evaluate training data
          trainLoss = self.get_loss(sess, dSet="train")
          trainAccs = self.get_accuracy(sess, dSet="train")

          self.writeSummary(trainLoss, "train/loss", summaryWriter, global_step)
          self.writeSummary(trainAccs, "train/acc", summaryWriter, global_step)
          self.history["train"].append((global_step, trainLoss, trainAccs))

          # Evaluate validation data
          valLoss = self.get_loss(sess, dSet="val")
          valAccs = self.get_accuracy(sess, dSet="val")

          self.writeSummary(valLoss, "val/loss", summaryWriter, global_step)
          self.writeSummary(valAccs, "val/acc", summaryWriter, global_step)
          self.history["val"].append((global_step, valLoss, valAccs))

          # Logging results
          print_info = "%i\tTraining %.5f / %.5f \tValidation %.5f / %.5f" %\
              (global_step, trainLoss, trainAccs, valLoss, valAccs)
          logging.info(print_info)

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


         #self.plot_results()

          if self.FLAGS.verbose:
            pass
            #print("Iteration(%i, %i)\tTrain[ Loss: %f\tAccuracy: %f]\tValidation[ Loss: %f\tAccuracy: %f]" % (epoch, ibt, self.loss_history["train"][-1], self.accuracy_history["train"][-1], self.loss_history["val"][-1], self.accuracy_history["val"][-1]))


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
    Plot the loss and accuracy history on both the train and validation
    datasets after training
    """

    itrs = np.arange(len(self.history.keys()))

    f1, (ax1) = plt.subplots()
    h1, = ax1.plot(itrs, self.history["train"], "b-", label = "Loss - Train")
    h2, = ax1.plot(itrs, self.history["val"], "b.", label = "Loss - Validation")

    ax1.set_ylabel("Loss", color = "blue")
    ax1.tick_params("y", color = "blue")
    ax1.yaxis.label.set_color("blue")
    ax1.set_xlabel("Training Steps [{}]".format(self.FLAGS.eval_every))

    ax2 = ax1.twinx()
    h3, = ax2.plot(itrs, self.accuracy_history["train"], "r-", \
        label = "Accuracy - Train")
    h4, = ax2.plot(itrs, self.accuracy_history["val"], "r.", \
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

    return self.sess.run(outputs, 
              feed_dict = { 
                  self.X_placeHolder : self.data["val_X"], 
                  self.Y_placeHolder : self.data["val_Y"], 
                  self.isTraining_placeHolder : False})
    """

    feed_dict = {self.isTraining : False}

    if (dataX is not None) and (dataY is not None):
      feed_dict[self.features] = dataX
      feed_dict[self.labels]   = dataY
    elif dSet is not None:
      batch = self.get_data(self.data[dSet+"_X"].shape[0], dSet)
      feed_dict[self.inputX]  = batch[0]
      feed_dict[self.inputY]  = batch[1]
    else:
      print("ERROR: Must specify dSet or dataX and dataY for get_accuracy!!!")
      raise RuntimeError
   
    return sess.run(outputs, feed_dict)


  #############################################################################
  def get_predictions(self, sess, dSet=None, dataX=None, dataY=None):
    
    feed_dict = {self.isTraining : False}

    if dataX is not None and dataY is not None:
      feed_dict[self.features] = dataX
      feed_dict[self.labels]   = dataY
      return sess.run([self.prediction], feed_dict) 

    if dSet is "train":
      sess.run(self.eval_iter_train_op)
    elif dSet is "val":
      sess.run(self.eval_iter_valid_op)
    elif dSet is "test":
      sess.run(self.eval_iter_test_op)
    else:
      print("ERROR: Do not recognize dataset {}".format(dSet))
      raise RuntimeError

    try:
      sess.run([self.features, self.labels])
      runningPredictions = sess.run([self.prediction], feed_dict)
    except tf.errors.OutOfRangeError:
      print("Error: Cannot get predictions because dataset is empty")
      raise RuntimeError

    while True:
      # Get batch data
      try :
        sess.run([self.features, self.labels])
        runningPredictions.concatenate(
                sess.run([self.prediction], feed_dict), 
                axis=0)
      except tf.errors.OutOfRangeError:
        break

    return runningPredictions


  #############################################################################
  def get_accuracy(self, sess, dSet=None, dataX=None, dataY=None):

    feed_dict = {self.isTraining : False}

    if dataX is not None and dataY is not None:
      feed_dict[self.features] = dataX
      feed_dict[self.labels]   = dataY
      return sess.run([self.accuracy], feed_dict)[0]

    runningAccuracy = 0
    if dSet is "train":
      sess.run(self.eval_iter_train_op)
      Nsamples = self.Ntrain_samples
    elif dSet is "val":
      sess.run(self.eval_iter_valid_op)
      Nsamples = self.Nvalid_samples
    elif dSet is "test":
      sess.run(self.eval_iter_test_op)
      Nsamples = self.Ntest_samples
    else:
      print("ERROR: Do not recognize dataset {}".format(dSet))
      raise RuntimeError

    while True:
      # Get batch data
      try :
        sess.run([self.features, self.labels])
        runningAccuracy += sess.run([self.accuracy_sum], feed_dict)[0]
      except tf.errors.OutOfRangeError:
        break

    return runningAccuracy/Nsamples


  #############################################################################
  def get_loss(self, sess, dSet=None, dataX=None, dataY=None):

    feed_dict = {self.isTraining : False}

    if dataX is not None and dataY is not None:
      feed_dict[self.features] = dataX
      feed_dict[self.labels]   = dataY
      return sess.run([self.loss], feed_dict)[0]

    if dSet is "train":
      sess.run(self.eval_iter_train_op)
      Nsamples = self.Ntrain_samples
    elif dSet is "val":
      sess.run(self.eval_iter_valid_op)
      Nsamples = self.Nvalid_samples
    elif dSet is "test":
      sess.run(self.eval_iter_test_op)
      Nsamples = self.Ntest_samples
    else:
      print("ERROR: Do not recognize dataset {}".format(dSet))
      raise RuntimeError

    runningLoss = 0
    while True:
      # Get batch data
      try :
        sess.run([self.features, self.labels])
        runningLoss += sess.run([self.loss_sum], feed_dict)[0]
      except tf.errors.OutOfRangeError:
        break

    return runningLoss/self.Ntrain_samples

  
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
      fullName = fileName + "_history_" + str(global_step) + ".pl"
    else:
      fullName = fileName + "_history.pl"

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

