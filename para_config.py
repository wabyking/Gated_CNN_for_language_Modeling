import tensorflow as tf



def getBigParameter():
	flags = tf.app.flags
	flags.DEFINE_string("ps_hosts", "none", "Comma-separated list of hostname:port pairs")
	flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
	# Flags for defining the tf.train.Server
	flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
	flags.DEFINE_integer("task_index", 0, "Index of task within the job")
	flags.DEFINE_integer("vocab_size", 10000, "Maximum size of vocabulary")
	flags.DEFINE_integer("embedding_size", 64, "Embedding size of each token")
	flags.DEFINE_integer("yc_size", 128, "Size of yc layer")
	flags.DEFINE_integer("out_channel", 4, "the out channel of the last conventional layer")
	flags.DEFINE_integer("filter_size", 32, "Featrue map")
	flags.DEFINE_integer("num_layers", 10, "Number of CNN layers")
	flags.DEFINE_integer("block_size", 5, "when to run residual block")
	flags.DEFINE_integer("filter_h", 3, "Height of the CNN filter")
	flags.DEFINE_integer("context_size", 10, "Length of sentence/context")
	flags.DEFINE_integer("batch_size", 128, "Batch size of data while training")
	flags.DEFINE_integer("epochs", 1000, "Number of epochs")
	flags.DEFINE_integer("num_sampled", 1, "Sampling value for NCE loss")
	flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for training")
	flags.DEFINE_float("l2_lambda", 0.0001, "Learning rate for training")
	flags.DEFINE_float("momentum", 0.99, "Nestrov Momentum value")
	flags.DEFINE_float("grad_clip", 0.1, "Gradient Clipping limit")
	flags.DEFINE_float("dropout_keep_prob", 1, "dropout_keep_prob")
	flags.DEFINE_integer("num_batches", 0, "Predefined: to be calculated")
	flags.DEFINE_string("ckpt_path", "/search/data/pangshuai/tfmodel/new", "Path to store checkpoints")
	flags.DEFINE_string("data_path", ".", "Path of data")
	flags.DEFINE_string("train_method", "RMS", "learn_method")
	flags.DEFINE_string("loss_type", "seq2seq", "loss function")#softmax_cross_entropy
	flags.DEFINE_boolean("TestAccuracy", False, "Test accuracy")
	flags.DEFINE_bool("RestoreDic", True, "Restore Dictionary")
	flags.DEFINE_bool("TrainModel", True, "TrainModel Flag")
	flags.DEFINE_bool("conv_multi", True, "multi size of convolution")
	flags.DEFINE_bool("view_cases", False, "view the predicting word and real word")
	flags.DEFINE_bool("multiGPU", False, "multi size of convolution")
	flags.DEFINE_bool("need_full_connected", True, "TrainModel Flag")
	FLAGS = flags.FLAGS
	worker_hosts = FLAGS.worker_hosts.split(",")
	FLAGS.workernum = len(worker_hosts)
	return flags.FLAGS


def getMiddleParameter():
	flags = tf.app.flags
	flags.DEFINE_string("ps_hosts", "none", "Comma-separated list of hostname:port pairs")
	flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
	# Flags for defining the tf.train.Server
	flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
	flags.DEFINE_integer("task_index", 0, "Index of task within the job")
	flags.DEFINE_integer("vocab_size", 10000, "Maximum size of vocabulary")
	flags.DEFINE_integer("embedding_size", 64, "Embedding size of each token")
	flags.DEFINE_integer("yc_size", 128, "Size of yc layer")
	flags.DEFINE_integer("out_channel", 4, "the out channel of the last conventional layer")
	flags.DEFINE_integer("filter_size", 32, "Featrue map")
	flags.DEFINE_integer("num_layers", 4, "Number of CNN layers")
	flags.DEFINE_integer("block_size", 2, "when to run residual block")
	flags.DEFINE_integer("filter_h", 3, "Height of the CNN filter")
	flags.DEFINE_integer("context_size", 8, "Length of sentence/context")
	flags.DEFINE_integer("batch_size", 128, "Batch size of data while training")
	flags.DEFINE_integer("epochs", 1000, "Number of epochs")
	flags.DEFINE_integer("num_sampled", 1, "Sampling value for NCE loss")
	flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training")
	flags.DEFINE_float("l2_lambda", 0.00001, "Learning rate for training")
	flags.DEFINE_float("momentum", 0.99, "Nestrov Momentum value")
	flags.DEFINE_float("grad_clip", 0.1, "Gradient Clipping limit")
	flags.DEFINE_float("dropout_keep_prob", 1, "dropout_keep_prob")
	flags.DEFINE_integer("num_batches", 0, "Predefined: to be calculated")
	flags.DEFINE_string("ckpt_path", "/search/data/pangshuai/tfmodel/new", "Path to store checkpoints")
	flags.DEFINE_string("data_path", ".", "Path of data")
	flags.DEFINE_string("train_method", "RMS", "learn_method")
	flags.DEFINE_string("loss_type", "seq2seq", "loss function")#softmax_cross_entropy
	flags.DEFINE_boolean("TestAccuracy", False, "Test accuracy")
	flags.DEFINE_bool("RestoreDic", True, "Restore Dictionary")
	flags.DEFINE_bool("TrainModel", True, "TrainModel Flag")
	flags.DEFINE_bool("conv_multi", False, "multi size of convolution")
	flags.DEFINE_bool("view_cases", False, "view the predicting word and real word")
	flags.DEFINE_bool("multiGPU", False, "multi size of convolution")
	flags.DEFINE_bool("need_full_connected", True, "TrainModel Flag")
	FLAGS = flags.FLAGS
	worker_hosts = FLAGS.worker_hosts.split(",")
	FLAGS.workernum = len(worker_hosts)
	return flags.FLAGS

def getSmallParameter():
	flags = tf.app.flags
	flags.DEFINE_string("ps_hosts", "none", "Comma-separated list of hostname:port pairs")
	flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
	# Flags for defining the tf.train.Server
	flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
	flags.DEFINE_integer("task_index", 0, "Index of task within the job")
	flags.DEFINE_integer("vocab_size", 10000, "Maximum size of vocabulary")
	flags.DEFINE_integer("embedding_size", 32, "Embedding size of each token")
	flags.DEFINE_integer("yc_size", 128, "Size of yc layer")
	flags.DEFINE_integer("out_channel", 4, "the out channel of the last conventional layer")
	flags.DEFINE_integer("filter_size", 32, "Featrue map")
	flags.DEFINE_integer("num_layers", 2, "Number of CNN layers")
	flags.DEFINE_integer("block_size", 2, "when to run residual block")
	flags.DEFINE_integer("filter_h", 3, "Height of the CNN filter")
	flags.DEFINE_integer("context_size", 8, "Length of sentence/context")
	flags.DEFINE_integer("batch_size", 64, "Batch size of data while training")
	flags.DEFINE_integer("epochs", 1000, "Number of epochs")
	flags.DEFINE_integer("num_sampled", 1, "Sampling value for NCE loss")
	flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training")
	flags.DEFINE_float("l2_lambda", 0.00001, "Learning rate for training")
	flags.DEFINE_float("momentum", 0.99, "Nestrov Momentum value")
	flags.DEFINE_float("grad_clip", 0.1, "Gradient Clipping limit")
	flags.DEFINE_float("dropout_keep_prob", 1, "dropout_keep_prob")
	flags.DEFINE_integer("num_batches", 0, "Predefined: to be calculated")
	flags.DEFINE_string("ckpt_path", "/search/data/pangshuai/tfmodel/new", "Path to store checkpoints")
	flags.DEFINE_string("data_path", ".", "Path of data")
	flags.DEFINE_string("train_method", "RMS", "learn_method")
	flags.DEFINE_string("loss_type", "seq2seq", "loss function")#softmax_cross_entropy
	flags.DEFINE_boolean("TestAccuracy", False, "Test accuracy")
	flags.DEFINE_bool("RestoreDic", True, "Restore Dictionary")
	flags.DEFINE_bool("TrainModel", True, "TrainModel Flag")
	flags.DEFINE_bool("conv_multi", False, "multi size of convolution")
	flags.DEFINE_bool("view_cases", False, "view the predicting word and real word")
	flags.DEFINE_bool("multiGPU", False, "multi size of convolution")
	flags.DEFINE_bool("need_full_connected", True, "TrainModel Flag")
	FLAGS = flags.FLAGS
	worker_hosts = FLAGS.worker_hosts.split(",")
	FLAGS.workernum = len(worker_hosts)
	return flags.FLAGS