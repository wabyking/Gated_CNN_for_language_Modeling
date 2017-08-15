import numpy as np 
import os,collections
import pandas as pd
from collections import Counter
import random,math
class DataHelper():
	def __init__(self,conf):
		self.conf=conf
		self.dataset={}

		self.dataset["train"] = os.path.join(".", "ptb.train.txt")
		self.dataset["test"] = os.path.join(".", "ptb.test.txt")
		self.dataset["valid"] = os.path.join(".", "ptb.valid.txt")

		words_total = self.read_words(self.dataset.values())
		self.word_to_idx, self.idx_to_word = self.index_words(words_total)

	def read_words(self,files,fixed=False):
		words = []

		for filename in files:

			with open(filename, 'r') as f:
				for line in f:
					tokens = line.split()
					# NOTE Currently, only sentences with a fixed size are chosen
					# to account for fixed convolutional layer size.
					if fixed:
						if len(tokens) == self.conf.context_size-2:
							words.extend((['<pad>']*(conf.filter_h/2)) + ['<s>'] + tokens + ['</s>'])
					else:
						words.extend(tokens)
		return words
	    	

	def index_words(self,words):
		print ("word count :%d", len( collections.Counter(words)))
		word_counter = collections.Counter(words).most_common(self.conf.vocab_size-4)

		word_to_idx = {token:i for i,token in enumerate(['<unk>','<pad>','<s>','</s>']) }
		idx_to_word = {i:token for i,token in enumerate(['<unk>','<pad>','<s>','</s>']) }

		for i,_ in enumerate(word_counter):
			word_to_idx[_[0]] = i+4
			idx_to_word[i+4] = _[0]
		
		return word_to_idx, idx_to_word
	def get_batch(self, data="train",shuffle=True,full=True,sample=False):
		pairs=[]
		with open(self.dataset[data]) as f:
			for line in f:
				tokens=line.split()
				# print (line)
				padding_tokens=['<pad>']*int(self.conf.filter_h/2) + ['<s>'] + tokens + ['</s>']
				ids= [ self.word_to_idx.get(word,0) for word in padding_tokens]
				# print(padding_tokens)
				# print (ids)
				for i in range(0,len(padding_tokens)-self.conf.context_size+1):
					pair=(ids[i:i+self.conf.context_size-1],[ids[i+self.conf.context_size-1]])
					pairs.append(pair)
					# print (pair)

		
		# print (pairs)
		if data=="train":
			if not full:
				if not sample:
					each_task_size= int(len(pairs)/self.conf.workernum) if self.conf.ps_hosts!="none" else int(len(pairs))

					pairs= pairs[self.conf.task_index*each_task_size:(1+self.conf.task_index)*each_task_size]
				else:
				# pairs= pairs[self.conf.task_index*each_task_size:(1+self.conf.task_index)*each_task_size]
					answers=[item[0]for item in pd.DataFrame(pairs)[1].values]
					counter=Counter(answers)
					counter_sum=sum(counter.values())
					# print(counter_sum)
					# print(counter)
					prob= {k:v*1.0 /counter_sum for k,v in counter.items() }
					t=10000.0/counter_sum
					slice = [random.random()> 1-math.sqrt(t*1.0/prob[i]) if prob[i]>t else True for i in answers]
					# print ([ 1-math.sqrt(t*1.0/prob[i]) if prob[i]>t else True for i in answers])

					pairs=(np.array(pairs)[slice])

					# probs = [counter[i] for i in answers]
					# sampled_p= (12-np.log(probs))
					# sampled_p=sampled_p/np.sum(sampled_p)

					# slice=np.random.choice(range(len(pairs)),size=int(len(pairs)/4),p=sampled_p,replace=False)

					# pairs=(np.array(pairs)[slice])


				# pairs=np.random.choice(pairs,size=int(len(pairs)/4),p=sampled_p,replace=False)
				# print(slice)
				# pairs=pairs[np.array(slice)]
			if shuffle:
				pairs = np.random.permutation(pairs)
		if data=="train":
			batch_size=self.conf.batch_size
		else:
			# batch_size=self.conf.batch_size* 4 if self.conf.batch_size<64 else 256
			batch_size=self.conf.batch_size
		n_batches= int(len(pairs)/ batch_size)
		for i in range(0,n_batches):
			batch = pairs[i*batch_size:(i+1) * batch_size]

			yield [[pair[j] for pair in batch]  for j in range(2)]
		batch= pairs[-1*batch_size:] 
		yield [[pair[i] for pair in batch]  for i in range(2)]

def getTestFlag():
	import tensorflow as tf


	flags = tf.app.flags
	flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
	flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
	# Flags for defining the tf.train.Server
	flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
	flags.DEFINE_integer("task_index", 0, "Index of task within the job")

	flags.DEFINE_integer("vocab_size", 10000, "Maximum size of vocabulary")
	flags.DEFINE_integer("embedding_size", 64, "Embedding size of each token")
	flags.DEFINE_integer("yc_size", 64, "Size of yc layer")
	flags.DEFINE_integer("f_map", 32, "Featrue map")
	flags.DEFINE_integer("num_layers", 2, "Number of CNN layers")
	flags.DEFINE_integer("block_yz", 2, "when to run residual block")
	flags.DEFINE_integer("filter_h", 3, "Height of the CNN filter")
	flags.DEFINE_integer("context_size", 8, "Length of sentence/context")
	flags.DEFINE_integer("batch_size", 128, "Batch size of data while training")
	flags.DEFINE_integer("epochs", 100, "Number of epochs")
	flags.DEFINE_integer("num_sampled", 1, "Sampling value for NCE loss")
	flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training")
	flags.DEFINE_float("momentum", 0.99, "Nestrov Momentum value")
	flags.DEFINE_float("grad_clip", 0.1, "Gradient Clipping limit")
	flags.DEFINE_integer("num_batches", 0, "Predefined: to be calculated")
	flags.DEFINE_string("ckpt_path", "/search/data/pangshuai/tfmodel/new", "Path to store checkpoints")
	flags.DEFINE_string("data_path", "/root/hhbtensorflow/lib/python2.7/site-packages/tensorflow/models/GCNN", "Path of data")
	flags.DEFINE_string("train_method", "RMS", "learn_method")
	flags.DEFINE_string("loss_type", "seq2seq", "loss function")
	flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")
	flags.DEFINE_bool("RestoreDic", True, "Restore Dictionary")
	flags.DEFINE_bool("TrainModel", True, "TrainModel Flag")
	FLAGS = flags.FLAGS
	FLAGS.workernum=4
	return flags.FLAGS

def main():
	FLAGS=getTestFlag()
	helper=DataHelper(FLAGS)
	i=0
	for batch in helper.get_batch("train"):
		i+=1

		print (i)
if __name__ == '__main__':
	main()
	pass


