# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import pdb
import types
import math

import numpy as np
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_integer("vocab_size", 10000, "Maximum size of vocabulary")
flags.DEFINE_integer("embedding_size", 64, "Embedding size of each token")
flags.DEFINE_integer("yc_size", 128, "Size of yc layer")
flags.DEFINE_integer("f_map", 32, "Featrue map")
flags.DEFINE_integer("num_layers", 10, "Number of CNN layers")
flags.DEFINE_integer("block_yz", 5, "when to run residual block")
flags.DEFINE_integer("filter_h", 3, "Height of the CNN filter")
flags.DEFINE_integer("context_size", 8, "Length of sentence/context")
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
FLAGS = flags.FLAGS



'''
def read_words(conf, file):
    words = []
    #for file in os.listdir(conf.data_path):
    with open(os.path.join(conf.data_path, file), 'r') as f:
        for line in f.readlines():
            tokens = line.split()
            if len(tokens) == conf.context_size-2:
                words.extend((['<pad>']*(int)(conf.filter_h/2)) + ['<s>'] + tokens + ['</s>'])
    return words
'''
def _read_words(conf, filename):  #读取文件， 将换行符替换为 " "， 然后将文件按空格分割。 返回一个 1-D list
    f=open(filename)
    #s = f.read().replace("\n", " ").split()
    content=[]
    for line in f:
        tokens = line.split()
        content.extend((['<pad>']*(int)(conf.filter_h/2)) + ['<s>'] + tokens + ['</s>'])
    f.close()
    return content
def _file_to_word_ids(file, word_to_idx, conf):
    data = _read_words(conf, file)    #_read_words( file)
    return [word_to_idx[word] if word_to_idx.get(word) else 0 for word in data]
	
def index_words(words, conf):
    word_counter = collections.Counter(words).most_common(conf.vocab_size-1)
    word_to_idx = {'<unk>': 0}
    idx_to_word = {0: '<unk>'}
    for i,_ in enumerate(word_counter):
        word_to_idx[_[0]] = i+1
        idx_to_word[i+1] = _[0]
    data = []
    for word in words:
        idx = word_to_idx.get(word)
        idx = idx if idx else word_to_idx['<unk>']
        data.append(idx)
    return np.array(data), word_to_idx, idx_to_word

def create_batches(data, conf):
    num_batches = int(len(data) / (conf.batch_size * conf.context_size))
    data = data[:num_batches * conf.batch_size * conf.context_size]
    xdata = data
    ydata = np.copy(data)
    # print (len(data))
    # print (conf.batch_size)
    # print (conf.context_size)
    # print(num_batches)
    # print (np.array(x_batches).shape)
    # ydata[:-1] = xdata[1:]
    # ydata[-1] = xdata[0]
    x_batches = np.split(xdata.reshape(conf.batch_size, -1), num_batches, 1)
    y_batches = np.split(ydata.reshape(conf.batch_size, -1), num_batches, 1)

    for i in xrange(num_batches):
        x_batches[i] = x_batches[i][:,:-1]
        y_batches[i] = y_batches[i][:,-1]
    return x_batches, y_batches, num_batches

def get_batch(x_batches, y_batches, batch_idx):
    x, y = x_batches[batch_idx], y_batches[batch_idx]
    batch_idx += 1
    if batch_idx >= len(x_batches):
        batch_idx = 0
    return x, y.reshape(-1,1), batch_idx


def prepare_data(conf):
    train_path = os.path.join(conf.data_path, "ptb.train.txt")
    test_path = os.path.join(conf.data_path, "ptb.test.txt")
    valid_path = os.path.join(conf.data_path, "ptb.valid.txt")
    words_total = _read_words(conf, train_path)
    data, word_to_idx, idx_to_word = index_words(words_total, conf)
	
    #words_total = _read_words(conf, train_path)
    #data, word_to_idx, idx_to_word = index_words(words_total, conf)
    #x_batches, y_batches, n_train_batch = create_batches(data, conf)

    train_data = _file_to_word_ids(train_path, word_to_idx, conf)
    x_batches, y_batches, n_train_batch = create_batches(np.array(train_data), conf)
    # for i in range(10):
    #     x=x_batches[0][i]
    #     y=y_batches[0][i]
    #     print(x)
    #     print(y)
    #     print (" ".join([idx_to_word[ii] for ii in x]))
    #     print (idx_to_word[y])
    # exit()
    valid_data = _file_to_word_ids(valid_path, word_to_idx, conf)
    x_batches_val, y_batches_val, n_valid_batch = create_batches(np.array(valid_data), conf)
	
    test_data = _file_to_word_ids(test_path, word_to_idx, conf)
    x_batches_test, y_batches_test, n_test_batch = create_batches(np.array(test_data), conf)
	
    len_voc = len(word_to_idx)+1

    #del words_train
    #del data
    #del valid_data
    #del test_data

    return x_batches, y_batches, n_train_batch, x_batches_val, y_batches_val, n_valid_batch, x_batches_test, y_batches_test, n_test_batch, len_voc, word_to_idx, idx_to_word







def conv_op( fan_in, shape, name):
    W = tf.get_variable("%s_W"%name, shape, tf.float32)#,tf.random_normal_initializer(0.0, 0.1))#, tf.random_normal_initializer(0.0, 0.1))
    b = tf.get_variable("%s_b"%name, shape[-1], tf.float32)#, tf.constant_initializer(0))#, tf.constant_initializer(1.0))
    paras.append(W)
    paras.append(b)
    return tf.add(tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME'), b)


def prepare_conf(conf):
    conf.filter_w = conf.embedding_size
    # conf.context_size += int(math.ceil(conf.filter_h/2))  #~~~~~~~~~~~~~~~~~~~~~~~
    
    # Check if data exists
    if not os.path.exists(conf.data_path):
        exit("Please download the data as mentioned in Requirements")

    # Create paths for checkpointing
    ckpt_model_path = 'vocab%d_embed%d_filters%d_batch%d_layers%d_block%d_fdim%d'%(conf.vocab_size, conf.embedding_size, 
            conf.f_map, conf.batch_size, conf.num_layers, conf.block_yz, conf.filter_h)
    conf.ckpt_path = os.path.join(conf.ckpt_path, ckpt_model_path)

    if not os.path.exists(conf.ckpt_path):
        os.makedirs(conf.ckpt_path)
    conf.ckpt_file = os.path.join(conf.ckpt_path, "m_ckpt")


    return conf 



timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
log_file = 'log/' +"task%d_" %FLAGS.task_index +timeStamp

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
workernum = len(worker_hosts)
conf = prepare_conf(FLAGS)
conf.workernum=workernum


print ("workernum:%d"% workernum)
# Create a cluster from the parameter server and worker hosts.
cluster = tf.train.ClusterSpec({ "ps": ps_hosts, "worker" : worker_hosts })

# start a server for a specific task
server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)


if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":						  
  with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)), open(log_file+"_task%d" %FLAGS.task_index,"w") as log:
    '''		
    print ("Resore dictionary...")
    word_to_id = []
    dic_path = os.path.join(FLAGS.data_path, "ptb_words.dic")	
    #dic_path = os.path.join(FLAGS.data_path, "words.dic")	
    fdic = open(dic_path, "r")
    word_dic = fdic.read().split()
    word_to_id = dict(zip(word_dic, range(len(word_dic))))
    print ("word_to_id_len:%d"% len(word_to_id))
	'''
    x_batches, y_batches, n_train_batch, x_batches_val, y_batches_val, n_valid_batch, x_batches_test, y_batches_test, n_test_batch, conf.vocab_size , word_to_idx, idx_to_word= prepare_data( conf)	
	
   	

 




    #tf.reset_default_graph()
    paras=[]        
    m_X = tf.placeholder(shape=[conf.batch_size, conf.context_size-1], dtype=tf.int32, name="X")
    m_y = tf.placeholder(shape=[conf.batch_size, 1], dtype=tf.int32, name="y")

    #with tf.device("/cpu:0"):	
    embeddings = tf.get_variable("embeding", (conf.vocab_size, conf.embedding_size), tf.float32,tf.random_uniform_initializer(-1.0,1.0))#, tf.random_uniform_initializer(-1.0,1.0))
    
    embed = tf.nn.embedding_lookup(embeddings, m_X)
    mask_layer = np.ones((conf.batch_size, conf.context_size-1, conf.embedding_size))
    mask_layer[:,0:int(math.ceil(conf.filter_h/2)),:] = 0
    embed *= mask_layer
        
    embed_shape = embed.get_shape().as_list()
    embed = tf.reshape(embed, (embed_shape[0], embed_shape[1], embed_shape[2], 1))	
    paras.append(embeddings)
    #embed_res = create_embeddings(m_X, conf)
    h, res_input = embed, embed
    
    for i in range(conf.num_layers):
        in_ch = h.get_shape()[-1]

        if not conf.conv_multi:
            f_map = conf.f_map if i < conf.num_layers-1 else 4
            shape = (conf.filter_h, conf.filter_w, in_ch, f_map)
            with tf.variable_scope("layer_%d"%i):
                conv_w = conv_op(h, shape, "linear")
                conv_v = conv_op(h, shape, "gated")
                h = conv_w * tf.sigmoid(conv_v)
        else:
            all_h=[]
            f_map = conf.f_map/len([3,5]) if i < conf.num_layers-1 else 4
            shape = (conf.filter_h, conf.filter_w, in_ch, f_map)
            for filter_h in [3,5]:  
                with tf.variable_scope("layer_%d_%d"%(i,filter_h)):
                    
                    conv_w = conv_op(h, shape, "linear")
                    conv_v = conv_op(h, shape, "gated")
                    sub_h = conv_w * tf.sigmoid(conv_v)
                    all_h.append(sub_h)
            h=tf.concat(all_h,3)
        if i % conf.block_yz == 0:
            h += res_input
            res_input = h
    # h=tf.reduce_mean(h,3) # need or not
    h = tf.reshape(h, (conf.batch_size, -1))
    h_shape = h.get_shape().as_list()

    yc_w = tf.get_variable("yc_w", [ h_shape[1], conf.yc_size])#, tf.float32,tf.random_normal_initializer(0.0, 0.1))#, tf.random_normal_initializer(0.0, 0.1))
    yc_b = tf.get_variable("yc_b", [conf.yc_size], tf.float32)#,tf.constant_initializer(0.1))#, tf.constant_initializer(1.0))	
    paras.append(yc_w)
    paras.append(yc_b)
	
    m_yc_layer_ori = tf.matmul(h, yc_w) + yc_b  	
    m_yc_layer = tf.nn.dropout(m_yc_layer_ori, conf.dropout_keep_prob)
    # m_yc_layer =tf.nn.sigmoid(m_yc_layer)
	
	
    y_shape = m_y.get_shape().as_list()
    m_y = tf.reshape(m_y, (y_shape[0] * y_shape[1], 1))

    #softmax_w = tf.get_variable("softmax_w", [conf.embedding_size, conf.vocab_size], tf.float32, tf.random_normal_initializer(0.0, 0.1))
       
		
        #m_pred_topK_idx, m_pred_topK = tf.nn.top_k(output_layer, 3)
        #top_k=tf.nn.in_top_k(output_layer, tf.reshape(_targets, [-1]), 3)  
		#correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(target0, [-1]))
        #accuracy = tf.reduce_mean(tf.cast(top_k, tf.float32))
		
		
        #Preferance: NCE Loss, heirarchial softmax, adaptive softmax
    softmax_w = tf.get_variable("softmax_w", [conf.yc_size, conf.vocab_size])#, tf.float32, tf.random_normal_initializer(0.0, 0.1))
    softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32)#, tf.constant_initializer(0.1))

    paras.append(softmax_w)
    paras.append(softmax_b)
    m_output_layer =tf.matmul(m_yc_layer, softmax_w) + softmax_b 
    
    if conf.loss_type=="nce":
        # shape different
        # softmax_w = tf.get_variable("softmax_w", [conf.vocab_size,conf.yc_size], tf.float32, tf.random_normal_initializer(0.0, 0.1))
        # softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32, tf.constant_initializer(1.0))
        
        m_loss = tf.reduce_mean(tf.nn.nce_loss(tf.transpose(softmax_w), softmax_b, m_yc_layer, m_y, conf.num_sampled, conf.vocab_size))
        # m_output_layer = tf.matmul(m_yc_layer,softmax_w) + softmax_b 
    elif conf.loss_type=="softmax_cross_entropy":
        tloss=tf.nn.softmax_cross_entropy_with_logits(labels=m_output_layer, logits= tf.one_hot(tf.reshape(m_y, [-1]), conf.vocab_size)) 
        m_loss = tf.reduce_mean(tloss) 
    else:
        if tf.__version__ not in [ "1.1.0","1.1.0-rc1"]:
            tloss = tf.nn.seq2seq.sequence_loss_by_example([m_output_layer], [tf.reshape(m_y, [-1])], 
                     [tf.ones([conf.batch_size ], dtype=tf.float32)])
        else:
            tloss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([m_output_layer], [tf.reshape(m_y, [-1])], 
                     [tf.ones([conf.batch_size ], dtype=tf.float32)])

        m_loss = tf.reduce_mean(tloss) 
    l2_loss=0.0    
    for para in paras:
        l2_loss+=tf.nn.l2_loss(para)
    m_loss+=l2_loss*conf.l2_lambda
    reg_loss=l2_loss*conf.l2_lambda

    m_loss=  tf.cond(m_loss>50.0,lambda:  50+m_loss*0.01 ,lambda:m_loss)
 

    predicted=tf.cast(tf.argmax(m_output_layer,dimension=1),dtype=tf.int32)   
    accuracy=  tf.cast(tf.equal(tf.reshape(predicted, [-1]),tf.reshape(m_y, [-1])),dtype=tf.int32)

    percision=  tf.reduce_mean(tf.cast(tf.nn.in_top_k(m_output_layer, tf.reshape(m_y, [-1]),1),dtype=tf.float32))
    percisionAT3=tf.reduce_mean(tf.cast(tf.nn.in_top_k(m_output_layer, tf.reshape(m_y, [-1]),3),dtype=tf.float32))
    percisionAT5=tf.reduce_mean(tf.cast(tf.nn.in_top_k(m_output_layer, tf.reshape(m_y, [-1]),5),dtype=tf.float32))
    percisionAT10=tf.reduce_mean(tf.cast(tf.nn.in_top_k(m_output_layer, tf.reshape(m_y, [-1]),10),dtype=tf.float32))
    # tloss = tf.nn.seq2seq.sequence_loss_by_example([m_output_layer], [tf.reshape(m_y, [-1])], 
		  #        [tf.ones([conf.batch_size ], dtype=tf.float32)])
    # m_loss = tf.reduce_sum(tloss) / conf.batch_size 
    
    

   
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    if conf.train_method=="RMS":     
        trainer = tf.train.RMSPropOptimizer(conf.learning_rate)
    elif conf.train_method=="moment":
        trainer = tf.train.MomentumOptimizer(conf.learning_rate, conf.momentum) 
    else:
        trainer = tf.train.AdamOptimizer(conf.learning_rate, conf.momentum) 
    gradients = trainer.compute_gradients(m_loss)
    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    m_optimizer = trainer.apply_gradients(clipped_gradients,global_step= global_step)

    #model = GatedCNN(conf)

    self_saver = tf.train.Saver(tf.trainable_variables())

    
    init_op = tf.initialize_all_variables()
    print("Variables initialized ...")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                global_step=global_step,
                                init_op=init_op)
    sess_config = tf.ConfigProto(allow_soft_placement = True)
    
    '''
    x_batches, y_batches, n_train_batch, x_batches_val, 
    y_batches_val, n_valid_batch, x_batches_test, y_batches_test, 
    n_test_batch, conf.vocab_size
    '''
	
    with sv.prepare_or_wait_for_session(server.target, config = sess_config) as sess:


        # if os.path.exists(conf.ckpt_file):
        #     self_saver.restore(sess, conf.ckpt_file)
        #     print("Model Restored")
        if True:
            from dataHelper import DataHelper
            helper=DataHelper(conf)

            for i in range(conf.epochs):
                acc_list=[]
                percisions=[]
                for batch_x,batch_y in helper.get_batch(shuffle=True):
            
                    start = time.time()

                    fetches = [ m_optimizer, global_step,m_loss, accuracy,percision,percisionAT3,percisionAT5,percisionAT10,reg_loss,clipped_gradients]
                    feed_dict = {m_X:batch_x, m_y:batch_y}
                    _,_global_step, cost_val,acc_cur,p1,p3,p5,p10, _reg_loss,gra= sess.run(fetches, feed_dict)
                    percisions.append([p1,p3,p5,p10])
                    acc_list.append(np.mean(acc_cur))
                    end = time.time()
                    line=("epoch :%d step: %d, Time: %.2f, p@1: %.3f, p@3: %.2f, p@5:%.2f, Loss:%.2f reg:%.6f "#PPL: %.2f    Speed: %.2f,
                                %(i, _global_step,  (end-start),p1,p3,p5, cost_val,_reg_loss)) #, np.exp(costs / iters)   conf.batch_size * conf.context_size / (end-start),

                    log.write(line+"\n")
                    if _global_step % 100 == 0 :                        
                        print(line)
                    if cost_val>20:
                        print ("___")
                        print(line)
                    
                        print ("loss explode")


                line=("Train-acc: %2.5f" % (np.mean(acc_list)))
                print (np.mean(percisions,0))
                log.write(line+"\n")
                print(line)
                percisions=[]
                dev_acc_list=[ ]
                for batch_x,batch_y in helper.get_batch(data="valid"):
            
                    fetches = [ m_output_layer,accuracy,percision,percisionAT3,percisionAT5,percisionAT10]# m_fc2]
                    feed_dict = {m_X: batch_x, m_y: batch_y}             

                    state ,acc_cur,p1,p3,p5,p10= sess.run(fetches, feed_dict)
                    percisions.append([p1,p3,p5,p10])
                    dev_acc_list.append(np.mean(acc_cur))               
         

                line=("Valid-acc: %2.5f" % (np.mean(dev_acc_list))) 
                
                log.write(line+"\n")
                print(line)
                print (np.mean(percisions,0))

                percisions=[]
                dev_acc_list=[]
                for batch_x,batch_y in helper.get_batch(data="test"):
            
                    fetches = [ m_output_layer,accuracy,percision,percisionAT3,percisionAT5,percisionAT10]# m_fc2]
                    feed_dict = {m_X: batch_x, m_y: batch_y}             

                    state ,acc_cur,p1,p3,p5,p10= sess.run(fetches, feed_dict)
                    percisions.append([p1,p3,p5,p10])
                    dev_acc_list.append(np.mean(acc_cur))               
                    
                    for ii in range(len(state)):

                        x=batch_x[ii]
                        y=batch_y[ii][0] 
                        if False: 
                            print (" ".join([helper.idx_to_word[iii] for iii in x])   + "-> "+ helper.idx_to_word[y] +":" + helper.idx_to_word[state[ii].argmax()]  )


                line=("Test-acc: %2.5f" % (np.mean(dev_acc_list))) 
                log.write(line+"\n")
                print(line)
                print (np.mean(percisions,0))
                log.flush()
                self_saver.save(sess, conf.ckpt_file)

            







