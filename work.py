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
import para_config
from dataHelper import DataHelper
from GCNN import GatedCNN
import numpy as np
import tensorflow as tf



FLAGS=para_config.getSmallParameter()

from functools import wraps
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco


def work(sess,gcnn,FLAGS,helper,log):

    # saver = tf.train.Saver(tf.trainable_variables())
    for i in range(FLAGS.epochs): 

        train_percisions=train(sess,gcnn,FLAGS,helper,log,i)
        print (train_percisions)

        dev_precisions =test(sess,gcnn,FLAGS,helper,log,"valid")
        test_precisions =test(sess,gcnn,FLAGS,helper,log,"test")
        print ("train-acc: %.4f valid-acc: %.4f test valid-acc %.4f"%(train_percisions[0],dev_precisions[0],test_precisions[0]))
        print (dev_precisions)


@log_time_delta
def train(sess,gcnn,FLAGS,helper,log,epoch):
 
    percisions=[]

    for batch_x,batch_y in helper.get_batch(shuffle=True):
    
        start = time.time()        
        feed_dict = {gcnn.X:batch_x, gcnn.y:batch_y}
        fetches = [ gcnn.optimizer, gcnn.global_step,gcnn.loss, gcnn.percision, gcnn.percisionAT3,gcnn.percisionAT5,gcnn.percisionAT10,gcnn.reg_loss]
        
        _,_global_step, cost_val,p1,p3,p5,p10, _reg_loss= sess.run(fetches, feed_dict)
        percisions.append([p1,p3,p5,p10])
        
        end = time.time()
        line=("epoch :%d step: %d, Time: %.2f, p@1: %.3f, p@3: %.2f, p@5:%.2f, Loss:%.2f reg:%.6f "#PPL: %.2f    Speed: %.2f,
                    %(epoch, _global_step,  (end-start),p1,p3,p5, cost_val,_reg_loss)) 
        log.write(line+"\n")
        log.flush()
        if  FLAGS.ps_hosts=="none"or _global_step % 100 == 0 :           
            print(line)
            
    return np.mean(percisions,0)

@log_time_delta
def test(sess,gcnn,FLAGS,helper,log,dataset):
    percisions=[]
    
    results=[]
  
    for batch_x,batch_y in helper.get_batch(data=dataset):

        fetches = [ gcnn.m_output_layer,gcnn.percision,gcnn.percisionAT3,gcnn.percisionAT5,gcnn.percisionAT10]# m_fc2]
        feed_dict = {gcnn.X: batch_x, gcnn.y: batch_y}             
       
        state ,p1,p3,p5,p10= sess.run(fetches, feed_dict)
        percisions.append([p1,p3,p5,p10])
        
        if FLAGS.view_cases:
            for ii in range(len(state)):
                x=batch_x[ii]
                y=batch_y[ii][0]                                 
                print (" ".join([helper.idx_to_word[iii] for iii in x])   + "-> "+ helper.idx_to_word[y] +":" + helper.idx_to_word[state[ii].argmax()]  )
 
    return (np.mean(percisions,0))               



def main():
    timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
    log_file = 'log/' +"task%d_" %FLAGS.task_index +timeStamp
    if FLAGS.ps_hosts != "none":
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")
        workernum = len(worker_hosts)
        print ("workernum:%d"% workernum)
        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec({ "ps": ps_hosts, "worker" : worker_hosts })
        # start a server for a specific task
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


        if FLAGS.job_name == "ps":
            server.join()
        elif FLAGS.job_name == "worker":                          
            with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)), open(log_file+"_task%d" %FLAGS.task_index,"w") as log:


                if True:

                    helper=DataHelper(FLAGS)
                    gcnn=GatedCNN(FLAGS)

                    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
                    init_op = tf.global_variables_initializer()
                    
                    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),global_step=global_step,init_op=init_op)
                    sess_config = tf.ConfigProto(allow_soft_placement = True)

                    with sv.prepare_or_wait_for_session(server.target, config = sess_config) as sess:                    
                        work(sess,gcnn,FLAGS,helper,log)
                


    else:
        helper=DataHelper(FLAGS)
        gcnn=GatedCNN(FLAGS)
        with tf.Session() as sess, open(log_file+"_task_cpu","w") as log:           
            
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            work(sess,gcnn,FLAGS,helper,log)

if __name__ == '__main__':
    main()