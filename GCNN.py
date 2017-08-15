import numpy as np
import tensorflow as tf
from dataHelper import DataHelper
import time
class GatedCNN(object):

    def __init__(self, conf):
        # tf.reset_default_graph()
        self.paras=[]    
        self.X = tf.placeholder(shape=[conf.batch_size, conf.context_size-1], dtype=tf.int32, name="X")
        self.y = tf.placeholder(shape=[conf.batch_size, 1], dtype=tf.int32, name="y")
        embed = self.create_embeddings(self.X, conf)
        self.conf=conf
        h, res_input = embed, embed

        for i in range(conf.num_layers):
            in_ch = h.get_shape()[-1]
            if not conf.conv_multi:
                filter_size = conf.filter_size if i < conf.num_layers-1 else conf.out_channel
                shape = (conf.filter_h, conf.embedding_size, in_ch, filter_size)
                with tf.variable_scope("layer_%d"%i):
                    conv_w = self.conv_op(h, shape, "linear")
                    conv_v = self.conv_op(h, shape, "gated")
                    h = conv_w * tf.sigmoid(conv_v)
            else:
                all_h=[]
                filter_size = conf.filter_size/len([3,5]) if i < conf.num_layers-1 else conf.out_channel
                shape = (conf.filter_h, conf.embedding_size, in_ch, filter_size)
                for filter_h in [3,5]:  
                    with tf.variable_scope("layer_%d_%d"%(i,filter_h)):
                        
                        conv_w = self.conv_op(h, shape, "linear")
                        conv_v = self.conv_op(h, shape, "gated")
                        sub_h = conv_w * tf.sigmoid(conv_v)
                        all_h.append(sub_h)
                h=tf.concat(all_h,3)
            if i % conf.block_size == 0:
                h += res_input
                res_input = h
        print (h.get_shape())        
        h = tf.reshape(h, (conf.batch_size, -1))
        # h = tf.reshape(h, (-1,(conf.context-1)*conf.embedding_size*conf.out_channel))
    
  
        y_shape = self.y.get_shape().as_list()
        
        self.y = tf.reshape(self.y, (y_shape[0] * y_shape[1], 1))

        if conf.need_full_connected:
            yc_w = tf.get_variable("yc_w", [ h.get_shape()[1], conf.yc_size])#, tf.float32,tf.random_normal_initializer(0.0, 0.1))#, tf.random_normal_initializer(0.0, 0.1))
            yc_b = tf.get_variable("yc_b", [conf.yc_size], tf.float32)#,tf.constant_initializer(0.1))#, tf.constant_initializer(1.0))   
            self.paras.append(yc_w)
            self.paras.append(yc_b)
            m_yc_layer_ori = tf.matmul(h, yc_w) + yc_b      
            m_yc_layer = tf.nn.dropout(m_yc_layer_ori, conf.dropout_keep_prob)

            softmax_w = tf.get_variable("softmax_w", [ conf.yc_size,conf.vocab_size], tf.float32, 
                                    tf.random_normal_initializer(0.0, 0.1))
            softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32, tf.constant_initializer(0))

            self.m_output_layer =tf.matmul(m_yc_layer, softmax_w) + softmax_b
        else:
            softmax_w = tf.get_variable("softmax_w", [ h.get_shape()[1],conf.vocab_size], tf.float32, 
                                               tf.random_normal_initializer(0.0, 0.1))
            softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32, tf.constant_initializer(1.0))

            self.m_output_layer =tf.matmul(h, softmax_w) + softmax_b


       
        #Preferance: NCE Loss, heirarchial softmax, adaptive softmax


        if conf.loss_type=="nce":
            # shape different
            # softmax_w = tf.get_variable("softmax_w", [conf.vocab_size,conf.yc_size], tf.float32, tf.random_normal_initializer(0.0, 0.1))
            # softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32, tf.constant_initializer(1.0))
            
            m_loss = tf.reduce_mean(tf.nn.nce_loss(tf.transpose(softmax_w), softmax_b, m_yc_layer, self.y, conf.num_sampled, conf.vocab_size))
            # self.m_output_layer = tf.matmul(m_yc_layer,softmax_w) + softmax_b 
        elif conf.loss_type=="softmax_cross_entropy":
            tloss=tf.nn.softmax_cross_entropy_with_logits(labels=self.m_output_layer, logits= tf.one_hot(tf.reshape(self.y, [-1]), conf.vocab_size)) 
            m_loss = tf.reduce_mean(tloss) 
        else:
            if tf.__version__ not in [ "1.1.0","1.1.0-rc1","1.2.0"]:
                tloss = tf.nn.seq2seq.sequence_loss_by_example([self.m_output_layer], [tf.reshape(self.y, [-1])], 
                         [tf.ones([conf.batch_size ], dtype=tf.float32)])
            else:
                tloss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.m_output_layer], [tf.reshape(self.y, [-1])], 
                         [tf.ones([conf.batch_size ], dtype=tf.float32)])

            m_loss = tf.reduce_mean(tloss) 
        self.perplexity = tf.exp(m_loss)
        l2_loss=0.0    
        for para in self.paras:
            l2_loss+=tf.nn.l2_loss(para)

        self.reg_loss=l2_loss*conf.l2_lambda
        m_loss+=self.reg_loss
        self.loss=  tf.cond(m_loss>50.0,lambda:  50+m_loss*0.01 ,lambda:m_loss)

        predicted=tf.cast(tf.argmax(self.m_output_layer,dimension=1),dtype=tf.int32)
        self.accuracy=  tf.cast(tf.equal(tf.reshape(predicted, [-1]),tf.reshape(self.y, [-1])),dtype=tf.int32)

        self.percision=  tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.m_output_layer, tf.reshape(self.y, [-1]),1),dtype=tf.float32))
        self.percisionAT3=tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.m_output_layer, tf.reshape(self.y, [-1]),3),dtype=tf.float32))
        self.percisionAT5=tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.m_output_layer, tf.reshape(self.y, [-1]),5),dtype=tf.float32))
        self.percisionAT10=tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.m_output_layer, tf.reshape(self.y, [-1]),10),dtype=tf.float32))
       

        self.global_step = tf.get_variable('global_step_in_class', [], initializer=tf.constant_initializer(0), trainable=False)
        if conf.train_method=="RMS":     
            trainer = tf.train.RMSPropOptimizer(conf.learning_rate)
        elif conf.train_method=="moment":
            trainer = tf.train.MomentumOptimizer(conf.learning_rate, conf.momentum) 
        else:
            trainer = tf.train.AdamOptimizer(conf.learning_rate, conf.momentum) 
        
        gradients = trainer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
        self.optimizer = trainer.apply_gradients(clipped_gradients,global_step= self.global_step)
        # self.create_summaries()
       
    def create_embeddings(self, X, conf):

        embeddings = tf.get_variable("embedding",(conf.vocab_size, conf.embedding_size), tf.float32, tf.random_uniform_initializer(-1.0,1.0))
        self.paras.append(embeddings)
        embed = tf.nn.embedding_lookup(embeddings, X)
        mask_layer = np.ones((conf.batch_size, conf.context_size-1, conf.embedding_size))
        mask_layer[:,0:int(conf.filter_h/2),:] = 0
        embed *= mask_layer
        
        embed_shape = embed.get_shape().as_list()
        embed = tf.reshape(embed, (embed_shape[0], embed_shape[1], embed_shape[2], 1))
        return embed


    def conv_op(self, fan_in, shape, name):
        W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
        b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
        self.paras.append(W)
        self.paras.append(b)
        # return tf.add(tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME'), b)
        return tf.add(tf.nn.conv2d(fan_in, W, strides=[1,1,self.conf.embedding_size,1], padding='SAME'), b)
    
    def create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        self.merged_summary_op = tf.summary.merge_all()

def getFlags():

    flags = tf.app.flags
    flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
    # Flags for defining the tf.train.Server
    flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
    flags.DEFINE_integer("task_index", 0, "Index of task within the job")

    flags.DEFINE_integer("vocab_size", 10000, "Maximum size of vocabulary")
    flags.DEFINE_integer("embedding_size", 32, "Embedding size of each token")
    flags.DEFINE_integer("yc_size", 64, "Size of yc layer")
    flags.DEFINE_integer("filter_size", 32, "Featrue map")
    flags.DEFINE_integer("num_layers", 4, "Number of CNN layers")
    flags.DEFINE_integer("block_size", 2, "when to run residual block")
    flags.DEFINE_integer("filter_h", 3, "Height of the CNN filter")
    flags.DEFINE_integer("context_size", 4, "Length of sentence/context")
    flags.DEFINE_integer("batch_size", 64, "Batch size of data while training")
    flags.DEFINE_integer("epochs", 100, "Number of epochs")
    flags.DEFINE_integer("num_sampled", 1, "Sampling value for NCE loss")
    flags.DEFINE_integer("workernum", 1, "Sampling value for NCE loss")
    flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for training")
    flags.DEFINE_float("l2_lambda", 0, "Learning rate for training")
    flags.DEFINE_float("momentum", 0.99, "Nestrov Momentum value")
    flags.DEFINE_float("grad_clip", 1, "Gradient Clipping limit")
    flags.DEFINE_float("dropout_keep_prob", 1, "dropout_keep_prob")
    flags.DEFINE_integer("num_batches", 0, "Predefined: to be calculated")
    flags.DEFINE_string("ckpt_path", "/search/data/pangshuai/tfmodel/new", "Path to store checkpoints")
    flags.DEFINE_string("data_path", "/root/hhbtensorflow/lib/python2.7/site-packages/tensorflow/models/GCNN", "Path of data")
    flags.DEFINE_string("summary_path", "logs", "Path to store summaries")
    flags.DEFINE_string("train_method", "RMS", "learn_method")
    flags.DEFINE_string("loss_type", "seq2seq", "loss function")
    flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")
    flags.DEFINE_bool("RestoreDic", True, "Restore Dictionary")
    flags.DEFINE_bool("TrainModel", True, "TrainModel Flag")
    flags.DEFINE_bool("conv_multi", False, "TrainModel Flag")
    flags.DEFINE_bool("need_full_connected", True, "TrainModel Flag")

    return flags.FLAGS

def main():
    conf=getFlags()

    helper=DataHelper(conf)
    gcnn=GatedCNN(conf)

    
    
    with tf.Session() as sess:
        if tf.__version__ != "1.1.0": 
            sess.run(tf.initialize_all_variables())
        else:
            sess.run(tf.global_variables_initializer())
        # summary_writer = tf.summary.FileWriter(conf.summary_path, graph=sess.graph)
        for i in range(conf.epochs):
            acc_list=[]

            for batch_x,batch_y in helper.get_batch(shuffle=True):

                start = time.time()

                fetches = [ gcnn.optimizer, gcnn.global_step,gcnn.loss, gcnn.accuracy]
                feed_dict = {gcnn.X:batch_x, gcnn.y:batch_y}
                _,_global_step, cost_val,acc_cur = sess.run(fetches, feed_dict)

                acc_list.append(np.mean(acc_cur))
                end = time.time()
                if _global_step % 2 == 0 :
                    print("epoch :%d step: %.d, Time: %.2f , train_acc: %.2f, Loss: %.2f "#PPL: %.2f    Speed: %.2f,
                            %(i, _global_step,  (end-start), np.mean(acc_cur), cost_val)) #, np.exp(costs / iters)   conf.batch_size * conf.context_size / (end-start),

                # summaries = sess.run(gcnn.merged_summary_op, feed_dict={gcnn.X:batch_x, gcnn.y:batch_y})
                # summary_writer.add_summary(summaries, i) 
            print ("train acc %.2f", np.mean(acc_list))
            dev_acc_list=[]
            for batch_x,batch_y in helper.get_batch(data="valid",shuffle=True):
            
                fetches = [ self.m_output_layer,accuracy]# m_fc2]
                feed_dict = {m_X: batch_x, m_y: batch_y}             

                state ,acc_cur= sess.run(fetches, feed_dict)
            
                dev_acc_list.append(np.mean(acc_cur))               
            

            print("Valid-acc: %2.2f" % (np.mean(dev_acc_list))) 
            self_saver.save(sess, conf.ckpt_file)

            
            dev_acc_list=[]
            for batch_x,batch_y in helper.get_batch(data="test",shuffle=True):
            
                fetches = [ self.m_output_layer,accuracy]# m_fc2]
                feed_dict = {m_X: batch_x, m_y: batch_y}             

                state ,acc_cur= sess.run(fetches, feed_dict)
            
                dev_acc_list.append(np.mean(acc_cur))               
                
                for ii in range(len(state)):

                    x=batch_x[ii]
                    y=batch_y[ii][0]  
                    print (" ".join([helper.idx_to_word[iii] for iii in x])   + "-> "+ helper.idx_to_word[y] +":" + helper.idx_to_word[state[ii].argmax()]  )


            print("Test-acc: %2.2f" % (np.mean(dev_acc_list))) 
            self_saver.save(sess, conf.ckpt_file)



if __name__ == '__main__':
    main()