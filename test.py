

import numpy as np
import tensorflow as tf

import os,collections
flags = tf.app.flags
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_integer("vocab_size", 10000, "Maximum size of vocabulary")
flags.DEFINE_integer("embedding_size", 64, "Embedding size of each token")
flags.DEFINE_integer("yc_size", 64, "Size of yc layer")
flags.DEFINE_integer("f_map", 64, "Featrue map")
flags.DEFINE_integer("num_layers", 2, "Number of CNN layers")
flags.DEFINE_integer("block_yz", 2, "when to run residual block")
flags.DEFINE_integer("filter_h", 3, "Height of the CNN filter")
flags.DEFINE_integer("context_size", 8, "Length of sentence/context")
flags.DEFINE_integer("batch_size", 128, "Batch size of data while training")
flags.DEFINE_integer("epochs", 100, "Number of epochs")
flags.DEFINE_integer("num_sampled", 1, "Sampling value for NCE loss")
flags.DEFINE_float("learning_rate", 1.0, "Learning rate for training")
flags.DEFINE_float("momentum", 0.99, "Nestrov Momentum value")
flags.DEFINE_float("grad_clip", 0.1, "Gradient Clipping limit")
flags.DEFINE_integer("num_batches", 0, "Predefined: to be calculated")
flags.DEFINE_string("ckpt_path", "/search/data/pangshuai/tfmodel/new", "Path to store checkpoints")
flags.DEFINE_string("data_path", ".", "Path of data")
flags.DEFINE_boolean("TestAccuracy", True, "Test accuracy")
flags.DEFINE_bool("RestoreDic", True, "Restore Dictionary")
flags.DEFINE_bool("TrainModel", True, "TrainModel Flag")
FLAGS = flags.FLAGS


def testGraph():
    import tensorflow as tf
    import numpy as np

    c=tf.constant(value=1)
    #print(assert c.graph is tf.get_default_graph())
    print(c.graph)
    print(tf.get_default_graph())

    g=tf.Graph()
    print("g:",g)
    with g.as_default():
        d=tf.constant(value=2)
        print(d.graph)
        #print(g)

    g2=tf.Graph()
    print("g2:",g2)
    g2.as_default()
    e=tf.constant(value=15)
    print(e.graph)
testGraph()
exit()


def testLoss():
    logits=tf.Variable([[0.5,0.2,0.1],[0.33,0.33,0.34],[0.4,0.3,0.3]])
    labels=tf.Variable([0,1,0])
    weights=tf.ones([3],dtype=tf.float32)
    predicted=tf.cast(tf.argmax(logits,axis=1),dtype=tf.int32)
    predicted=tf.cast(tf.argmax(logits,axis=1),dtype=tf.int32)
    accuracy= tf.reduce_mean(tf.cast(tf.equal(predicted,labels),dtype=tf.float32))
    y=tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[labels],[weights])
    x=tf.nn.softmax_cross_entropy_with_logits(labels=logits, logits= tf.one_hot(labels, 3))
    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        print(sess.run(x))
        print(sess.run(y))
        print(sess.run(accuracy))
testLoss()
exit()

def testtopk():
    logits=tf.Variable([[0.5,0.2,0.1],[0.33,0.32,0.34],[0.4,0.25,0.3]])
    value,index=tf.nn.top_k(logits, 2)
    output=tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, [1,1,1],2),dtype=tf.float32))
    with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())
        print(sess.run(value))
        print(sess.run(index))
        print(sess.run(output))
def testConv():
    input = tf.Variable(tf.random_normal([1,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,1]))
    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())
        print(sess.run(op))
        print(sess.run(op).shape)

testConv()
exit()
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

    # ydata[:-1] = xdata[1:]
    # ydata[-1] = xdata[0]
    print (len(data))
    print (conf.batch_size)
    print (conf.context_size)
    print(num_batches)

    x_batches = np.split(xdata.reshape(conf.batch_size, -1), num_batches, 1)
    y_batches = np.split(ydata.reshape(conf.batch_size, -1), num_batches, 1)

    print (np.array(x_batches).shape)
    
    for i in range(num_batches):

        # print  (y_batches[i])
        x_batches[i] = x_batches[i][:,:-1]
        y_batches[i] = y_batches[i][:,-1]

        # print (y_batches[i])

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
    for i in range(10):
        x=x_batches[0][i]
        y=y_batches[0][i]
        print(x)
        print(y)
        print (" ".join([idx_to_word[ii] for ii in x]))
        print (idx_to_word[y])
    
    valid_data = _file_to_word_ids(valid_path, word_to_idx, conf)
    x_batches_val, y_batches_val, n_valid_batch = create_batches(np.array(valid_data), conf)
    # print (n_valid_batch)
    test_data = _file_to_word_ids(test_path, word_to_idx, conf)
    x_batches_test, y_batches_test, n_test_batch = create_batches(np.array(test_data), conf)
    # print (n_test_batch)
    len_voc = len(word_to_idx)+1

    #del words_train
    #del data
    #del valid_data
    #del test_data

    return x_batches, y_batches, n_train_batch, x_batches_val, y_batches_val, n_valid_batch, x_batches_test, y_batches_test, n_test_batch, len_voc

prepare_data(FLAGS)
def testSLice():
    import tensorflow as tf

    import numpy as np
    x=[[1,2,3],[4,5,6]]
    y=np.arange(24).reshape([2,3,4])
    z=tf.constant([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]],  [[13,14,15],[16,17,18]]])
    sess=tf.Session()
    begin_x=[1,0]        #第一个1，决定了从x的第二行[4,5,6]开始，第二个0，决定了从[4,5,6] 中的4开始抽取
    size_x=[1,2]           # 第一个1决定了，从第二行以起始位置抽取1行，也就是只抽取[4,5,6] 这一行，在这一行中从4开始抽取2个元素
    out=tf.slice(x,begin_x,size_x)
    print (sess.run(out))  #  结果:[[4 5]]

    begin_y=[1,0,0]
    size_y=[1,2,3]
    out=tf.slice(y,begin_y,size_y)   
    print (sess.run(out))  # 结果:[[[12 13 14] [16 17 18]]]

    begin_z=[0,1,1]
    size_z=[-1,1,2] 
    out=tf.slice(z,begin_z,size_z)
    print (sess.run(out))  # size[i]=-1 表示第i维从begin[i]剩余的元素都要被抽取，结果：[[[ 5&nbsp; 6]] [[11 12]] [[17 18]]]

def testTranspose():

    import tensorflow as tf  
      
 
    x = tf.Variable(tf.random_normal([3,2]),name='x')  
      
    model = tf.initialize_all_variables()  
    y = tf.transpose(x)  
    with tf.Session() as session:  
          
        session.run(model)  
        print(session.run(x))
        print(session.run(y))
# testTranspose()