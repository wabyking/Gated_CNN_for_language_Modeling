# GCNN


### code说明
* 整个代码被拆分成几个部分
    - 模型部分 
        + GCNN.py 将模型相关的封装成类，同时输入一个配置信息
    - 数据预处理部分
        + dataHelper 将数据的预处理，batch划分，subsample等策略都加到这里面
    - 驱动函数部分
        + work.py 驱动所有模块进行运行，训练验证和测试模块如何协作，同时实时写日志信息，并监控每个步骤的运行时间
    - 参数配置部分
        + para_config.py 将参数文件直接直接封装在文件里，同时提供Small、Middle和Big三种不同规模的参数放到一起
以上各模块都有专门的单元测试

除此之外，代码兼容
    - 不同版本的Tensorflow，可运行在Linux平台和Windows平台
    - CPU和GPU版本可以直接不该一行代码直接兼容，多机多GPU和在本地测试都可以改一次参数，多处可以跑
    - 不同的配置不需要修改源代码，可以直接作为参数写在命令行里
    - 不同数据集可以直接喂给模型，（中文只要分词之后以空格隔开放到文件里）

同时代码还有
    - 实时记录日志，程序如果崩溃，也可以将中间结果保存起来
    - 监控每一个步骤的运行时间，改动参数时同时观察时间的变化
    - 可以直接输出预测的实际case，分析哪些结果好，哪些的结果不好

尽量最大程度上复用代码。
        

###调参说明


*  样本喂入方式的改变，最开始的版本的数据喂入方式没有以自然句子切分，过多地预测标记<s> <pad> </s>，导致结果比较虚高。现在以单个句子以窗口移动的方式做样本数据，一方面增大了数据量，另一方面减少了一些无用的预测。原结果在新数据集的设置下为0.18，现在能做到0.23左右。
*  原版本有很多导致网络不收敛的原因，通过多种策略让网络更深更宽的时候尽可能收敛。最主要的原因是卷积层最后一个out_channel设置为1导致网络表达能力蜕化，还有一些情况会导致loss爆炸的问题，单方面地减梯度也不能解决问题，做一个loss平滑的方法，loss大于一个阈值会有一个折扣。如代码所示
<code> self.loss=  tf.cond(m_loss>50.0,lambda:  50+m_loss*0.01 ,lambda:m_loss)</code>
* 网络训练的经验第一步是让训练集过拟合，然后再抗过拟合。暂时用深卷积层、较多卷积核的数量和变长卷积窗口，可以是训练集的top1 的准确率提升到40%。
* 抗过拟合的问题可以从增大数据样本、加大正则项、加大dropout率的角度上看。由于大网络的参数太大，没有足够时间让我好好调参数，目前的的经验是正则项是最好调的，但是测试集合的结果一致不能超过0.23. 究其问题可能来自于数据样本少的问题。第二个dropout的参数设置的问题，该设置依赖于最后一层的卷积out_channel和全连接层的设置，如果这卷积层和全连接设置不合理的时候，dropout有可能其恶化作用。dropout策略暂时用在最后的全连接层
* 对于卷积的一些问题，暂时的卷积方式应该不是很合理，其实做的是一个一维卷积，也就是说在卷积的时候应该采取strides第三维用embedding的长度，而不是1。在走的时候才发现这一点，代码应该以这个基础接着去修改
<code>tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME')</code>
原版本为
<code>tf.nn.conv2d(fan_in, W, strides=[1,1,self.conf.embedding_size,1], padding='SAME')</code>
假如我们使用了宽度为embedding_size的步长，我们应该同时增大最后一层卷积的out_channel的大小，其具体效果来不及去测试，但是时间基本上可以加快很多。
* 损失函数，seq2seq 收敛性更好，文章用的是adative_softmax, 也是来自来自于Facebook AI 实验室 [1] ，实验代码是lua的[2]，腾讯AI实验室自己fork的tensorflow实现了adative softmax，示例代码见https://github.com/TencentAILab/tf-adaptive-softmax-lstm-lm。 但是该TF库暂时支持TF 1.0，如果想用可以考虑用该损失函数
* 对于初始化，TF 0.12 的get_Variable的默认初始化是uniform_unit_scaling_initializer，该方法并不好，卷积的初始化可以考虑用tf.contrib.layers.xavier_initializer() 或者考虑TF后续版本的默认初始化glorot_uniform_initializer。
* 正则化因子lambda,设置大了会欠拟合，小了会过拟合。但是暂时的观察发现是，结果在参数规模不大的时候，适当地过拟合效果会更好。但是在新的场景可能结论并不一致
* Dropout在全连接层用，如果全连接层节点数（代码中的yc_size参数）不多，还用比较大的dropout率，会导致结果不收敛。当网络更加复杂，参数更多的时候，应该加大lambda，如果在小参数配置下，设置为0.00001，大参数的版本下设置为0.0001（配套修改学习率，大参数配置下的学习率应该最小，这个经验不一样合适吧，初步观察是这样的）
* 数据shuffle，不同epoch里面的数据样例，如果能shuffle尽量做一下，可以避免过拟合。
* Subsample，这个思路借鉴自word2vec，过多地训练一些高频词，比较浪费时间，对高频词做一个采样。采样概率代码所示：
<code>t=10000.0/counter_sum
slice = [random.random()> 1-math.sqrt(t*1.0/prob[i]) if prob[i]>t else True for i in answers]</code> 
主要做的就是多高频词做丢弃，频率越高越容易被丢弃。 
* 最后一层卷积层的out_chanel，不要设置成1 吧
* 梯度切除很有必要，代码里也做了一个loss切除，有时候梯度爆炸了，梯度切除加loss配套用更好
* 变长卷积窗口可以增加网络的表达能力和效果，但是速度会变慢, 预测时间也会变慢，这个需要权衡。
* 如果加深卷积深度，需暂时的观察室残差块的大小也对应增大，这个经验并不一定恰当，这个地方的参数没有调太细
* 修正了代码复杂的和混乱的维度shape的操作，使逻辑更加清晰
* 做一些ablation的操作，测试每一个模块的实际效果如何，先约减网络结构再有方向地使其更复杂有好的表达能力。比如最后的全连接层的实际效果可以拆除掉比较效果。
* Percision@1 效果不稳定的时候，同时得到Percision@1 Percision@3 Percision@5 的batch内评价效果。
* 输出信息实时监控单个batch的正则损失和实际损失。


###

sogou输入法实习生王本友
mailto:waby@tju.edu.cn






















[1] https://arxiv.org/pdf/1609.04309v2.pdf
[2] https://github.com/facebookresearch/adaptive-softmax

