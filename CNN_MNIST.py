
# coding: utf-8

# In[1]:



# coding: utf-8

# In[3]:


#-*- coding:utf8 -*-
import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


"""
权重初始化
初始化为一个接近０的很小的正数
"""

def weight_variable(shape,name=None):
    initial=tf.truncated_normal(shape,stddev=0.1)  #tf.truncated_normal(shape, mean, stddev)
                                                   #shape表示生成张量的维度，mean是均值，stddev是标准差。
                                                   #这个函数产生正太分布，均值和标准差自己设定。
    return tf.Variable(initial,name=name)

def bias_variable(shape,name=None):
    initial=tf.constant(0.1,shape=shape)  #第一个参数是值
    return tf.Variable(initial,name=name)

"""
卷积和池化，使用卷积步长为1（stride size）,0边距（padding size）
池化用简单传统的2x2大小的模板做max pooling
"""

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # x(input)  : [batch, in_height, in_width, in_channels]
    # W(filter) : [filter_height, filter_width, in_channels, out_channels]
    # strides   : The stride of the sliding window for each dimension of input.
    #             For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # x(value)              : [batch, height, width, channels]
    # ksize(pool大小)        : A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides(pool滑动大小)   : A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
"""
输入层
"""
with tf.name_scope("input_layer"):
    x=tf.placeholder(tf.float32,[None,784],name="x_input")
    x_image=tf.reshape(x,[-1,28,28,1])
    #y=tf.placeholder(tf.float32,[None,10])


"""
第一层　卷积层
"""
with tf.name_scope('conv_layer1'):
    W_conv1=weight_variable([5,5,1,32],name='w1')
    b_conv1=bias_variable([32],name='b1')
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1,name='activate_layer1')
# x_image -> [batch, in_height, in_width, in_channels]
#            [batch, 28, 28, 1]
# W_conv1 -> [filter_height, filter_width, in_channels, out_channels]
#            [5, 5, 1, 32]
# output  -> [batch, out_height, out_width, out_channels]
#            [batch, 28, 28, 32]

"""
第一层 池化层
"""
with tf.name_scope('pooling_layer1'):
    h_pool1=max_pool_2x2(h_conv1)
# h_conv1 -> [batch, in_height, in_weight, in_channels]
#            [batch, 28, 28, 32]
# output  -> [batch, out_height, out_weight, out_channels]
#            [batch, 14, 14, 32]



"""
第二层　卷积层
"""
with tf.name_scope('conv_layer2'):
    W_conv2=weight_variable([5,5,32,64],name='w2')
    b_conv2=bias_variable([64],name='b2')
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2,name='activate_layer2')
# h_pool1 -> [batch, 14, 14, 32]
# W_conv2 -> [5, 5, 32, 64]
# output  -> [batch, 14, 14, 64]

"""
第二层池化
"""
with tf.name_scope("pooling_layer2"):
    h_pool2=max_pool_2x2(h_conv2)
# h_conv2 -> [batch, 14, 14, 64]
# output  -> [batch, 7, 7, 64]


"""
全链接层
"""
with tf.name_scope("full_connect_layer"):
    W_fc1=weight_variable([7*7*64,1024],name='w_fc1')
    b_fc1=bias_variable([1024],name='b_fc1')

    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64],name="flat_layer")
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1,name='activate_layer3')

"""
Dropout
"""
with tf.name_scope("Dropout_layer"):
    keep_prob=tf.placeholder("float")
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

"""
第四层 Softmax输出层
"""
with tf.name_scope("Softmax_output_layer"):
    W_fc2 = weight_variable([1024, 10],name="W_fc2")
    b_fc2 = bias_variable([10],name="b_fc2")

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name="soft_max")

"""
训练和评估模型

ADAM优化器来做梯度最速下降,feed_dict中加入参数keep_prob控制dropout比例
"""
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv)) #计算交叉熵  reduce_sum是求和函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #使用adam优化器来以0.0001的学习率来进行微调
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #判断预测标签和实际标签是否匹配 argmax返回最大值的索引，0按列来，1按行来
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))  #cast均值转化函数 reduce_mean均值函数

def main():
    with tf.Session() as sess:
        writer=tf.summary.FileWriter("/home/ww/SCHOOL/curriculum/DS/TheWork",sess.graph)
        #sess = tf.Session() #启动创建的模型
        sess.run(tf.global_variables_initializer()) #初始化变量
        
        
        for i in range(2001): #开始训练模型，循环训练5000次
            start_time=time.clock()
            batch = mnist.train.next_batch(50) #batch大小设置为50
            if i % 100 == 0:
                train_accuracy = accuracy.eval(session = sess,
                                               feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
                print("step %d, train_accuracy %g" %(i, train_accuracy))
            train_step.run(session = sess, feed_dict = {x:batch[0], y_:batch[1],
                           keep_prob:0.5}) #神经元输出保持不变的概率 keep_prob 为0.5


        Accuracy=accuracy.eval(session = sess,  #评估函数，即一个图投入值，然后训练
                              feed_dict = {x:mnist.test.images, y_:mnist.test.labels,
                                           keep_prob:1.0}) #神经元输出保持不变的概率 keep_prob 为 1，即不变，一直保持输出
        print("Accuracy: ",Accuracy)
        end_time = time.clock() #计算程序结束时间
        print("Time cost:",(end_time-start_time)/60)
        return Accuracy,(end_time-start_time)/60
        
if __name__=='__main__':
    main()


