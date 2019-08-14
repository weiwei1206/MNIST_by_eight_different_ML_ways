
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
import pickle  #加载文件封装为对象
import gzip  #解压数据
import numpy as np
from sklearn import svm
import time



def load_data():

    #返回包含训练数据、验证数据、测试数据的元组的模式识别数据
    #训练数据包含50，000张图片，测试数据和验证数据都只包含10,000张图片
  
    f = gzip.open('./MNIST_data/mnist.pkl.gz')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')  #从字节对象中读取被封装的对象并返回
    f.close()
    return (training_data, validation_data, test_data)




def svm_baseline():
    start_time=time.clock()
    training_data, validation_data, test_data = load_data()
    # 传递训练模型的参数，这里用默认的参数
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)
    # clf = svm.SVC(C=8.0, kernel='rbf', gamma=0.00,cache_size=8000,probability=False)
    # 进行模型训练
    clf.fit(training_data[0], training_data[1])  #clf是分类器哦
    # test
    # 测试集测试预测结果
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print ("Accuracy rate:",num_correct/len(test_data[1]),'%')
    end_time=time.clock()
    print("Time cost:",(end_time-start_time))
 
svm_baseline()

"""


# In[1]:


import numpy as np
import time
from sklearn import svm
from tensorflow.examples.tutorials.mnist import input_data



def main():
    
    mnist=input_data.read_data_sets('MNIST_data',one_hot=False)

    x_train=mnist.train.images
    y_train=mnist.train.labels
    x_test=mnist.test.images
    y_test=mnist.test.labels
    
    start_time=time.clock()
    
    clf=svm.SVC(C=100.,kernel='rbf',gamma=0.03)
    clf.fit(x_train,y_train)
    prediction=clf.predict(x_test)
    
    accuracy=np.sum(np.equal(prediction,y_test))/len(y_test)
    print("Accuracy:",accuracy)
    
    end_time=time.clock()
    print("Time cost:",(end_time-start_time)/60,"minutes")
    return accuracy,(end_time-start_time)/60
    
if __name__=='__main__':
    main()

