
# coding: utf-8

# In[8]:


from sklearn import tree
import pydotplus
import os
import numpy as np
import time
import graphviz
from tensorflow.examples.tutorials.mnist import input_data



def main():
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels

    print("x_train_shape:",x_train.shape)
    print("y_train_shape:",y_train.shape)
    print("x_test_shape:",x_test.shape)
    print("y_test_shape:",y_test.shape)
    #print("x_train_head",x_train.head())
    
    start_time=time.clock()
    # 获得一个决策树分类器
    clf = tree.DecisionTreeClassifier(criterion='gini',splitter="random")
    '''
    gini和信息熵的选择问题
    通常选用基尼系数
    数据维度很大，噪音很大使用基尼系数
    维度地，数据比较清晰的时候，信息熵和基尼系数没有区别
    当决策树的拟合程度不够的时候，使用信息熵
    '''
    """
    splitter的选择问题
    best是遍历所有的特征来进行选择，
    random是在随机的特征中选择较大值，
    后者在数据量较大的时候使用
    """
    """
    max_depth是预先剪枝的层数
    """
    """
    min_impurity_split当叶子节点的不纯度小于这个值的时候，
    就不再生长子节点
    """
    # 拟合
    clf.fit(x_train, y_train)
    
    """
    # 保存模型
    with open("iris.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

    # 画图，保存到pdf文件
    # 设置图像参数
    dot_data = tree.export_graphviz(clf, out_file=None,
                             #feature_names=iris.feature_names,
                             class_names=['0','1','2','3','4','5','6','7','8','9'],
                             filled=True, rounded=True,
                             special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 保存图像到pdf文件
    graph.write_pdf("DT.pdf")
    """
    

    
    # 预测
    prediction = clf.predict(x_test)

    accuracy = np.sum(np.equal(prediction, y_test)) / len(y_test)
    #print('prediction : ', prediction)
    print('accuracy : ', accuracy)
    end_time=time.clock()
    print("Time cost:",(end_time-start_time)/60,'minutes')
    return accuracy,(end_time-start_time)/60
    
if __name__ == '__main__':
    main()
    

