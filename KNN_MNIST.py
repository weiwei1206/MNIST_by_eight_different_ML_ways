
# coding: utf-8

# In[1]:


#coding:utf-8
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier as knn




def main():
    mnist=input_data.read_data_sets("MNIST_data",one_hot=False)

    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    
    
    start_time=time.clock()
    
    
        # Train the model
    clf= knn(n_neighbors=5)
    clf.fit(x_train,y_train)

    # Test the test examples
    prediction = clf.predict(x_test)
    accuracy=np.sum(np.equal(prediction,y_test))/len(y_test)
    print("Accuracy:",accuracy)
 
    
    end_time=time.clock()
    print("Time cost:",(end_time-start_time)/60,"minutes")
    return accuracy,(end_time-start_time)/60
    
if __name__=="__main__":
    main()



