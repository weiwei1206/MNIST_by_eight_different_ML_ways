
# coding: utf-8

# In[ ]:


import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import CNN_MNIST
import SVM_MNIST
import KNN_MNIST
import NB_MNIST
import DT_MNIST
#import test

class Lab_UI(QTabWidget):
    def __init__(self,parent=None):
        super(QTabWidget, self).__init__(parent)

        #窗口名字
        self.setWindowTitle("Images Classifier")
        
        #创建5个选项卡小控件窗口
        self.tab1=QWidget()
        self.tab2=QWidget()
        self.tab3=QWidget()
        self.tab4=QWidget()
        self.tab5=QWidget()

        #将五个选项卡添加到顶层窗口中
        self.addTab(self.tab1, "Tab 1")
        self.addTab(self.tab2, "Tab 2")
        self.addTab(self.tab3, "Tab 3")
        self.addTab(self.tab4, "Tab 4")
        self.addTab(self.tab5, "Tab 5")

        #每个选项卡自定义的内容
        self.tab1UI()
        self.tab2UI()
        self.tab3UI()
        self.tab4UI()
        self.tab5UI()
        
    def compute1(self):
        ac_sum=0;
        ti_sum=0;
        times=self.CNN_le.text()
        if times=='':
            times='0'
        for i in range(int(times)):
            ac,ti= CNN_MNIST.main()
            ac_sum+=ac
            ti_sum+=ti
        ac=ac_sum/int(times)
        ti=ti_sum/int(times)
        
        self.CNN_ac_label.setText(str(ac))
        self.CNN_ti_label.setText(str(ti))
        
    def compute2(self):
        ac_sum=0;
        ti_sum=0;
        times=self.SVM_le.text()
        if times=='':
            times='0'
        for i in range(int(times)):
            ac,ti= SVM_MNIST.main()
            ac_sum+=ac
            ti_sum+=ti
        ac=ac_sum/int(times)
        ti=ti_sum/int(times)
        
        self.SVM_ac_label.setText(str(ac))
        self.SVM_ti_label.setText(str(ti))
        
    def compute3(self):
        ac_sum=0;
        ti_sum=0;
        times=self.KNN_le.text()
        if times=='':
            times='0'
        for i in range(int(times)):
            ac,ti= KNN_MNIST.main()
            ac_sum+=ac
            ti_sum+=ti
        ac=ac_sum/int(times)
        ti=ti_sum/int(times)
        
        self.KNN_ac_label.setText(str(ac))
        self.KNN_ti_label.setText(str(ti))
        
    def compute4(self):
        ac_sum=0;
        ti_sum=0;
        times=self.NB_le.text()
        if times=='':
            times='0'
        for i in range(int(times)):
            ac,ti= NB_MNIST.main()
            ac_sum+=ac
            ti_sum+=ti
        ac=ac_sum/int(times)
        ti=ti_sum/int(times)
        
        self.NB_ac_label.setText(str(ac))
        self.NB_ti_label.setText(str(ti))
        
    def compute5(self):
        ac_sum=0;
        ti_sum=0;
        times=self.DT_le.text()
        if times=='':
            times='0'
        for i in range(int(times)):
            ac,ti= DT_MNIST.main()
            ac_sum+=ac
            ti_sum+=ti
        ac=ac_sum/int(times)
        ti=ti_sum/int(times)
        
        self.DT_ac_label.setText(str(ac))
        self.DT_ti_label.setText(str(ti))
       
    
    def on_click(self):
        pass
        
        """
        def on_click(self,tab_name,times):
        cost_time=0
        accuracy=0
        
        for i in range(0,times):
            if tab_name='CNN':
                tmp1,tmp2=CNN_MNIST()
                accuracy=(accuracy+tmp1)/(i+1)
                cost_time=(cost_time+tmp2)/(i+1)    
            elif tab_name='SVM':
                tmp1,tmp2=SVM_MNIST()
                accuracy=(accuracy+tmp1)/(i+1)
                cost_time=(cost_time+tmp2)/(i+1)   
            elif tab_name='KNN':
                tmp1,tmp2=KNN_MNIST()
                accuracy=(accuracy+tmp1)/(i+1)
                cost_time=(cost_time+tmp2)/(i+1)   
            elif tab_name='NB':
                tmp1,tmp2=NB_MNIST()
                accuracy=(accuracy+tmp1)/(i+1)
                cost_time=(cost_time+tmp2)/(i+1)   
            else tab_name='DT':
                tmp1,tmp2=DT_MNIST()
                accuracy=(accuracy+tmp1)/(i+1)
                cost_time=(cost_time+tmp2)/(i+1)   
    
        """
    
    #想法是，在点击按扭之后将LineEdit的文本以数字的形式传出来
    #但是有一个想法是如何判断是哪一个tab里面的东西，事实上，想要多个同时还要涉及一些多线程的知识
    #假设传入指定文本框的数字，然后写一个for循环做累加就可以了
    
        
    
        

    def tab1UI(self):
        #全局布局
        whole_layout=QVBoxLayout()
        
        ###局部布局
        #表单布局
        form_layout=QFormLayout()
        #水平布局
        h_layout=QHBoxLayout()
        
        #表单布局内容
        self.CNN_le=QLineEdit();
        self.CNN_ac_label=QLabel();
        self.CNN_ti_label=QLabel();
        form_layout.addRow('请输入实验次数：',self.CNN_le)
        form_layout.addRow('平均准确率：',self.CNN_ac_label)
        form_layout.addRow('平均花费时间：',self.CNN_ti_label)
        
        #水平布局内容
        self.CNN_btn=QPushButton(self)
        self.CNN_btn.setText("开始训练")
        self.CNN_btn.clicked.connect(self.compute1)
        h_layout.addWidget(self.CNN_btn)
        
        #准备部件装小布局
        form=QWidget()
        h=QWidget()
        
        form.setLayout(form_layout)
        h.setLayout(h_layout)
        
        #设置整体布局
        whole_layout.addWidget(form)
        whole_layout.addWidget(h)
        
        
        
        #设置选项卡的小标题与布局方式
        self.setTabText(0,'CNN')
        self.tab1.setLayout(whole_layout)

    def tab2UI(self):
        #全局布局
        whole_layout=QVBoxLayout()
        
        ###局部布局
        #表单布局
        form_layout=QFormLayout()
        #水平布局
        h_layout=QHBoxLayout()
        
        #表单布局内容
        self.SVM_le=QLineEdit();
        self.SVM_ac_label=QLabel();
        self.SVM_ti_label=QLabel();
        form_layout.addRow('请输入实验次数：',self.SVM_le)
        form_layout.addRow('平均准确率：',self.SVM_ac_label)
        form_layout.addRow('平均花费时间：',self.SVM_ti_label)
        
        #水平布局内容
        self.SVM_btn=QPushButton(self)
        self.SVM_btn.setText("开始训练")
        self.SVM_btn.clicked.connect(self.compute2)
        h_layout.addWidget(self.SVM_btn)
        
        #准备部件装小布局
        form=QWidget()
        h=QWidget()
        
        form.setLayout(form_layout)
        h.setLayout(h_layout)
        
        #设置整体布局
        whole_layout.addWidget(form)
        whole_layout.addWidget(h)
        

        #设置标题与布局
        self.setTabText(1,'SVM')
        self.tab2.setLayout(whole_layout)

    def tab3UI(self):
        #全局布局
        whole_layout=QVBoxLayout()
        
        ###局部布局
        #表单布局
        form_layout=QFormLayout()
        #水平布局
        h_layout=QHBoxLayout()
        
        #表单布局内容
        self.KNN_le=QLineEdit();
        self.KNN_ac_label=QLabel();
        self.KNN_ti_label=QLabel();
        form_layout.addRow('请输入实验次数：',self.KNN_le)
        form_layout.addRow('平均准确率：',self.KNN_ac_label)
        form_layout.addRow('平均花费时间：',self.KNN_ti_label)
        
        #水平布局内容
        self.KNN_btn=QPushButton(self)
        self.KNN_btn.setText("开始训练")
        self.KNN_btn.clicked.connect(self.compute3)
        h_layout.addWidget(self.KNN_btn)
        
        #准备部件装小布局
        form=QWidget()
        h=QWidget()
        
        form.setLayout(form_layout)
        h.setLayout(h_layout)
        
        #设置整体布局
        whole_layout.addWidget(form)
        whole_layout.addWidget(h)
        

        #设置小标题与布局方式
        self.setTabText(2,'KNN')
        self.tab3.setLayout(whole_layout)
        
    def tab4UI(self):
        #全局布局
        whole_layout=QVBoxLayout()
        
        ###局部布局
        #表单布局
        form_layout=QFormLayout()
        #水平布局
        h_layout=QHBoxLayout()
        
        #表单布局内容
        self.NB_le=QLineEdit();
        self.NB_ac_label=QLabel();
        self.NB_ti_label=QLabel();
        form_layout.addRow('请输入实验次数：',self.NB_le)
        form_layout.addRow('平均准确率：',self.NB_ac_label)
        form_layout.addRow('平均花费时间：',self.NB_ti_label)
        
        #水平布局内容
        self.NB_btn=QPushButton(self)
        self.NB_btn.setText("开始训练")
        self.NB_btn.clicked.connect(self.compute4)
        h_layout.addWidget(self.NB_btn)
        
        #准备部件装小布局
        form=QWidget()
        h=QWidget()
        
        form.setLayout(form_layout)
        h.setLayout(h_layout)
        
        #设置整体布局
        whole_layout.addWidget(form)
        whole_layout.addWidget(h)
        
        #设置选项卡的小标题与布局方式
        self.setTabText(3,'NB')
        self.tab4.setLayout(whole_layout)
        
    def tab5UI(self):
        #全局布局
        whole_layout=QVBoxLayout()
        
        ###局部布局
        #表单布局
        form_layout=QFormLayout()
        #水平布局
        h_layout=QHBoxLayout()
        
        #表单布局内容
        self.DT_le=QLineEdit();
        self.DT_ac_label=QLabel();
        self.DT_ti_label=QLabel();
        form_layout.addRow('请输入实验次数：',self.DT_le)
        form_layout.addRow('平均准确率：',self.DT_ac_label)
        form_layout.addRow('平均花费时间：',self.DT_ti_label)
        
        #水平布局内容
        self.DT_btn=QPushButton(self)
        self.DT_btn.setText("开始训练")
        self.DT_btn.clicked.connect(self.compute5)
        h_layout.addWidget(self.DT_btn)
        
        #准备部件装小布局
        form=QWidget()
        h=QWidget()
        
        form.setLayout(form_layout)
        h.setLayout(h_layout)
        
        #设置整体布局
        whole_layout.addWidget(form)
        whole_layout.addWidget(h)
        
        #设置选项卡的小标题与布局方式
        self.setTabText(4,'DT')
        self.tab5.setLayout(whole_layout)
    
    
    
    
    
if __name__ == '__main__':
    app=QApplication(sys.argv)
    demo=Lab_UI()
    demo.show()
    sys.exit(app.exec_())

