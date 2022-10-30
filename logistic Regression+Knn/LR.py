from cmath import nan
from math import sqrt
import math
import numpy as np
from PIL import Image 
from random import random
import matplotlib.pyplot as plt

class LR():
    learning_rate = 0.00001
    epochs = 500

    cost_list=[[0]*1000 for _ in range(10)]
    correct=0
    sum=0

    w1=np.random.rand(784,10)    # 가중치와 편향 초기화
    b1=np.random.rand(10)

    w2=np.random.rand(10,10)
    b2=np.random.rand(10)

    def sigmoid(value):

        over_fl=[]
        for i in range(len(value)):
            if -value[i]>np.log(np.finfo(type(value[i])).max):
                over_fl.append(0.0)
            else:
                a=math.pow(math.e,-value[i])
                over_fl.append(1.0/(1.0+a))
        over_fl=np.array(over_fl)

        
        return over_fl


    def softmax(value):
        tmp_max=np.max(value)
        value-=tmp_max

        return np.exp(value)/np.sum(np.exp(value))    

    def cost(x,y) :
        m=len(y)
        delta = 1e-7    #log 오류 방지

        
        tmp_dot=np.dot(x,LR.w1)+LR.b1
        z1=LR.sigmoid(tmp_dot)
        z2=LR.sigmoid(np.dot(z1,LR.w2)+LR.b2)

        cost_v = (-1/m) * np.sum( y*np.log(z2+delta) + (1-y)*np.log((1-z2)+delta))   

        grad_w1 = (1/m) *np.sum(np.dot(np.dot(x,LR.w1)+LR.b1,z1-y))
        grad_b1=(1/m)*np.sum(z1-y)

        grad_w2 = (1/m) *np.sum(np.dot(np.dot(z1,LR.w2)+LR.b2,z2-y))
        grad_b2=(1/m)*np.sum(z2-y)

        return cost_v,grad_w1,grad_b1,grad_w2,grad_b2

    def learn(train_list,test_list,train_label,test_label):

        for k in range(1,10):
            LR.w1=np.random.rand(784,10)
            LR.b1=np.random.rand(10)

            LR.w2=np.random.rand(10,10)
            LR.b2=np.random.rand(10)

            for i in range(LR.epochs):

                tmp_y=np.max(train_label)+1   #test_list one hot encoding
                y=np.eye(tmp_y)[train_label]
                x=np.array(train_list)
         

                for j in range(0,10000):   #10000개의 데이터에 대하여 학습
                    if train_label[j]==k:
                        cost_v,grad_w1,grad_b1,grad_w2,grad_b2=LR.cost(x[j],y[j])
                        LR.w1-=LR.learning_rate*grad_w1
                        LR.b1-=LR.learning_rate*grad_b1
                        LR.w2-=LR.learning_rate*grad_w2
                        LR.b2-=LR.learning_rate*grad_b2
                        if j%200==0:
                            print("epochs = ", i, ", index = ", j, ", loss value = ", cost_v)
                LR.cost_list[k][i]=cost_v
            LR.predict(test_list,test_label,k)            #각 숫자별 epoch 단위 학습이 끝나면 예측 실행
                
        print("정확도:" + str(LR.correct/LR.sum))    
    def predict(test_list,test_label,num):
        print("nnnn"+str(num))
        # LR.correct=0
        # LR.sum=0
        for i in range(len(test_list)):     
            if test_label[i]==num:
                LR.sum+=1
                z1=LR.sigmoid(np.dot(test_list[i],LR.w1)+LR.b1)
                z2=LR.sigmoid(np.dot(z1,LR.w2)+LR.b2)
                temp_p=np.argmax(z2)
                # print(temp_p)
                # print(test_label[i])
                if temp_p==test_label[i]:
                    LR.correct+=1 
        print(LR.sum)            
        print("정확도:" + str(LR.correct/LR.sum))            
    
