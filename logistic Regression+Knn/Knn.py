from math import sqrt
import math

import numpy as np
from PIL import Image 

class KNN():
    K=15
    dist_list=[]
    test_num=0
    global_input_list=[]
    global_label_list=[]

    def get_distance(train_list,train_label,test_list,test_label):
        # np_train=np.array(train_list)
        # np_test=np.array(test_list)

        KNN.test_num=len(test_list)
        KNN.global_label_list=test_label

        for idx_test,val_test in enumerate(test_list):
            for idx_train,val_train in enumerate(train_list):
                dist=0
                for i in range(0,28):
                    for j in range(0,28):
                        dist+=math.pow(val_test[i][j]-val_train[i][j],2)

                KNN.dist_list.append([sqrt(dist),train_label[idx_train]])


    def get_neighbors():
        correct_count=0
        np_dist=np.array(KNN.dist_list)
        sort_dist_list=np.split(np_dist,KNN.test_num)  #앞에서부터 순서대로 1,2,3..번째 테스트 데이터와의 거리계산값

        
        for i in range(KNN.test_num):    
            sort_dist_list[i]=sort_dist_list[i][sort_dist_list[i][:,0].argsort()]
        
        no_wieght_list=[[0]*11 for _ in  range(100)]        # 이웃들의 클래스 저장해주는 리스트
        wieght_list=[[0]*11 for _ in  range(100)]        # 이웃들의 가중치 클래스 저장해주는 리스트

        for idx,val in enumerate(sort_dist_list):
            for i in range(0,KNN.K):
                no_wieght_list[idx][int(val[i][1])]+=1

        for idx,val in enumerate(sort_dist_list):
            for i in range(0,KNN.K):
                wieght_list[idx][int(val[i][1])]+=(1/val[i][0])

        for i in range(KNN.test_num):                       #결과 출력
            print(str(no_wieght_list[i].index(max(no_wieght_list[i])))+' '\
                +str(KNN.global_label_list[i]) )
            if no_wieght_list[i].index(max(no_wieght_list[i]))==KNN.global_label_list[i]:
                correct_count+=1
        print('정확도'+str(correct_count/KNN.test_num))    

        print('--------next>>weighted--------------')
        correct_count=0
        for i in range(KNN.test_num):                       #결과 출력 (가중치)
            print(str(wieght_list[i].index(max(wieght_list[i])))+' '\
                +str(KNN.global_label_list[i]) )
            if wieght_list[i].index(max(wieght_list[i]))==KNN.global_label_list[i]:
                correct_count+=1
        print('정확도'+str(correct_count/KNN.test_num))  
    # def get_neighbors_weight():
    #     print()

 