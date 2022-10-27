from math import sqrt
import math

import numpy as np
from PIL import Image 

class KNN():
    K=15
    #int_list28=[]
    dist_list=[]
    test_num=11
    global_input_list=[]
    global_label_list=[]

    def get_distance(input_list,label_list):
        KNN.test_num=len(input_list)//100 #100번째 데이터를 테스트로 사용하므로

        np_input=np.array(input_list)
        np_label=np.array(label_list)
        KNN.global_label_list=np_label

        # print(np_input[0])
        for idx, val in enumerate(np_input):
            if(idx+1)%100==0: #10000개의 데이터중 100번째 데이터인 경우 테스트 데이터로 사용

                for idx_10000,val_10000 in enumerate(np_input):
                    
                    if (idx_10000+1)%100!=0:    
                        tmp_list=np.power(np_input[idx_10000]-np_input[idx],2)

                        # for i in range(0,28):
                        #     for j in range(0,28):
                        #       dist+=math.pow((val[i][j]-val_10000[i][j]),2)
                    KNN.dist_list.append([sqrt(np.sum(tmp_list)),label_list[idx_10000]])   #거리,라벨 저장

        #print(KNN.dist_list)
    def get_neighbors():
        np_dist=np.array(KNN.dist_list)
        sort_dist_list=np.split(np_dist,KNN.test_num)#앞에서부터 순서대로 0, 100...200 번째의 거리값
        
        sort_dist_list=np.sort(sort_dist_list,axis=1)
        # for i in range(len(sort_dist_list)):
        #     np.sort(sort_dist_list[i],axis=1)
        print(sort_dist_list[0])
        no_wieght_list=[[0]*11 for _ in  range(100)]        # 이웃들의 클래스 저장해주는 리스트
        
        #print(sort_dist_list)
        for idx,val in enumerate(sort_dist_list):
            for i in range(0,KNN.K):
                no_wieght_list[idx][int(val[i][1])]+=1


        for i in range(KNN.test_num):
            print(str(no_wieght_list[i].index(max(no_wieght_list[i])))+' '\
                +str(KNN.global_label_list[i*100]) )

        # print((no_wieght_list[3]))
        # print(KNN.global_label_list[300])
        #print(sort_dist_list[0])


 