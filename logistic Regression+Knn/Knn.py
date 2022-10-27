from math import sqrt
import math

import numpy as np
from PIL import Image 

class KNN():
    K=7
    #int_list28=[]
    dist_list=[]
    test_num=0

    def get_distance(input_list,label_list):
        KNN.test_num=len(input_list)%100+1 #100번째 데이터를 테스트로 사용하므로

        np_input=np.array(input_list)
        np_label=np.array(label_list)
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
        sort_dist_list=np.split(np_dist,KNN.test_num)
        
        np.sort(sort_dist_list,axis=1)
        print(sort_dist_list[0])

 