from dis import dis
from math import sqrt
import math
import re
import numpy as np


class KNN():
    K=7
    
    tmp_list=[]
    num_neighbor_list=[]
    ans_vote=[]
    ans_weight=[]
    true_class=[]
    def get_distance(X,y):
        
        for idTest,valTest in enumerate (X):
            if (idTest+1)%15==0 : # 15번째 데이터이면 거리 구하기
                dist_list=[]

                if y[idTest]==0: # true class 저장하기
                    KNN.true_class.append('Setosa')
                elif y[idTest]==1:
                    KNN.true_class.append('Versicolor' )
                else:
                    KNN.true_class.append('Virginica' )

                for idx,val in enumerate(X):              
                    if (idx+1)%15!=0:   #15번째 데이터가 아닐때만 거리 계산
                        dist= math.pow((val[0]-valTest[0]),2)+math.pow((val[1]-valTest[1]),2)\
                            +math.pow((val[2]-valTest[2]),2)+math.pow((val[3]-valTest[3]),2)  
                        dist_list.append([sqrt(dist),y[idx]])                 
                KNN.tmp_list.append(dist_list)   

    def get_neighbors():
        
        #print(KNN.tmp_list[1])
        for i in range(len(KNN.tmp_list)):
            KNN.num_neighbor_list=KNN.tmp_list[i]
            KNN.num_neighbor_list.sort(key=lambda x:x[0])
            KNN.ans_vote.append(KNN.get_predict_vote()) # 일반적인 vote 예측값 저장
            KNN.ans_weight.append(KNN.get_predict_weight())# 가중치 vote 예측값 저장

        KNN.get_output(KNN.ans_vote) # output 함수 실행
        print("--------weight-------")
        KNN.get_output(KNN.ans_weight)# output 함수 실행
        
    def get_predict_vote():  # 일반적인 vote
        min_list=[]
        vote_list=[0,0,0]


        for i in range(KNN.K):
            min_list.append(KNN.num_neighbor_list[i])
        for idx,val in enumerate(min_list):
            vote_list[val[1]]+=1

        max_num=vote_list.index(max(vote_list))    

        if max_num==0:
            return 'Setosa'
        elif max_num==1:
            return 'Versicolor'    
        else:
            return 'Virginica'

    def get_predict_weight():  # 가중치 vote
            min_list=[] 
            weight_list=[0,0,0]


            for i in range(KNN.K):
                min_list.append(KNN.num_neighbor_list[i])
            for idx,val in enumerate(min_list):
                weight_list[val[1]]+=(1/val[0])

            max_num=weight_list.index(max(weight_list))    

            if max_num==0:
                return 'Setosa'
            elif max_num==1:
                return 'Versicolor'    
            else:
                return 'Virginica'

    def get_output(result_list): # output
        for idx,val in enumerate(result_list):
            print("Test Data Index:"+str(idx)+" Computed class:"+val+", True class:"+KNN.true_class[idx])