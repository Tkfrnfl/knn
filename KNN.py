from dis import dis
from math import sqrt
import math
import re
import numpy as np


class KNN():
    K=3
    
    tmp_list=[]
    num_neighbor_list=[]
    ans_vote=[]
    ans_weight=[]

    def get_distance(X,y):
        
        for idTest,valTest in enumerate (X):
            if (idTest+1)%15==0 : # 15번째 데이터이면
                dist_list=[]
                for idx,val in enumerate(X):              
                    if (idx+1)%15!=0:   #15번째 데이터가 아니면
                        dist= math.pow((val[0]-valTest[0]),2)+math.pow((val[1]-valTest[1]),2)\
                            +math.pow((val[2]-valTest[2]),2)+math.pow((val[3]-valTest[3]),2)  
                        dist_list.append([sqrt(dist),y[idx]])                 
                KNN.tmp_list.append(dist_list)   

    def get_neighbors():
        
        #print(KNN.tmp_list[1])
        for i in range(len(KNN.tmp_list)):
            KNN.num_neighbor_list=KNN.tmp_list[i]
            KNN.num_neighbor_list.sort(key=lambda x:x[0])
            KNN.ans_vote.append(KNN.get_predict_vote())

        print(KNN.ans_vote)
        # for i in range(len(KNN.tmp_list)):
        #     np.sort (num_neighbor_list[0],axis=1)
        #     #KNN.tmp_list[i].sort(key=lambda x:x[0])
        #     #print(num_neighbor_list[0][0])
        #     for j in range(KNN.K):
        #         #print(j)
        #         KNN.neighbors_list[i].append(KNN.tmp_list[i][j])

        #print (KNN.neighbors_list[1])    


    def get_predict_vote():
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

        #print(vote_list[0])    
