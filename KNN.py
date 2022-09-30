from dis import dis
from math import sqrt
import math
import numpy as np


class KNN():
    K=3
    
    tmp_list=[]
    neighbors_list=[[]]*100


    def get_distance(X,y):
        dist_list=[[]]*100
        for idTest,valTest in enumerate (X):
            if (idTest+1)%15==0 : # 15번째 데이터이면
                for idx,val in enumerate(X):
                    if (idx+1)%15 !=0:   #15번째 데이터가 아니면
                        dist= math.pow((val[0]-valTest[0]),2)+math.pow((val[1]-valTest[1]),2)\
                            +math.pow((val[2]-valTest[2]),2)+math.pow((val[3]-valTest[3]),2)
                        dist_list[(idTest+1)//15-1].append([sqrt(dist),y[idx]])
        KNN.tmp_list=dist_list   

    def get_neighbors():
        #print(KNN.tmp_list[0][0])
        num_neighbor_list=np.array(KNN.tmp_list)
        test=KNN.tmp_list[0]
        test.sort(key=lambda x:x[0])
        print(test)

        for i in range(len(KNN.tmp_list)):
            np.sort (num_neighbor_list[0],axis=1)
            #KNN.tmp_list[i].sort(key=lambda x:x[0])
            #print(num_neighbor_list[0][0])
            for j in range(KNN.K):
                #print(j)
                KNN.neighbors_list[i].append(KNN.tmp_list[i][j])

        #print (KNN.neighbors_list[1])    


    def get_predict_vote():
        vote_list=[[0]]*100

        for idx,val in enumerate(KNN.neighbors_list):
            for id_sub, val_sub in enumerate(val):
                vote_list[idx][val[1]]+=1

        #print(vote_list[0])    
