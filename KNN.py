from math import sqrt

class KNN():
    K=3
    def get_distance(X,y):
        for idx,val in enumerate(X):
            if idx+1%15 !=15:   #15번째 데이터가 아니면
                

