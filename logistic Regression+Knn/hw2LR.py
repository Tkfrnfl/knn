import sys, os 
sys.path.append(os.pardir)
import numpy as np 
from dataset.mnist import load_mnist 
from PIL import Image 
from LR import LR


(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=True, normalize=True) 

train_list=[]
test_list=[]
for idx, val in enumerate(x_train):   #학습 리스트생성
    if idx<60000:
       #val=val.reshape(28,28)
        train_list.append(val)

for idx, val in enumerate(x_test):  #테스트 리스트 생성
    if idx<100:
        #val=val.reshape(28,28)
        test_list.append(val)

LR.learn(train_list,test_list,t_train,t_test)
#LR.predict(test_list,t_test)