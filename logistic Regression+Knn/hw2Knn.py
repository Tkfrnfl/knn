import sys, os 
sys.path.append(os.pardir)
import numpy as np 
from dataset.mnist import load_mnist 
from PIL import Image 
from Knn import KNN
import time
import math

(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=True, normalize=True) 

# image = x_train[1]
# #label = t_train[0] 
# # 첫번째 데이터
# #print(label) 
# #print(image.shape)

def img_show(img): 
 pil_img = Image.fromarray(np.uint8(img)) 
 pil_img.show() 
# # image를 unsigned int로

#img_show(test)
# image = image.reshape(28,28) 
# # 1차원 —> 2차원 (28x28) 
# print(image) 
# img_show(image)

train_list=[]
test_list=[]
for idx, val in enumerate(x_train):   #학습 리스트생성
    if idx<10000:
        val=val.reshape(28,28)
        train_list.append(val)

for idx, val in enumerate(x_test):  #테스트 리스트 생성
    if idx<100:
        val=val.reshape(28,28)
        test_list.append(val)

start =time.time()  
math.factorial(100000)
KNN.get_distance(train_list,t_train,test_list,t_test)
end_dist = time.time()
KNN.get_neighbors()
end_neighbors = time.time()
print("Fit time: "+ f"{end_neighbors - start:.10f}")
# KNN.get_neighbors_weight()
# end_neighbors_weight = time.time()
# print("Fit time: "+ f"{(end_dist-start)+(end_neighbors_weight-end_neighbors):.10f}")