import sys, os 
sys.path.append(os.pardir)
import numpy as np 
from dataset.mnist import load_mnist 
from PIL import Image 
from Knn import KNN

(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=True, normalize=False) 

# image = x_train[0]
# #label = t_train[0] 
# # 첫번째 데이터
# #print(label) 
# #print(image.shape)

# def img_show(img): 
#  pil_img = Image.fromarray(np.uint8(img)) 
#  pil_img.show() 
# # image를 unsigned int로

# image = image.reshape(28,28) 
# # 1차원 —> 2차원 (28x28) 
# print(image) 
# img_show(image)
input_list=[]
for idx, val in enumerate(x_train,t_train):
    input_list.append(val.reshape(28,28))
#KNN.get_distance(input_list)