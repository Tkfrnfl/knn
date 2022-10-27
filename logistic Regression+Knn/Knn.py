from math import sqrt
import math

import numpy as np
from PIL import Image 

class KNN():
    K=7
    #int_list28=[]

    def get_distance(input_list,label_list):

        for idx, val in enumerate(input_list):
            if(idx+1)%100==0: #10000개의 데이터중 100번째 데이터인 경우 테스트 데이터로 사용
               
                for idx_10000,val_10000 in (input_list):
                    if (idx+1)%100!=0:
                        for idx_28,val_28 in enumerate(val):
                            

                     

            # if idx==0:
            #     print(KNN.int_list28[0])
            #     KNN.img_show(KNN.int_list28[0])

    def img_show(img): 
        pil_img = Image.fromarray(np.uint8(img)) 
        pil_img.show() 