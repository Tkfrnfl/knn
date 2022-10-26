from cProfile import label
from copyreg import pickle
import sys ,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def get_data():
    (x_train,t_train),(x_test,t_test)=\
    load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb')as f:
        network=pickle.load(f)
    return network    