import matplotlib.pyplot as plt
import numpy as np

x=np.array([1,1])
w=np.array([0.5,0.5])
b=-0.7
w*x

def Nand():
    tmp =np.sum(w*x)+b

    if tmp<0:
        return 1 
    else:
        return 0

def And():
    tmp =np.sum(w*x)+b

    if tmp>0:
        return 1 
    else:
        return 0
def xor():

    tmp= And()*Nand()
    return tmp    
def Or():
    tmp =np.sum(w*x)+b
    if tmp>0:
        return 1
    else:
        return 0    

print(Nand())