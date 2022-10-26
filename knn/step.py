import matplotlib.pyplot as plt
import numpy as np

# def step_func(x):
#     return np.array(x>0,dtype=int)

# def sigmoid(x):
#     return np.array(1/(1+np.exp(-x)))

def relu(x):
    return np.maximum(x,0) #음수 , 양수 관계없이 큰거 고르므로
x=np.arange(-5.0,5.0,0.1)    
y=relu(x)
plt.plot(x,y)
plt.show()