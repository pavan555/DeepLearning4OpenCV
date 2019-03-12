from LP.nn.backprop import NeuralNetwork
import numpy as np


X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[0]])

nn=NeuralNetwork([2,2,1],alpha=0.5)
nn.fit(X,Y,epochs=20000)

for (x,y) in zip(X,Y):
    pred=nn.predict(x)[0][0]
    step=1 if pred>0.5 else 0
    print("[INFO..] data:{}, Truth:{}, pred:{:.4f}, step:{}".format(x,y[0],pred,step))
    