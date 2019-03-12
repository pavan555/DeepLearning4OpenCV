import numpy as np
from LP.nn.perceptron import Perceptron
from time import sleep


def Train_Perceptron(X,Y):
    print("[INFO...]training Perceptron...")
    p=Perceptron(X.shape[1],alpha=0.1)
    p.fit(X,Y,epochs=21)
    sleep(0.5)
    Test_Perceptron(X,Y,p)


def Test_Perceptron(X,Y,p):
    print("[INFO]>>> Testing Perceptron...")
    for (x,y) in zip(X,Y):
        pred=p.predict(x)
        print("data:{}  ground Truth:{}  predicted:{}".format(x,y[0],pred))


        

'''first AND is applied then trained and tested whether it is correctly classifying the data and then OR and then XOR also... '''
#fit a Perceptron model to the bitwise AND dataset(Linearly Separable)
# 0 AND 0=0,1 AND 1=1
print("[INFO..]Bitwise AND")

Xa=np.array([[0,0],[0,1],[1,0],[1,1]])
Ya=np.array([[0],[0],[0],[1]])
Train_Perceptron(Xa,Ya)
#Test_Perceptron(Xa,Ya)

#to show difference we are taking some time break
sleep(5)

print("[INFO..]Bitwise OR")
#fit a Perceptron model to the bitwise OR dataset(Linearly Seaparable)
# 0 or 0=0,1 or 1=1,0 or 1=1 , 1 or 0=1

Xo=np.array([[0,0],[0,1],[1,0],[1,1]])
Yo=np.array([[0],[1],[1],[1]])

Train_Perceptron(Xo,Yo)
#Test_Perceptron(Xo,Yo)

#to show difference we are taking some time break
sleep(5)


#fit a Perceptron model to the bitwise XOR dataset(NON-Linearly Separable)
# 0 xor 0=0,1 xor 1=0, 0 xor 1 =1, 1 xor 0 =1

print("[INFO..]Bitwise XOR")
Xx=np.array([[0,0],[0,1],[1,0],[1,1]])
Yx=np.array([[0],[1],[1],[0]])

Train_Perceptron(Xx,Yx)
#sTest_Perceptron(Xx,Yx)

