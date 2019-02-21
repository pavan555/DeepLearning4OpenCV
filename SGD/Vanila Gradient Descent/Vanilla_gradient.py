from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import argparse

def sigmoid_activation(x):
    #compute the sigmoid activation value lies between 1 and 0 if >0.5 then ON <0.5 OFF
    return 1.0/(1 + np.exp(-x))

def predict(X,W):
    
    preds=sigmoid_activation(X.dot(W))
    
    preds[preds<=0.5]=0 #we are setting all preds less than or equal to 0.5 as 0 and 0.5 updated as 0 
    #so preds whose value=0.5 becomes 0 and updating values which are greater than 0 as 1
    
    preds[preds>0]=1
    
    return preds


ap=argparse.ArgumentParser()
ap.add_argument("-e","--epochs",type=float,default=100,help="# of epochs(No.of iterations)")
ap.add_argument("-a","--alpha",type=float,default=0.01,help="learning rate")

args=vars(ap.parse_args())


(X,y)=make_blobs(n_samples=1000,n_features=2,centers=2,cluster_std=1.5,random_state=1)

y=y.reshape((y.shape[0],1))
X=np.c_[X,np.ones((X.shape[0]))]

(trainX,testX,trainY,testY)=train_test_split(X,y,test_size=0.5,random_state=42)

print("[INFO....] training....")
W=np.random.randn(X.shape[1],1)
losses=[]

for e in np.arange(0,args["epochs"]):
    
    preds=sigmoid_activation(trainX.dot(W))
    ##preds=sigmoid_activation(W.dot(trainX))
    
    error=preds-trainY
    loss=np.sum(error ** 2)
    losses.append(loss)
    
    
    gradient=trainX.T.dot(error)
    W+= -(args["alpha"]*gradient)
    
    if e==0 or (e+1)%5==0:
        print("[INFO..] epoch={},loss={:.7f}".format(int(e+1),loss))
print("[INFO....] Evaluating")

preds=predict(testX,W)
print(classification_report(testY,preds))

#testing classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Blobs Data")
plt.scatter(testX[:,0],testX[:,1],marker='o',c=testY[:,0],s=30)
#testing loss data over time

plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0,args['epochs']),losses)
plt.title("Blobs Data Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()