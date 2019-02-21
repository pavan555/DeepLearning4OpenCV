from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import argparse

def sigmoid_activation(x):
    
    return 1.0/( 1 + np.exp(-x))

def pred(X,W):
    
    preds=sigmoid_activation(X.dot(W))
    preds[preds<=0.5]=0
    preds[preds>0]=1
    
    return preds

def nextBatch(X,y,BatchSize):
    for i in np.arange(0,X.shape[0],BatchSize):
        yield (X[i:i+BatchSize],y[i:i+BatchSize])


(X,y)=make_blobs(n_samples=1000,n_features=2,centers=2,cluster_std=0.5,random_state=1)

ap=argparse.ArgumentParser()
ap.add_argument("-e","--epochs",help="#No.of iterations",default=100,type=int)
ap.add_argument("-a","--alpha",help="#learning rate",default=0.01,type=float)
ap.add_argument("-bs","--BatchSize",help="#No.of training samples per batch(Siz e of SGD mini batch)",type=int,default=32)
args=vars(ap.parse_args())

y=y.reshape((y.shape[0],1))
#trick to adding a column
X=np.c_[X,np.ones(X.shape[0])]


(trainX,testX,trainY,testY)=train_test_split(X,y,train_size=0.5,random_state=42)

print("[INFO......] Training..")
W=np.random.randn(X.shape[1],1)
losses=[]
#training the model using mini batch SGD
for epoch in np.arange(0,args["epochs"]):
    
    epochLoss=[]
    #dividing the given training data into mini batches
    for (BatchX,BatchY) in nextBatch(trainX,trainY,args["BatchSize"]):
        
        preds=sigmoid_activation(BatchX.dot(W))
        #calculating error
        error=preds-BatchY
        #squared error
        epochLoss.append(np.sum(error ** 2))
        #applying gradient to bia
        gradient=BatchX.T.dot(error)
        W+=-args["alpha"]*gradient
    
    #adding mean squared error
    loss=np.average(epochLoss)
    losses.append(loss)
    
    if epoch==0 or (epoch+1)%5==0:
        print("[INFO] epochs={},loss={:.7f}".format(int(epoch)+1,loss))
        
print("[INFO...] Testing..")
test=pred(testX,W)
print(classification_report(testY,test))
#showing the (testing) classified data
plt.style.use("ggplot")
plt.figure()
plt.title("given DATA")
plt.scatter(trainX[:,0],testX[:,1],marker='o',c=testY[:,0],s=30)

#showing loss over time
plt.style.use("ggplot")
plt.figure()
plt.title("Training loss")
plt.plot((np.arange(0,args["epochs"])),losses)
plt.xlabel("#Epochs")
plt.ylabel("loss")
plt.show()