import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    
    def __init__(self,layers,alpha=0.1):
        
        self.layers=layers
        self.W=[]
        self.alpha=alpha
        
        for i in np.arange(0,len(layers)-2):
            w=np.random.randn(layers[i]+1,layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[i]))
            
        w=np.random.randn(layers[-2]+1,layers[-1])#last one is output it doesn't contain bias term
        self.W.append(w/np.sqrt(layers[-2]))
        
    def __repr__(self):
        return "Neural Network:{}".format("-".join(str(l) for l in self.layers))
    
    
    def sigmoid(self,x):
        return 1.0/(1 + np.exp(-x))
    
    
    def sigmoid_deriv(self,x):
        
        return x * (1-x)
    
    
    def fit(self,X,Y,epochs=1000,displayUpdate=100):
        X=np.c_[X,np.ones((X.shape[0]))]
        losses=[]
        for epoch in np.arange(0,epochs):
            
            for (x,y) in zip(X,Y):
                self.fit_partial(x,y)
            if epoch==0 or (epoch+1)%displayUpdate==0:
                loss=self.calc_loss(X,Y)
                losses.append(loss)
                print("[INFO..]Epoch={},loss={:.7f}".format(epoch+1,loss))
        '''plt.style.use("ggplot")
        plt.figure()
        plt.title("given DATA")
        plt.scatter(trainX[:,0],testX[:,1],marker='o',c=testY[:,0],s=30)
        '''
        
        '''
        backprop_2.png
        [INFO..]Epoch=20000,loss=0.0000024
        [INFO..] data:[0 0], Truth:0, pred:0.0106, step:0
        [INFO..] data:[0 1], Truth:1, pred:0.9892, step:1
        [INFO..] data:[1 0], Truth:1, pred:0.9869, step:1
        [INFO..] data:[1 1], Truth:0, pred:0.0111, step:0
        '''

        #showing loss over time
        plt.style.use("ggplot")
        plt.figure()
        plt.title("Training loss")
        plt.plot((np.arange(0,epochs,displayUpdate)),losses[:-1])#200 values for 20000
        plt.xlabel("#Epochs")
        plt.ylabel("loss")
        plt.show()
    
    def fit_partial(self,x,y):
        A=[np.atleast_2d(x)]
        
        #3 phases
        #1.FEED FORWARD
        for layer in np.arange(0,len(self.W)):
            net=A[layer].dot(self.W[layer])
            output=self.sigmoid(net)
            A.append(output)
        #2.BackPropagation
        
        error=A[-1]-y
        D=[error*self.sigmoid_deriv(A[-1])]
        
        for layer in np.arange(len(A)-2,0,-1):
            delta=D[-1].dot(self.W[layer].T)
            delta=delta*self.sigmoid_deriv(A[layer])
            D.append(delta)
        D=D[::-1]
        #3.WEIGHT UPDATE PHASE
        
        for layer in np.arange(0,len(self.W)):
            self.W[layer]+=-self.alpha*A[layer].T.dot(D[layer])
            
    def predict(self,X,addBias=True):
        
        p=np.atleast_2d(X)
        if addBias:
            p=np.c_[p,np.ones(p.shape[0])]
        for layer in np.arange(0,len(self.W)):
            p=self.sigmoid(np.dot(p,self.W[layer]))
        return p
    
    def calc_loss(self,X,Y):
        Y=np.atleast_2d(Y)
        preds=self.predict(X,addBias=False)
        loss=(0.5)*(np.sum(preds-Y)**2)
        
        return loss
        
        