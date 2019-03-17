
import numpy as np

class Perceptron(object):
    
    def __init__(self,N,alpha=0.1):#N:the number of columns in our feature vector
        
        self.W=np.random.randn(N+1)/np.sqrt(N)
        self.alpha=alpha
        
    def step(self,x):#Activation Function: Step Function
        return 1 if x>0 else 0
    
    def fit(self,X,Y,epochs=10):
        #epochs:No.of Iterations
        X=np.c_[X,np.ones((X.shape[0]))]
        
        for epoch in np.arange(0,epochs):
            
            for(x,y) in zip(X,Y):
                
                p=self.step(np.dot(x,self.W))
                
                if p!=y:
                    error=p-y
                    
                    self.W+=-self.alpha * error * x
                    
    def predict(self,X,addBias=True):
        
        #checking whether given array is a matrix or not
        X=np.atleast_2d(X)
        if addBias:
            X=np.c_[X,np.ones((X.shape[0]))]
        return self.step(np.dot(X,self.W))

                
            
    