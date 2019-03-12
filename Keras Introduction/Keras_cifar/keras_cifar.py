from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
#from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import cifar10
from keras.optimizers import SGD

##import matplotlib.pyplot as plt # for any machine we can use like this
##for virtual environments
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",help="#Path for Output Graph",required=True)
args=vars(ap.parse_args())

#Loading Datasets From Keras loads from ~.keras/datasets/...

print("[INFO..]***** Loading Dataset *****")
((trainX,trainY),(testX,testY))=cifar10.load_data()
trainX=trainX.astype("float")/255.0 

#converting 8-bit unsigned integers to floating and converting into in th erange of[0,1]

testX=testX.astype("float") / 255.0


trainX=trainX.reshape((trainX.shape[0],3072))# converting the given image in 50000 imges in 32*32*3 shape(50000,32,32,3)==>(50000,3072) which is 32*32*3=3072

testX=testX.reshape((testX.shape[0],3072))
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)
names=["aeroplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#creating a 3072-1024-512-10 feedforward neural network fully connected
model=Sequential()
model.add(Dense(1024,input_shape=(3072,),activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation="sigmoid"))

print("[INFO..]**** Training Network ****")

gradient=SGD(0.01)#alpha=0.01(Learning rate)
model.compile(loss="categorical_crossentropy",optimizer=gradient,metrics=["accuracy"])

M=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,batch_size=32)
print("[INFO....]***** Evaluating Network *****")

preds=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),preds.argmax(axis=1),target_names=names))


plt.style.use("ggplot")
plt.figure()
plt.title("Keras CIFAR10 training Loss and Accuracy")
plt.ylabel("Loss/Accuracy")
plt.xlabel("#Epochs")

plt.plot(np.arange(0,100),M.history["loss"],label="training Loss")
plt.plot(np.arange(0,100),M.history["val_loss"],label="testing Loss")
plt.plot(np.arange(0,100),M.history["acc"],label="Training Accuracy")
plt.plot(np.arange(0,100),M.history["val_acc"],label="Testing Accuracy")

plt.legend()
plt.show()
plt.savefig(args["output"])






















