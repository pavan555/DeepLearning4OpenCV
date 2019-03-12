from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

import argparse
import numpy as np
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument("-o","--output-path",help="Path to Output image graph")
args=vars(ap.parse_args())

print("[INFO..]****Loading FULL MNIST dataset****")
dataset=datasets.fetch_mldata("MNIST Original")
data=dataset.data.astype("float") / 255.0
(trainX,testX,trainY,testY)=train_test_split(data,dataset.target,test_size=0.25)

lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.fit_transform(testY)

model=Sequential()
#model==> 784-256-128-64-10
model.add(Dense(256,input_shape=(784,),activation="sigmoid"))
model.add(Dense(128,activation="sigmoid"))
#model.add(Dense(64,activation="sigmoid"))
model.add(Dense(10,activation="softmax"))

print("[INFO..]**** Training Network ****")
gradient=SGD(0.01)#Learning rate alpha=0.01
model.compile(loss="categorical_crossentropy",optimizer=gradient,metrics=["accuracy"])

S=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,batch_size=128)

print("[Info..]**** EValuating Network ****")


preds=model.predict(testX,batch_size=128)
print(classification_report(testY.argmax(axis=1),preds.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100),S.history["loss"],label="Train_Loss")
plt.plot(np.arange(0,100),S.history["val_loss"],label="Validation_loss")
plt.plot(np.arange(0,100),S.history["acc"],label="Train_accuracy")
plt.plot(np.arange(0,100),S.history["val_acc"],label="Validation_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("#Epochs..")
plt.ylabel(" Loss/Accuracy ")
plt.legend()
plt.show()
#plt.savefig(args["OUTPUT-PATH"])
