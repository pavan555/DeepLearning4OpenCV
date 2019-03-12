from LP.nn.backprop import NeuralNetwork
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets

print("[INFO]==>Loading MNIST dataset")
digits=datasets.load_digits()
data=digits.data.astype("float")

data=(data-data.min())/(data.max()-data.min())

print(data)
#min/max normalizing by scaling each digit into the range [0, 1]

print("[info]==>Samples:{},dim:{}".format(data.shape[0],data.shape[1]) )
(trainX,testX,trainY,testY)=train_test_split(data,digits.target,test_size=0.25)

trainY=LabelBinarizer().fit_transform(trainY)
testY=LabelBinarizer().fit_transform(testY)

print("[INFO].. Training.........")
nn=NeuralNetwork([trainX.shape[1],32,16,10])

print("[INFo]==>Architecture:{}".format(nn))

nn.fit(trainX,trainY,epochs=1000)

print("[INFO]....EValuating.........")
preds=nn.predict(testX)
preds=preds.argmax(axis=1)
print(classification_report(testY.argmax(axis=1),preds))

