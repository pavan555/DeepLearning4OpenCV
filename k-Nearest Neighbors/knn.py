

import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from LP.datasets.Simpledatasetloader import Simpledatasetloader
from LP.preprocessor.Simplepreprocessor import Simplepreprocessor


#argument Parsing

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to datasets")
ap.add_argument("-k","--neighbors",type=int,default=1,help="# of nearest neighbors for classification")
ap.add_argument("-j","--jobs",type=int,default=-1,help="# of jobs used for k-NN distance (-1 uses all available cores)")

args=vars(ap.parse_args())


#loading and preprocessing

print("[INFO..] loading images")
imagepaths=list(paths.list_images(args["dataset"]))

sp=Simplepreprocessor(32,32)
sd=Simpledatasetloader(preprocessors=[sp])

(data,label)=sd.load(imagepaths,verbose=500)
data=data.reshape((data.shape[0],3072))

print("[INFO..] features matrix: {:.1f}MB".format(data.nbytes/(1024*1000.0)))

le=LabelEncoder()
labels=le.fit_transform(label)

#splitting into training and testing datasets
(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25,random_state=42)

print("[INFO..] evaluating k-NN classifier ")
model=KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX,trainY)

print(classification_report(testY,model.predict(testX),target_names=le.classes_))