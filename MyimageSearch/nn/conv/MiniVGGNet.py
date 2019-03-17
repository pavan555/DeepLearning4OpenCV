from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    
    def build(width,height,depth,classes):
        '''
        Model: Takes all convolutional filters as 3*3
        
        the model is (CONV{32 filters}==>RELU==>BN)*2==>POOL==>DO(25%)
        ==>(CONV{64 FILTERS}==>RELU==>BN)*2==>POOL==>DO(25%)
        ==>FC==>RELU==>BN==>DO(50%)==>FC==>SOFTMAX
        
        '''
        inputShape=(width,height,depth)
        chanDim=-1#means channels last
        if K.image_data_format()=="channels_first":
            inputShape=(depth,width,height)
            chanDim=1#means Channels First
        model=Sequential()
        for _ in range(2):
            model.add(Conv2D(32,(3,3),input_shape=inputShape,padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        for _ in range(2):
            model.add(Conv2D(64,(3,3),padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model