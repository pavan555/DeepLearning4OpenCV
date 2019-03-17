from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    
    def __init__(self,dataFormat=None):
        #store the dataformat
        self.dataFormat=dataFormat
    
    def preprocess(self,image):
        #applying keras utility function that correctly rearranges the dimensions of image
        return img_to_array(image,data_format=self.dataFormat)
    