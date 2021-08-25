import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.imagenet_utils import preprocess_input

from keras.applications import (vgg16,  
                                vgg19, 
                                xception, 
                                inception_v3,  
                                inception_resnet_v2, 
                                mobilenet,
                                densenet, 
                                nasnet, 
                                mobilenet_v2)
vgg_model = vgg16.VGG16(weights='imagenet')
# vgg19_model = vgg19.VGG19(weights='imagenet')
# mobv2= mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights='imagenet')
# nasnetmobile = nasnet.NASNetMobile(weights="imagenet")
# largest_dense_net = densenet.DenseNet201(weights="imagenet")
# mobilenet_ = mobilenet.MobileNet(weights="imagenet")
# incepv2 = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# incepv3 = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# Xception_ = xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# large_nasnet = nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

# print(vgg_model.summary())

# For one of the models above, get the expected input shape for an image
image_width = eval(str(vgg_model.layers[0].output.shape[1]))
image_height = eval(str(vgg_model.layers[0].output.shape[2]))

# Load image as a PIL format image
pil_img = load_img('image.png',  target_size=(image_width, image_height))

# Turn the image into a (224, 224, 3) matrix, and then add an extra dimension to match the input dimension of the model
array_img = img_to_array(pil_img)
images = np.expand_dims(array_img, axis=0)

# Do this conversion for every image in the dataset
list_of_pics_array= []
for image in image_files:
    original = load_img(f, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image = np.expand_dims(numpy_image, axis=0)

    importedImages.append(image)
    
images = np.vstack(list_of_pics_array)
dense_mat = preprocess_input(images)


'''
References:
https://towardsdatascience.com/image-recommendation-engine-leverage-transfert-learning-ec9af32f5239
'''