# Convolutional Neural Network


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout



# intialize you cNN as a squence of layres not as a graph
classifier=Sequential()

                              # convolution step 
                             
# define how many features maps and the (row,col) of the feature map
# first argument is the number of features maps 
# second argument is (row,col) of each feature map
# third a rgument you need to force your image to a certain (row,col) cause your 
# images are not all in the same format (row,col) differs so you need to enforce them to equal one
# remeber input image is converted into 2d array if it's black and white 
# and converted into 3d array if it's colored and we are working with colored image 
# a 3d array is composed of 3 channels and every channel coresponds to one 2d array
# now the third argument input_shape takes three arguments the third one 
# is the number of chaneels 1 if it's black and White and 2 if it's colored image 
#  input_shape = (64, 64, 3) first (64,64) is the the dim of the image and 3 is the number of channels
# if you choosed 128*128 you will take much time cause you are working in the cpu
classifier.add(Conv2D(32,(3,3), input_shape = (128,128,3),activation='relu'))
classifier.add(Dropout(rate = 0.1))
                            # pooling step
classifier.add(MaxPooling2D(pool_size=(2,2)))


                            # adding another layer and pooling it
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(Dropout(rate = 0.1))
classifier.add(MaxPooling2D(pool_size=(2,2)))
                          
                       # adding another layer and pooling it
classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(Dropout(rate = 0.1))
classifier.add(MaxPooling2D(pool_size=(2,2)))


                           
                            # flattening stpe
classifier.add(Flatten())
                           # full connection (our Ann)
 # common to choose units a power of two                          
                               # first full connection layer 
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dropout(rate = 0.1))
                               # second full connection layer 
                               
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dropout(rate = 0.1))

                           # output layer 

classifier.add(Dense(units=1,activation='sigmoid'))

# compiling 
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])    


# using keras to fit the cNN in this step we 
# encrich our dataset by rotating our images scalling them shearing and zooming
# so we get a lot more data than the original data 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255, # scale our values between 0 and 1 (originally they are 1 up to 255)
        shear_range=0.2, # random tranvictions (transitions)
        zoom_range=0.2, # applyinig some random zooms
        horizontal_flip=True) # flipping our image

test_datagen = ImageDataGenerator(rescale=1./255) # scale our test set

training_set= train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128,128), # the expected size of our input images look at the input layer
        batch_size=32,       # the number of images that will go after updating the weights
        class_mode='binary') # two or more objects here only binary outcome cat or dog

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')


# saving our model aftre every epoch
import os 
from keras.callbacks import ModelCheckpoint

check_point_path='saved_model/cp.ckpt'
check_point_dir=os.path.dirname(check_point_path)
cp_call_back=ModelCheckpoint(check_point_path,save_weights_only=True,verbose=1)


classifier.fit_generator( 
        training_set,
        steps_per_epoch=8000, # the number of images in our training set
        epochs=25,          # how many time to train our training set 
        validation_data=test_set, # evaluate preformance on this data
        validation_steps=2000
        ,callbacks=[cp_call_back]
        )      # number of images on our test_set  


# make a new prediction 



import numpy as np
from keras.preprocessing import image                                        # same as training 
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
