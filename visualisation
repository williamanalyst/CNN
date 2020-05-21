# In[]: 
# Prepare a structure for the data to differ categories/ labels
# not needed in this case, since the images have already been manually classified with labels
from keras.models import Sequential # for initialzing the CNN (sequence of layers)
from keras.layers import Convolution2D # add the convolutional layers (2D package for images, 3D for videos)
from keras.layers import MaxPooling2D # add the pooling layers
from keras.layers import Flatten # convert the pooling layer into large feature vectors
from keras.layers import Dense # add the fully-connected layers into the CNN
from keras.layers import Dropout # 
from keras.callbacks import EarlyStopping
import os
os.chdir('D:\Collection_Dataset\wine_label_test1')
# 
# In[]: Initialising the CNN:
#
classifier = Sequential() # CNN is initialised
#
classifier.add(Convolution2D(32, (4, 4), # Convolution2D --> 32 feature dectors of 3*3 dimentions  
                             border_mode = 'same', input_shape = (64, 64, 3), activation = 'relu')) # input_shape --> colour, width, height
classifier.add(MaxPooling2D(pool_size = (2,2))) # reduce the size of convolutional layer by 50%, therefore reduce future input(nodes) by 50%
#
# Adding a 2-nd convolutional layers:
classifier.add(Convolution2D(32, (4, 4), activation = 'relu')) # no need to put in the 'input_shape'
classifier.add(MaxPooling2D(pool_size = (2,2))) # reduce the size of convolutional layer by 50%, therefore reduce future input(nodes) by 50%
#
# Adding a 3-rd convolutional layers:
classifier.add(Convolution2D(32, (4, 4), activation = 'relu')) # no need to put in the 'input_shape'
classifier.add(MaxPooling2D(pool_size = (2,2))) # reduce the size of convolutional layer by 50%, therefore reduce future input(nodes) by 50%
#
classifier.add(Flatten()) # 
#
classifier.add(Dense(output_dim = 16, activation = 'relu')) # 
classifier.add(Dropout(0.2))
#
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #  
classifier.add(Dropout(0.2))
#
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # 
# 
# Image Augmentation: reduce over-fitting by modifying the images (reverse, move, other transformation...), therefore reducing errors
from keras.preprocessing.image import ImageDataGenerator
# Prepare the image augmentation:
train_datagen = ImageDataGenerator( 
        rescale=1./255, # rescale all the pixel values to between 0 and 1 (as pixel values is originally defined between 0 and 255)
        shear_range=0.2, # how much you want to apply the random transformation (image augmentation)
        zoom_range=0.2,
        horizontal_flip=True) # 
#
test_datagen = ImageDataGenerator(rescale=1./255) # recale the pixel values
# Apply the image augmentation
batch_size_selected = 32 # batch size determines the number of samples in each mini-batch, which range between [1, number of samples]
steps_per_epoch_selected = 50 # the number of batch iterations before a training spoch is considered finished
epochs_selected = 50
#
training_set = train_datagen.flow_from_directory('training_set',
                                                target_size=(64, 64), #
                                                batch_size=batch_size_selected, # 
                                                class_mode='binary') # output classifications (binary/ 3+)
# this code section will create a test-set
test_set = test_datagen.flow_from_directory('test_set', 
                                            target_size=(64, 64),
                                            batch_size=batch_size_selected,
                                            class_mode='binary')
#
earlystop = EarlyStopping(monitor='val_loss', min_delta = 0.00001, patience=3, verbose= 0, mode='auto')
#
fitted_model = classifier.fit_generator(training_set,
                    steps_per_epoch = steps_per_epoch_selected,
                    epochs= epochs_selected ,
                    validation_data = test_set,
                    validation_steps = 20) # 
#
#classifier.fit
fitted_model.history # 
#
# In[]:
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(fitted_model.history['accuracy']) # 
plt.plot(fitted_model.history['val_accuracy'])
plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch number')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(fitted_model.history['loss']) #
plt.plot(fitted_model.history['val_loss'])
plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
##
import numpy as np
from keras.preprocessing import image
#
test_image = image.load_img('D:/Collection_Dataset/wine_label_test1/check/l204.jpg', 
                            target_size = (64, 64)) # resize image to 64*64 in size
#
test_image = image.img_to_array(test_image) # convert the image into 64*64*3 array
test_image = np.expand_dims(test_image, axis = 0) # 
#
result = classifier.predict(test_image) # predict method require 4 dimentions (input must be in a batch)
print(result)
#
training_set.class_indices
#
if result[0][0] == 1:
    prediction = 'low'
else:
    prediction = 'high'
print(prediction)
#
#def test_image(file_location):
#    for i in file_location:
#        test_image = image.load_img(i, target_size = (64, 64))
#        test_image = image.img_to_array(test_image)
#        test_image = np.expand_dims(test_image, axis = 0)
#        result = classifier.predict(test_image)
#        training_set.class_indices
#        if result[0] > 0.5:
#            prediction = 'high'
#        else:
#            prediction = 'low'
#        print(prediction)
#
#file_location = 'D:\Collection_Dataset\wine_label_test1\check\h206.jpg'
#test_image(file_location)
#file_location = ['C:\Python_Project\Deep_Learning\CNN\dataset\single_prediction\cat_or_dog_' + str(i) +'.png' for i in range(21, 29)]
#test_image(file_location)
##
#file_location2 = ['C:\Python_Project\Deep_Learning\CNN\dataset\single_prediction\cat_or_dog_' + str(i) +'.jpg' for i in range(11, 19)]
#test_image(file_location2)
#
# In[]: Plot the feature map, tested
from keras.models import Model
#
classifier.summary()
model = Model(inputs=classifier.inputs, outputs=classifier.layers[1].output)
#
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
#
img = load_img('D:/Collection_Dataset/wine_label_test1/check/h206.jpg', target_size=(64, 64))
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
#
from matplotlib import pyplot
filter_shape_row = 4
filter_shape_column = 8
ix = 1
for _ in range(filter_shape_row):
	for _ in range(filter_shape_column):
		# specify subplot and turn of axis
		ax = pyplot.subplot(filter_shape_row, filter_shape_column, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()
#
#
from keras import models
layer_outputs = [layer.output for layer in classifier.layers[:12]] 
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) 
# Creates a model that will return these outputs, given the model input
activations = activation_model.predict(img) 
# Returns a list of five Numpy arrays: one array per layer activation
first_layer_activation = activations[0]
print(first_layer_activation.shape)
#
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
#
layer_names = []
for layer in classifier.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
#
images_per_row = 8
#for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
layer_selected = 0
layer_name = layer_names[layer_selected]
layer_activation = activations[layer_selected]
n_features = layer_activation.shape[-1] # Number of features in the feature map
size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
display_grid = np.zeros((size * n_cols, images_per_row * size))
for col in range(n_cols): # Tiles each filter into a big horizontal grid
    for row in range(images_per_row):
        channel_image = layer_activation[0,
                                         :, :,
                                         col * images_per_row + row]
        channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size, # Displays the grid
                     row * size : (row + 1) * size] = channel_image
scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1],
                    scale * display_grid.shape[0]))
plt.title(layer_name)
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')
#
#images_per_row = 12 # will cause memory error
#for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
#    n_features = layer_activation.shape[-1] # Number of features in the feature map
#    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
#    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
#    display_grid = np.zeros((size * n_cols, images_per_row * size))
#    for col in range(n_cols): # Tiles each filter into a big horizontal grid
#        for row in range(images_per_row):
#            channel_image = layer_activation[0,
#                                             :, :,
#                                             col * images_per_row + row]
#            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
#            channel_image /= channel_image.std()
#            channel_image *= 64
#            channel_image += 128
#            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#            display_grid[col * size : (col + 1) * size, # Displays the grid
#                         row * size : (row + 1) * size] = channel_image
#    scale = 1. / size
#    plt.figure(figsize=(scale * display_grid.shape[1],
#                        scale * display_grid.shape[0]))
#    plt.title(layer_name)
#    plt.grid(False)
#    plt.imshow(display_grid, aspect='auto', cmap='viridis')

# In[]: Analyze features and weightings:
#
check_layer = layer_activation # 
