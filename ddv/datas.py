from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.utils import plot_model
from keras.utils import to_categorical
import pickle
'''
load_txt = open('2right/data.txt','rb')
data = pickle.load(load_txt)
load_txt.close()
'''
# Initialing the CNN
classifier = Sequential()

# Step 1 - Convolutio Layer
classifier.add(Convolution2D(32, (3, 3), data_format="channels_last", input_shape = (320, 64, 3), activation = 'relu'))

#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Adding second convolution layer
classifier.add(Convolution2D(32, (3,  3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Adding 3rd Concolution Layer
classifier.add(Convolution2D(64, (3,  3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))


#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(256, activation = 'relu'))
#classifier.add(Dropout(0.5))
classifier.add(Dense(2, activation = 'softmax'))

#Compiling The CNN
classifier.compile(
              optimizer = optimizers.Adam(lr = 0.001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'datas/train',
        target_size=(320, 64),
        batch_size=5,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'datas/test',
        target_size=(320, 64),
        batch_size=7,
        class_mode='categorical')
'''
load_txt = open('datas/train/train-l2r','rb')
data = pickle.load(load_txt)
load_txt.close()
'''
model = classifier.fit_generator(
        training_set,
        steps_per_epoch=1,
        epochs=100,
        validation_data = test_set,
        validation_steps = 10
      )

###Saving the model
import h5py
classifier.save('Trained_model.h5')

print(model.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
