from keras.models import load_model

classifier = load_model('Trained_model.h5')

import numpy as np

from keras.preprocessing import image
from PIL import Image

img_name = input('Enter Image Name: ')
img_path = 'predicting_data/'+str(img_name)+'.jpg'

test_img = image.load_img(img_path)
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis = 0)
predict  = classifier.predict(test_img)
print('Predicted Sign is:')
print('')
if predict[0][0] == 1:
        print('Right')
elif predict[0][1] == 1:
        print('Left')
elif predict[0][2] == 1:
        print('Up')
