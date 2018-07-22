from PIL import Image
import pickle
from keras.utils import plot_model
import numpy as np
longimg = np.zeros(20480)
longimg2 = longimg.reshape(320,64)
load_txt = open('datas/train/train-l2r/7.txt','rb')
arr = pickle.load(load_txt)
load_txt.close()
z = 0
for i in range(5):
    for j in range(64):
        longimg2[z] = arr[i][j]
        z = 1 + z
print (longimg2)

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im
new_im = MatrixToImage(longimg2)
new_im.show()
new_im.save('lena_7.jpg')
# for i in arr:
#     print (i)

#img = Image.fromarray(arr)
#img.show()
