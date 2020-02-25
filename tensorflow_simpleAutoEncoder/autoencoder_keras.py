import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras import backend as K
import gzip



#LOAD MNIST

def open_mnist_image(fname) :
    f = gzip.open(fname, 'rb')
    data = np.frombuffer( f.read(), np.uint8, offset=16)
    f.close()
    return data.reshape((-1, 784)) # (n, 784)の行列に整形, nは自動で決定

def open_mnist_label(fname):
    f = gzip.open(fname, 'rb')
    data = np.frombuffer( f.read(), np.uint8, offset=8 )
    f.close()
    return data.flatten() # (n, 1)の行列に整形, nは自動で決定

fname_train_img   = "./mnist/train-images-idx3-ubyte.gz"
fname_train_label = "./mnist/train-labels-idx1-ubyte.gz"
x_train = open_mnist_image( fname_train_img   )
t_train = open_mnist_label( fname_train_label )
print(x_train.shape)
print(t_train.shape)

#normalize
x_train = x_train / 255.0


# modeling y = f( w2*f(w1^t x + b1)+b2)
io_dim  = x_train[0].shape[0]
mid_dim = 30
model = Sequential()
model.add( Dense(input_dim=io_dim, units=mid_dim, activation="linear"))
model.add( Dense(input_dim=mid_dim, units=io_dim, activation="linear"))
model.compile(loss="mean_squared_error", optimizer=SGD(lr=0.05))

#fitting
np.random.seed(0)
model.fit(x_train, x_train, epochs=50, batch_size=20)


#output results
for i in range(10) :
    pred_x = model.predict( x_train[i:i+1], batch_size = 1)
    img_out = np.clip( pred_x.reshape(28, 28), 0, 1)
    cv2.imwrite(str(i) + "pred.png", np.uint8(img_out * 255))
    img_out = x_train[i].reshape(28, 28)
    cv2.imwrite(str(i) + "input.png", np.uint8(img_out * 255))

#output decorder kernels
print( model.layers[0].bias.shape )
print( model.layers[0].kernel.shape )
print( model.layers[1].bias.shape )
print( model.layers[1].kernel.shape )

decorder_kernel = model.layers[0].kernel
for i in range(30) :
    kernel_i = K.eval(decorder_kernel[:,i])
    kernel_i = kernel_i * 255 + 128
    img_out = kernel_i.reshape(28,28)
    cv2.imwrite("kernel_" + str(i) + ".png", np.uint8(img_out))
