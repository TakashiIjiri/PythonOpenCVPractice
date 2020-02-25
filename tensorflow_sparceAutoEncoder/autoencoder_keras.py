import numpy as np
import cv2

from keras.models import Model
from keras.layers import Dense, Input, Activation
from keras import regularizers
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



def custom_loss_mid(y_true, y_pred) : # y_true's shape=(batch_size, mid_dim)
    p = 0.03
    p_hat = K.mean( y_pred,0) + 0.000001
    #p_hat = K.print_tensor(p_hat, "hat1=")
    p_hat = p * K.log( p / p_hat ) + (1-p) * K.log( (1-p) / (1-p_hat) )
    #p_hat = K.print_tensor(p_hat,"hat2=")
    return  K.mean( p_hat )

def custom_loss_out(y_true, y_pred) : # y_true's shape=(batch_size, 728)
    #y_pred = K.print_tensor(y_pred, "aaaa=" + str( K.shape(y_pred) ))
    return K.mean( K.abs(y_pred - y_true))




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
mid_dim = 2000

layer1 = Dense(input_dim=io_dim , units=mid_dim, activation="relu", name="output1" )
layer2 = Dense(input_dim=mid_dim, units=io_dim , activation="linear" , name="output2")

inputs      = Input(shape=(io_dim,))
mid_output  = layer1(inputs    )
main_output = layer2(mid_output)
model = Model(inputs=inputs, outputs=[mid_output, main_output])

#model.compile(loss="mean_squared_error", optimizer=SGD(lr=0.05))
model.compile(loss        ={"output1":custom_loss_mid, "output2":"mean_squared_error"},#custom_loss_out},
              loss_weights={'output1': 3.0          , 'output2':1.0           },
              optimizer=SGD(lr=0.05))

model.summary()

#fitting
np.random.seed(0)
model.fit(x_train, {"output1":np.zeros((60000,mid_dim)), "output2":x_train}, epochs=50, batch_size=500)

#output results
for i in range(10) :
    pred_x = model.predict( x_train[i:i+1], batch_size = 1)[1]
    img_out = np.clip( pred_x.reshape(28, 28), 0, 1)
    cv2.imwrite(str(i) + "pred.png", np.uint8(img_out * 255))
    img_out = x_train[i].reshape(28, 28)
    cv2.imwrite(str(i) + "input.png", np.uint8(img_out * 255))


#output decorder kernels
print( model.layers[1].bias.shape )
print( model.layers[1].kernel.shape )

decorder_kernel = model.layers[1].kernel
for i in range(100) :
    kernel_i = K.eval(decorder_kernel[:,i])
    kernel_i = kernel_i * 255 + 128
    img_out = kernel_i.reshape(28,28)
    cv2.imwrite("kernel_" + str(i) + ".png", np.uint8(img_out))
