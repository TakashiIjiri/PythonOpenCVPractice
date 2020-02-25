import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

#詳解ディープラーニング p. 094 より

np.random.seed(0)

# modeling
# y = w^t x + b
model = Sequential([Dense(input_dim=2, units=1), Activation("sigmoid")])
model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.1))

#dataset
X = np.array([[0,0], [1,0], [0,1], [1,1]])
Y = np.array([[0], [1], [1], [1]])
model.fit(X,Y, epochs=200, batch_size=1)


for i in range(10) :
    predict_x = model.predict( np.array( [train_x[i]] ), batch_size = 1)

    predict_x = predict_x * 255
    predict_x.reshape(28, 28)

    cv2.imwrite( str(i) + "pred.png", )



#results
classified = model.predict_classes(X, batch_size=1)
print(Y==classified)

prob = model.predict_proba(X, batch_size=1)
print(prob)
