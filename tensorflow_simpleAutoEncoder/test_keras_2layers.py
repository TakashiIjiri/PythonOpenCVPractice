import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

#詳解ディープラーニング p. 094 より

np.random.seed(0)

# modeling
# y = w^t x + b
model = Sequential()
model.add( Dense(input_dim=2, units=2))
model.add( Activation("sigmoid") )
model.add( Dense(input_dim=2, units=1))
model.add( Activation("sigmoid") )

model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.1))

#dataset
X = np.array([[0,0], [1,0], [0,1], [1,1]])
Y = np.array([[0], [1], [1], [1]])
model.fit(X,Y, epochs=1000, batch_size=1)


#results
classified = model.predict_classes(X, batch_size=1)
print(Y==classified)

prob = model.predict_proba(X, batch_size=1)
print(prob)
