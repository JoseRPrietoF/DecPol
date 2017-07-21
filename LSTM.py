from preprocess import Data
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, Activation
from keras.layers import Dropout, BatchNormalization, GaussianNoise
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

# TODO

dataset = Data(path="data/", stem=False, simply=True, stop_word = False, delete_class=['0','000'],codif = 'bagofwords',max_features=None)
X_train, y_train, X_test, y_test = dataset.train_test_split(0.8)

x,y = dataset.get_non_coded()

# create the model
embedding_vecor_length = 2
model = Sequential()
model.add(Embedding(n_features, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=8)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



