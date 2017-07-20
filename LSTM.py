import preprocess
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


tf_idf = False

X_train, y_train, X_test, y_test,n_features, _  = preprocess.get_data(simply=True,tfidf=tf_idf)

print("****" * 100)
if tf_idf:
    print("Shape")
    print(X_train[0].shape)
    print("****" *100)
else:
    for i in range(5):
        print("*"*10)
        #print(X_train[i])
        #print(len(X_train[i]))

print("Number of features %d " % n_features)
# truncate and pad input sequences
max_review_length = 15
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

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



