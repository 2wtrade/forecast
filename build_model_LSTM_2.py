from tensorflow import keras

import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt

# Load input data set
df = pd.read_csv('input.csv',header=0)#, nrows=1000)
print(df.shape)

# Prdictor matrix
X = df.drop(['Target'], axis=1)
X = X.values

n=X.shape[1]

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

#X = X.astype('float32')

# Target
y = df['Target']
y = y.values
#y = y.astype('int')

# Split into training and testing data
# Using hold-out cross validation
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

train_n=int(X.shape[0]*0.7)
X_train=X[0:train_n]
X_test=X[train_n:X.shape[0]]
y_train=y[0:train_n]
y_test=y[train_n:y.shape[0]]

# Simple deep LSTM
model = keras.Sequential()
model.add(keras.layers.LSTM(units=96, activation='elu', return_sequences=True, input_shape=(1, n)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LSTM(units=48, activation='elu', return_sequences=True))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(units=1, activation='sigmoid'))

# Compile model
opt = keras.optimizers.Adam(lr=0.1, amsgrad=True)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# !!!!!!!!!!!!!!!!!!!!!!!!!! Set number epoches !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model.fit(X_train, y_train, epochs=250, validation_split=0.3, batch_size=50, )

# save model and architecture to single file
model.save("model_lstm.h5")
print("Saved model to disk")

# Testing model
# Load model
model = keras.models.load_model('model_lstm.h5')
# Summarize model.
model.summary()
# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:",np.round(score[1]*100,2), "%")

