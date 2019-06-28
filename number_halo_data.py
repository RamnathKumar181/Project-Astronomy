import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# from sklearn.metrics import cohen_kappa_score
# from keras.callbacks import ReduceLROnPlateau
# from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

# Importing the dataset
dataset = pd.read_csv('halo_info.csv')
temp = dataset.iloc[:,:].values

X = []
y = []
for i in range(len(temp)-4):
    X.append(np.reshape(temp[i:i+4,:],24))
    y.append(temp[i+4,-1])
print(X[0])


input = Input(shape=(100,))
model = Embedding(input_dim=24, output_dim=50, input_length=100)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(5, activation="softmax"))(model)  # softmax output layer

model = Model(input, out)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X, np.array(y), batch_size=32, epochs=5, validation_split=0.1, verbose=1)
