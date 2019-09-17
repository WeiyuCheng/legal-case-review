
import numpy as np
import h5py
from keras.models import model_from_json

np.random.seed(1337)  # for reproducibility
from datapre import *
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import pickle  as pk




if __name__ == '__main__':
        # Embedding
        num_words = 80000
        maxlen = 400
        kernel_size = 3
        DIM = 512
        batch_size = 512

        # Convolution
        #filter_length = 3
        nb_filter = 64
        pool_length = 2

        # LSTM
        lstm_output_size = 1

        # Training
        batch_size = 512
        nb_epoch = 20


        print('Loading data...')
        # fact数据集
        X_train, X_test = train_test_split(fact, test_size=0.05, random_state=1)

        # 标签数据集
        l = np.transpose(l)
        y_train, y_test = train_test_split(l, test_size=0.05, random_state=1)

        print('Build model...')

        model = Sequential()
        model.add(Embedding(input_dim=num_words + 1,
                        input_length=maxlen,
                        output_dim=DIM,
                        mask_zero=0,
                        name='Embedding'))
        model.add(Dropout(0.2))
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=10,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))
        model.add(MaxPooling1D(pool_length=pool_length))
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=5,
                                border_mode='valid',
                                activation='sigmoid',
                                subsample_length=1))
        model.add(MaxPooling1D(pool_length=pool_length))

        model.add(LSTM(lstm_output_size))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        print('Train...')
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                validation_data=(X_test, y_test))

        #json_string = model.to_json()
        #open('my_model_rat.json', 'w').write(json_string)
        #model.save_weights('my_model_rat_weights.h5')
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
        print('Test loss:', score)
        print('Test accuracy:', acc)
        print('***********************************************************************')