# define vars
input_num_units = 400
hidden_num_units = 50
output_num_units = 1

epochs = 20
batch_size = 512

# import keras modules
from datapre import *
from keras.models import Sequential
from keras.layers import InputLayer, Convolution2D, MaxPooling2D, Flatten, Dense

if __name__ == '__main__':
        print('Loading data...')
        # fact数据集
        train_x, val_x = train_test_split(fact, test_size=0.05, random_state=1)

        # 标签数据集
        l = np.transpose(l)
        train_y, val_y = train_test_split(l, test_size=0.05, random_state=1)

        print('Build model...')
        # create model
        model = Sequential([
        Dense(units=hidden_num_units, input_dim=input_num_units, activation='relu'),
        Dense(units=output_num_units, input_dim=hidden_num_units, activation='sigmoid'),
        ])

        # compile the model with necessary attributes
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


        trained_model = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))