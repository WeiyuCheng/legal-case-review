from datapre import *
from keras.layers import Conv1D, BatchNormalization, Activation, GlobalMaxPool1D
def textcnn_one(word_vec=None, kernel_size=1, filters=512):
    x = word_vec
    x = Conv1D(filters=filters, kernel_size=[kernel_size], strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv1D(filters=filters, kernel_size=[kernel_size], strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = GlobalMaxPool1D()(x)

    return x


from sklearn.metrics.scorer import f1_score
def predict2half(predictions):
    return np.where(predictions>0.5,1.0,0.0)

def f1_avg(y_pred, y_true):
    '''
    mission 1&2
    :param y_pred:
    :param y_true:
    :return:
    '''
    f1_micro = f1_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='micro')
    f1_macro = f1_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='macro')
    return (f1_micro + f1_macro) / 2

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Embedding, Input,Dropout
from keras.layers import BatchNormalization, Concatenate
import pandas as pd
import time
from keras.models import load_model
from keras.utils import plot_model
if __name__ == '__main__':
        print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    

        num_words = 80000
        maxlen = 400
        filters = 256
        print('num_words = 80000, maxlen = 400')

        # fact数据集
        fact_train, fact_test = train_test_split(fact, test_size=0.05, random_state=1)

        # 标签数据集
        l = np.transpose(l)
        labels_train, labels_test = train_test_split(l, test_size=0.05, random_state=1)




        data_input = Input(shape=[maxlen])
        word_vec = Embedding(input_dim=num_words + 1,
                            input_length=maxlen,
                            output_dim=512,
                            mask_zero=False,
                            name='Embedding')(data_input)

        x1 = textcnn_one(word_vec=word_vec, kernel_size=1, filters=filters)
        x2 = textcnn_one(word_vec=word_vec, kernel_size=2, filters=filters)
        x3 = textcnn_one(word_vec=word_vec, kernel_size=3, filters=filters)
        x4 = textcnn_one(word_vec=word_vec, kernel_size=4, filters=filters)
        x5 = textcnn_one(word_vec=word_vec, kernel_size=5, filters=filters)


        x = Concatenate(axis=1)([x1, x2, x3, x4, x5])
        x = BatchNormalization()(x)
        x = Dense(1000, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=data_input, outputs=x)
        plot_model(model, './textcnn_cnn.png', show_shapes=True)
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        plot_model(model, './resnet.png', show_shapes=True)
        
        
        n_start = 1
        n_end = 21
        score_list1 = []
        score_list2 = []
        print('fact_train',fact_train)
        for i in range(n_start, n_end):
            model.fit(x=fact_train, y=labels_train, batch_size=512, epochs=1, verbose=1)

            model.save('./model/%d_%d/accusation/TextCNN_%d_epochs_%d.h5' % (num_words, maxlen, filters, i))

            y = model.predict(fact_test[:])
            y2 = predict2half(y)
    

            print('%s accu:' % i)
            s2 = [(labels_test[i] == y2[i]).min() for i in range(len(y2))]
            print(sum(s2) / len(s2))
        
            print('%s f1:' % i)
            s5 = f1_avg(y_pred=y2, y_true=labels_test)
            print(s5)

            score_list1.append([i, sum(s2) / len(s2)])
            score_list2.append([i, s5])

        print(pd.DataFrame(score_list1))
        print(pd.DataFrame(score_list2))
        print('end', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        print('#####################\n')
