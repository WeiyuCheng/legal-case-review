from datapre import *
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import GRU, MaxPooling1D, Bidirectional
import pandas as pd
import time
from keras.models import load_model
from keras.layers import Activation, Add
from keras.utils import plot_model


if __name__ == '__main__':
        print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        print('accusation')

       
        num_words = 80000
        maxlen = 400
        kernel_size = 3
        DIM = 512
        batch_size = 512
        print('num_words = 80000, maxlen = 400')

        # fact数据集
        fact_train, fact_test = train_test_split(fact, test_size=0.05, random_state=1)

        # 标签数据集
        l = np.transpose(l)
        labels_train, labels_test = train_test_split(l, test_size=0.05, random_state=1)




        

        data_input = Input(shape=[maxlen])
        word_vec = Embedding(input_dim=num_words + 1,
                            input_length=maxlen,
                            output_dim=DIM,
                            mask_zero=0,
                            name='Embedding')(data_input)
        block1 = block(x=word_vec, kernel_size=kernel_size)
        block2 = block(x=block1, kernel_size=kernel_size)
        x = GlobalMaxPool1D()(block2)
        x = BatchNormalization()(x)
        x = Dense(1000, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=data_input, outputs=x)
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        plot_model(model, './resnet.png', show_shapes=True)
        n_start = 6
        n_end = 13
        score_list1 = []
        score_list2 = []
        for i in range(n_start, n_end):
            model.fit(x=fact_train, y=labels_train, batch_size=batch_size, epochs=1, verbose=2)

            model.save('./model/%d_%d/accusation/RES_epochs_%d.h5' % (num_words, maxlen,i))

            y = model.predict(fact_test[:])
            y2 = predict2half(y)
          

            print('%s accu:' % i)
           
            # 只取置信度大于0.5的准确率
            s2 = [(labels_test[i] == y2[i]).min() for i in range(len(y2))]
            print(sum(s2) / len(s2))
           

            print('%s f1:' % i)
          
            # 只取置信度大于0.5的准确率
            s5 = f1_avg(y_pred=y2, y_true=labels_test)
            print(s5)
           

            score_list1.append([i,sum(s2) / len(s2)])
            score_list2.append([i,  s5])

        print(pd.DataFrame(score_list1))
        print("mean accuracy", np.mean(score_list1))
        print(pd.DataFrame(score_list2))
        print("mean f1", np.mean(score_list2))
      
        print('end', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        print('#####################\n')
    
       