import pickle as pkl
import numpy as np

from pandas import DataFrame, get_dummies
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Flatten, Dropout

FC_dim = 256
PKL_DIR = './'

epoch = 15

def load():
    with open(PKL_DIR + 'train.pkl', 'rb') as input:
        train = pkl.load(input)
        train_x = train['x'].astype('float32')
        train_y = train['y']

    with open(PKL_DIR + 'valid.pkl', 'rb') as input:
        valid = pkl.load(input)
        valid_x = valid['x'].astype('float32')
        valid_y = valid['y']

    with open(PKL_DIR + 'test.pkl', 'rb') as input:
        test_x = pkl.load(input).astype('float32')

    return train_x, train_y, valid_x, valid_y, test_x

def merge_data(data1, data2):
    return np.concatenate( (data1, data2) )

def printout(y, filename='submit.csv'):
    df = DataFrame({
                    'ImageId': range(1, y.shape[0]+1),
                    'Label'  : y
                   })
    df.to_csv(filename, index=False)

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(FC_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(FC_dim))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    train_x, train_y, valid_x, valid_y, test_x = load()
    train_x = train_x.reshape(-1, 28, 28, 1)
    valid_x = valid_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)
    dummied_train_y = get_dummies(train_y).values
    dummied_valid_y = get_dummies(valid_y).values

    model = create_model(train_x.shape[1:])
    #model.fit(train_x, dummied_train_y, batch_size=128, nb_epoch=epoch, validation_data=(valid_x, dummied_valid_y))
    model.fit(merge_data(train_x, valid_x),
              merge_data(dummied_train_y, dummied_valid_y),
              batch_size=128,
              nb_epoch=epoch,
              validation_data=(valid_x, dummied_valid_y))

    test_pred = model.predict_classes(test_x)

    printout(test_pred, filename='cnn_submit.csv')

if __name__ == '__main__':
    main()
