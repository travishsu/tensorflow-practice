import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split

DIR_IN = 'inputs/'
VALID_SIZE = 0.2
def split():
    train = pd.read_csv(DIR_IN + 'train.csv')
    test = pd.read_csv(DIR_IN + 'test.csv')

    train_mtx = train.values
    test_mtx = test.values

    n_train = train_mtx.shape[0]
    n_test = test_mtx.shape[0]

    train_x = train_mtx[:, 1:].reshape(n_train, 28, 28)
    train_y = train_mtx[:, 0]
    test_x = test_mtx.reshape(n_test, 28, 28)

    return train_x, train_y, test_x

def save(tr_x, tr_y, valid_x, valid_y, te_x):
    with open('train.pkl', 'wb') as output:
        pkl.dump({'x':tr_x.astype('float32'), 'y':tr_y}, output)

    with open('valid.pkl', 'wb') as output:
        pkl.dump({'x':valid_x.astype('float32'), 'y':valid_y}, output)

    with open('test.pkl', 'wb') as output:
        pkl.dump(te_x.astype('float32'), output)

def main():
    train_x, train_y, test_x = split()
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = VALID_SIZE)
    save(train_x, train_y, valid_x, valid_y, test_x)

if __name__ == '__main__':
    main()
