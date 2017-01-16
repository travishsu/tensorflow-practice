import pickle as pkl

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

compresses_dim = 2
PKL_DIR = './'

def load():
    with open(PKL_DIR + 'train.pkl', 'rb') as input:
        train = pkl.load(input)
        train_x = train['x']
        train_y = train['y']

    with open(PKL_DIR + 'valid.pkl', 'rb') as input:
        valid = pkl.load(input)
        valid_x = valid['x']
        valid_y = valid['y']

    with open(PKL_DIR + 'test.pkl', 'rb') as input:
        test_x = pkl.load(input)

    return train_x, train_y, valid_x, valid_y, test_x

def flat_feature(X):
    return X.reshape(X.shape[0], -1)

def printout(y, filename='submit.csv'):
    df = DataFrame({
                    'ImageId': range(1, y.shape[0]+1),
                    'Label'  : y
                   })
    df.to_csv(filename, index=False)

def compressing(train_x, valid_x, test_x, compresses_dim = 16):
    pca = PCA(n_components=compresses_dim)
    new_train = pca.fit_transform(train_x)
    new_valid = pca.fit_transform(valid_x)
    new_test  = pca.fit_transform(test_x)
    return new_train, new_valid, new_test

def create_model(X, y):
    clf = SVC(C=1, kernel='linear')
    clf.fit(X, y)
    return clf

def main():
    train_x, train_y, valid_x, valid_y, test_x = load()

    train_x, valid_x, test_x = compressing(flat_feature(train_x), flat_feature(valid_x), flat_feature(test_x), compresses_dim)
    model = create_model(train_x, train_y)

    score = model.predict(valid_x, valid_y)
    test_pred = model.predict(test_x)

    print("Validation score: {}".format(score))
    printout(test_pred)

if __name__ == '__main__':
    main()
