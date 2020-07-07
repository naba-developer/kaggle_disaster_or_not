import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from get_features import get_features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
TRAIN_PATH = 'train.csv'

def start_training():
    df = pd.read_csv(TRAIN_PATH)
    print('Number of training sentences: ', len(df))
    #df = df.sample(200)
    labels  =  df['target']
    print(labels)
    features = get_features(df)
    print(len(features))
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)
    print(lr_clf.score(test_features, test_labels))
    return lr_clf


if __name__ == "__main__":
    start_training()



