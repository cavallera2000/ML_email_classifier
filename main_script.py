import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from preprocessing import preprocess


def train_test(train='train.csv', test='test.csv'):
    start_tt = time.time()

    # Importing the data
    train_df = pd.read_csv(train, header=None)
    train_df.columns = ['text', 'class']
    test_df = pd.read_csv(test, header=None, usecols=[0], squeeze=True)

    # Preprocessing #
    start_prep = time.time()

    train_df.text = preprocess(train_df.text)
    test_df = preprocess(test_df)
    train_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)

    print("Preprocessing time: %s seconds" % (time.time() - start_prep))

    # Training #
    start_train = time.time()

    x_train = train_df['text']
    y_train = train_df['class']
    x_test = test_df

    # Encode train classes: ham=0, spam=1
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)

    # Tf_Idf Vectorize and normalize
    tf = TfidfVectorizer(ngram_range=(1,3), min_df=3)
    x_train = tf.fit_transform(x_train)
    x_train = normalize(x_train)
    x_test = tf.transform(x_test)
    x_test = normalize(x_test)

    # SVC
    model = SVC(kernel='linear',
                C=10.0,
                class_weight='balanced',
                break_ties=True)

    model.fit(x_train, y_train)
    print("Training time: %s seconds" % (time.time() - start_train))

    # Testing #
    start_test = time.time()

    predict = pd.Series(encoder.inverse_transform(model.predict(x_test)))

    print("Testing time: %s seconds" % (time.time() - start_test))

    print("Total execution time: %s seconds" % (time.time() - start_tt))
    return predict.to_csv('prediction.txt', header=False, index=False)


train_test()