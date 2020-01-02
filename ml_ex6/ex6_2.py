import re
from nltk.stem import PorterStemmer
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
import pandas as pd


def vocab_dict(list):
    dict = {}
    for i in list:
        value, key = i.split("\t")[:]
        dict[key] = value
    return dict


def process_email(email, vocab):
    email = email.lower()     # lower case
    email = re.sub('http://[^\s]*' or 'https://[^\s]*', 'httpaddr',  email)     # normalizing URLs
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)     # normalizing email addresses
    email = re.sub('[0-9]+', 'number', email)     # normalizing numbers
    email = re.sub('[$]+', 'dollar', email)     # normalizing dollars

    # strip all special characters
    special_char = ["<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":"]
    for char in special_char:
        email = email.replace(str(char), "")
        email = email.replace("\n", " ")

    # stem the word
    ps = PorterStemmer()
    email = [ps.stem(token) for token in email.split(" ")]
    email = " ".join(email)

    # process the email and return word_indices
    word_indices = []
    for char in email.split():
        if len(char) > 1 and char in vocab:
            word_indices.append(int(vocab[char]))
    return word_indices


def email_features(word_indices, vocab):
    n = len(vocab)
    features = np.zeros((n, 1))
    for i in word_indices:
        features[i] = 1
    return features


def top_predictors(spam_svc, vocab):
    weights = spam_svc.coef_[0]
    n = len(weights)
    weights = np.hstack((np.arange(1, n+1).reshape(n, 1),weights.reshape(n, 1)))
    weights_df = pd.DataFrame(weights)
    weights_df.sort_values(by=[1],ascending = False,inplace=True)

    predictors = []
    for i in weights_df[0][:15]:
        for keys, values in vocab.items():
            if str(int(i)) == values:
                predictors.append(keys)
    return predictors


def main():
    file_contents = open('emailSample1.txt', 'r').read()
    vocab_list = open('vocab.txt', 'r').read()

    vocab_list = vocab_list.split('\n')[:-1]
    vocab = vocab_dict(vocab_list)
    word_indices = process_email(file_contents, vocab)
    features = email_features(word_indices, vocab)

    # print("Length of feature vector: ", len(features))
    # print("Number of non-zero entries: ", np.sum(features))

    spam_train = loadmat('spamTrain.mat')
    X_train, y_train = spam_train['X'], spam_train['y']
    spam_svc = SVC(C=0.1, kernel='linear')
    spam_svc.fit(X_train, y_train.ravel())
    print("Training Accuracy:", (spam_svc.score(X_train, y_train.ravel())) * 100, "%")

    spam_test = loadmat('spamTest.mat')
    X_test, y_test = spam_test['Xtest'], spam_test['ytest']
    spam_svc.predict(X_test)
    print("Test Accuracy:", (spam_svc.score(X_test, y_test.ravel())) * 100, "%")

    main_predictors = top_predictors(spam_svc, vocab)
    print('Top predictors of spam: ', main_predictors)


main()
