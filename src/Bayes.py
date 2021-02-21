import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection as ms
from sklearn import metrics

import os


def get_dataset():
    labels = ['comment', 'label', 'fold']
    comments = np.empty((0, 9), dtype=str)
    label = np.empty((1, 0), dtype=int)
    fold = np.empty((1, 0), dtype=int)

    for polarity in ['pos', 'neg']:
        d = "../venv/txt_sentoken/" + polarity + "/"
        all_files = os.listdir(d)
        for f in all_files:
            fold = np.append(fold, [[(int(f[2:5]) // 100) + 1]])
            fh = open(d + f, 'r')
            comment = fh.read().lower().replace("\n", "")
            fh.close()
            comments = np.append(comments, comment)

        label = np.append(label, np.zeros(len(comments)) if polarity == 'neg' else np.ones(len(comments)))

    df = pd.DataFrame(columns=labels)
    df['comment'] = pd.Series(comments)
    df['label'] = pd.Series(label)
    df['fold'] = pd.Series(fold)

    return df

def bayes():
    df = get_dataset()

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.3, stop_words='english')
    X_train = vectorizer.fit_transform(df.comment)

    for name,clf in [('Bernoulli',BernoulliNB()),('Multinomial',MultinomialNB())]:
        scores = ms.cross_validate(clf, X_train, df.label, scoring=['accuracy', 'average_precision', 'precision', 'recall', 'f1'], cv=10)
        print('---------------------------')
        print('  Algorithm: '+ name)
        print('---------------------------')
        print("Training scores")
        print(scores)
        print("---------------")
        test = ms.cross_val_predict(clf, X_train, df.label, cv=10)
        score = metrics.roc_auc_score(df.label, test, average='micro')
        print("Test roc auc score micro")
        print(score)
        print("------------")
        score = metrics.roc_auc_score(df.label, test, average='macro')
        print("Test roc auc score macro")
        print(score)
        print("------------")
        score = metrics.accuracy_score(df.label, test)
        print("Test accuracy score")
        print(score)
        print("------------")
        score = metrics.cohen_kappa_score(df.label, test)
        print("Test cohen_kappa_score score")
        print(score)
        print("------------")
        score = metrics.precision_recall_fscore_support(df.label, test, average='micro')
        print("Test precision_recall_fscore score micro")
        print(score)
        print("------------")
        score = metrics.precision_recall_fscore_support(df.label, test, average='macro')
        print("Test precision_recall_fscore score macro")
        print(score)
        print("--------------------------------------------------------------------------------------")

if __name__ == '__main__':
    bayes()