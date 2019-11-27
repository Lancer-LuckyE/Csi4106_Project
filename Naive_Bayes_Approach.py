from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd


def buildin_naive_bayes():
    df = pd.read_csv('data/TEST_OUTPUT.csv', encoding='utf-8')
    # all_possible_emojis = df['emoji_1'].append(df['emoji_2']).drop_duplicates()
    # chunks = np.array_split(df, 5)
    clf_nb_emoji_1 = MultinomialNB()
    clf_nb_emoji_2 = MultinomialNB()

    # for c in chunks:
    train_texts, test_texts, train_emoji_1, test_emoji_1 = train_test_split(df['text'], df['emoji_1'],
                                                                            test_size=0.1, random_state=15)
    _, _, train_emoji_2, test_emoji_2 = train_test_split(df['text'], df['emoji_2'],
                                                         test_size=0.1, random_state=15)
    # fit on the training set
    # transform on the test set
    count_vect_emoji_1 = CountVectorizer()
    train_counts_emoji_1 = count_vect_emoji_1.fit_transform(train_texts)
    test_counts_emoji_1 = count_vect_emoji_1.transform(test_texts)

    count_vect_emoji_2 = CountVectorizer()
    train_counts_emoji_2 = count_vect_emoji_2.fit_transform(train_texts)
    test_counts_emoji_2 = count_vect_emoji_2.transform(test_texts)

    # train the model
    clf_nb_emoji_1 = clf_nb_emoji_1.fit(train_counts_emoji_1, train_emoji_1)
    clf_nb_emoji_2 = clf_nb_emoji_2.fit(train_counts_emoji_2, train_emoji_2)


    # predict on the test set
    predicted_emoji_1 = clf_nb_emoji_1.predict(test_counts_emoji_1)
    predicted_emoji_2 = clf_nb_emoji_2.predict(test_counts_emoji_2)

    correct = 0
    for tag, pred in zip(test_emoji_1, predicted_emoji_1):
        if tag == pred:
            correct += 1
    print("The rate of correctness on emoji_1 prediction is: %s" %(correct/test_emoji_1.size))

    correct = 0
    for tag, pred in zip(test_emoji_2, predicted_emoji_2):
        if tag == pred:
            correct += 1
    print("The rate of correctness on emoji_2 prediction is: %s" %(correct/test_emoji_2.size))


# class Emojify_NB:
#     """
#     Reimplement naive bayes
#     """
#     def __init__(self, csv_file, emoji_class):
#         self.df = pd.read_csv(csv_file, encoding='utf-8')
#         self.texts = self.df['text']
#         self.emoji = self.df[emoji_class]
#
#         self.word_count  # a dict: all words from the texts as key, number of appearance as value
#         self.emoji_class  # a list: all possible emoji
#
#     def train_test_split(self, train_size):
#         chunks = np.array_split(self.df, 100)
#         training_set = chunks[:(train_size * 100)]
#         testing_set = chunks[(train_size * 100):]
#         return pd.concat(training_set), pd.concat(testing_set)



if __name__ == "__main__":
    buildin_naive_bayes()

