from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import metric_calculator
import pandas as pd


def buildin_naive_bayes():

    clf_nb_emoji_1 = MultinomialNB()
    clf_nb_emoji_2 = MultinomialNB()

    # load text file
    train_texts = []
    test_texts = []
    with open('data/splitted/t2e_train.text') as t2e_train_text:
        with open('data/splitted/t2e_test.text') as t2e_test_text:
            for i, line in enumerate(t2e_train_text):
                train_texts.append(line.rstrip())
            for i, line in enumerate(t2e_test_text):
                test_texts.append(line.rstrip())

    # load emoji file
    # construct training emoji
    with open('data/splitted/t2e_train.emoji') as t2e_train_emoji:
        emoji_1 = []
        emoji_2 = []
        for i, line in enumerate(t2e_train_emoji):
            el = line.split(' ')
            emoji_1.append(el[0])
            emoji_2.append(el[1].rstrip())
    train_emoji_1 = pd.Series(emoji_1)
    train_emoji_2 = pd.Series(emoji_2)

    # construct testing emoji
    with open('data/splitted/t2e_test.emoji') as t2e_test_emoji:
        emoji_1 = []
        emoji_2 = []
        for i, line in enumerate(t2e_test_emoji):
            el = line.split(' ')
            emoji_1.append(el[0])
            emoji_2.append(el[1].rstrip())
    test_emoji_1 = pd.Series(emoji_1)
    test_emoji_2 = pd.Series(emoji_2)

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

    # calculate accuracy
    predicted = []
    test_tags = []
    for e1, e2 in zip(predicted_emoji_1, predicted_emoji_2):
        ans = (e1, e2)
        predicted.append(ans)
    for e1, e2 in zip(test_emoji_1, test_emoji_2):
        tag = (e1, e2)
        test_tags.append(tag)
    print(metric_calculator.overall_accuracy(test_tags, predicted))
    print(metric_calculator.both_emoji_accuracy(test_tags, predicted))


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

