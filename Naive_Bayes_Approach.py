from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import metric_calculator
import pandas as pd


def build_in_naive_bayes():
    # df = pd.read_csv("data/TEST_OUTPUT.csv", names=["texts", "emoji_1", "emoji_2"], nrows=200000)

    clf_nb_emoji_1 = MultinomialNB()
    clf_nb_emoji_2 = MultinomialNB()

    # train_texts, test_texts, train_emoji_1, test_emoji_1 = train_test_split(df['texts'], df['emoji_1'], random_state=10, test_size=0.1)
    # _, _, train_emoji_2, test_emoji_2 = train_test_split(df['texts'], df['emoji_2'], random_state=10, test_size=0.1)

    # load text file
    train_texts = []
    test_texts = []
    with open('data/t2e_train.text') as t2e_train_text:
        with open('data/splitted/t2e_test.text') as t2e_test_text:
            for i, line in enumerate(t2e_train_text):
                train_texts.append(line.rstrip())
            for i, line in enumerate(t2e_test_text):
                test_texts.append(line.rstrip())

    # load emoji file
    # construct training emoji
    with open('data/t2e_train.emoji') as t2e_train_emoji:
        emoji_1 = []
        emoji_2 = []
        for i, line in enumerate(t2e_train_emoji):
            el = line.split(' ')
            emoji_1.append(el[0])
            emoji_2.append(el[1].rstrip())
    train_emoji_1 = pd.Series(emoji_1)
    train_emoji_2 = pd.Series(emoji_2)

    # construct testing emoji
    with open('data/t2e_test.emoji') as t2e_test_emoji:
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
    print("Overall accuracy: " + str(metric_calculator.overall_accuracy(test_tags, predicted)))
    print("Both emoji accuracy: " + str(metric_calculator.both_emoji_accuracy(test_tags, predicted)))
    print("The performance evaluation [class:(precision, recall)]: ")
    print(naive_bayes_eval(test_tags, predicted))


def naive_bayes_eval(test_tags, predcited):
    all_classes = list(dict.fromkeys(predcited))
    result = {}
    for c in all_classes:
        Tp = 0
        Tn = 0
        Fp = 0
        Fn = 0
        for t, p in zip(test_tags, predcited):
            if t == p:
                Tp += 1
            elif (t == c) and (p != c):
                Fn += 1
            elif (t != c) and (p == c):
                Fp += 1
            else:
                Tn += 1
        # precision
        precision = round(Tp / (Tp + Fp), 6)
        # recall
        recall = round(Tp / (Tp + Fn), 6)
        # set into result
        result[c] = (precision, recall)
    return result


if __name__ == "__main__":
    build_in_naive_bayes()

