from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

def buildin_naive_bayes():
    df = pd.read_csv('data/TEST_OUTPUT.csv', encoding='utf-8')
    train_texts, test_texts, train_emoji_1, test_emoji_1 = train_test_split(df['text'], df['emoji_1'], test_size=0.1,
                                                                            random_state=15)
    _, _, train_emoji_2, test_emoji_2 = train_test_split(df['text'], df['emoji_2'], test_size=0.1,
                                                         random_state=15)
    # fit on the training set
    # transform on the test set
    count_vect_emoji_1 = CountVectorizer()
    train_count_emoji_1 = count_vect_emoji_1.fit_transform(train_texts)
    test_count_emoji_1 = count_vect_emoji_1.transform(test_texts)

    count_vect_emoji_2 = CountVectorizer()
    train_counts_emoji_2 = count_vect_emoji_2.fit_transform(train_texts)
    test_counts_emoji_2 = count_vect_emoji_2.transform(test_texts)

    # train the model
    clf_nb_emoji_1 = MultinomialNB().fit(train_count_emoji_1, train_emoji_1)
    clf_nb_emoji_2 = MultinomialNB().fit(train_counts_emoji_2, train_emoji_2)

    # predict on the test set
    predicted_emoji_1 = clf_nb_emoji_1.predict(test_count_emoji_1)
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


if __name__ == "__main__":
    buildin_naive_bayes()

