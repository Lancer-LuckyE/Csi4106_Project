import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
import sklearn
from collections import Counter
import math


def transform(headlines, test):
    headlines.append(nltk.word_tokenize(test))

    tokens = [w for s in headlines for w in s]

    results = []
    label_enc = sklearn.preprocessing.LabelEncoder()
    onehot_enc = sklearn.preprocessing.OneHotEncoder()

    encoded_all_tokens = label_enc.fit_transform(list(set(tokens)))
    encoded_all_tokens = encoded_all_tokens.reshape(len(encoded_all_tokens), 1)

    onehot_enc.fit(encoded_all_tokens)

    for headline_tokens in headlines:
        encoded_words = label_enc.transform(headline_tokens)
        encoded_words = onehot_enc.transform(encoded_words.reshape(len(encoded_words), 1))
        results.append(np.sum(encoded_words.toarray(), axis=0))
    return results


def eucldistanceCal(fromedResults):
    results = []
    for i, fromedResult in enumerate(fromedResults):
        if i < len(fromedResults) - 1:
            score = sklearn.metrics.pairwise.euclidean_distances([fromedResults[i]], [fromedResults[-1]])[0][0]
            results.append((score, i))
    return results


def cosdistanceCal(fromedResults):
    results = []
    for i, fromedResult in enumerate(fromedResults):
        if i < len(fromedResults) - 1:
            score = sklearn.metrics.pairwise.euclidean_distances([fromedResults[i]], [fromedResults[-1]])[0][0]
            results.append((score, i))
    return results


"""
    Finding the posistion (from lookup table) of word instead of using 1 or 0
    to prevent misleading of the meaning of "common" word
"""


def calculate_position(values):
    x = []
    for pos, matrix in enumerate(values):
        print(matrix)
        if matrix > 0:
            x.append(pos)
    return x


"""
    Since scikit-learn can only compare same number of dimension of input. 
    Add padding to the shortest sentence.
"""


def padding(sentence1, sentence2):
    x1 = sentence1.copy()
    x2 = sentence2.copy()

    diff = len(x1) - len(x2)

    if diff > 0:
        for i in range(0, diff):
            x2.append(-1)
    elif diff < 0:
        for i in range(0, abs(diff)):
            x1.append(-1)
    return x1, x2


def jaccarddistanceCal(fromedResults):
    results = []
    y_actual = calculate_position(fromedResults[-1])

    for i, fromedResult in enumerate(fromedResults):
        if i < len(fromedResults) - 1:
            y_compare = calculate_position(fromedResults[i])
            x1, x2 = padding(y_actual, y_compare)
            score = sklearn.metrics.jaccard_similarity_score(x1, x2)
            results.append((score, i))
    return results


def knn(data, query, k):
    tokenizedResult = []
    print('testing ', query)
    # 3. For each example in the data
    for ele in data:
        tokenizedResult.append(nltk.word_tokenize(ele))

    fromedResult = transform(tokenizedResult, query)
    # 3.1 Calculate the distance between the query example and the current
    # example from the data.
    neighbor_distances_and_indices = eucldistanceCal(fromedResult)

    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)

    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i] for distance, i in k_nearest_distances_and_indices]

    # 7. If regression (choice_fn = mean), return the average of the K labels
    # 8. If classification (choice_fn = mode), return the mode of the K labels
    return k_nearest_distances_and_indices, k_nearest_labels


def mean(labels):
    return sum(labels) / len(labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]


def main():
    df = pd.read_csv("TEST_OUTPUT_500000.csv", encoding="ISO-8859-1")
    ddf = df.sample(n=1000, random_state=5)
    ddf.columns = ['A', 'B', 'C']
    train_reviews, test_reviews, train_tags, test_tags = train_test_split(ddf['A'],
                                                                          ddf['B'],
                                                                          test_size=0.1,
                                                                          random_state=10)
    train_tags = train_tags.to_numpy()
    train_reviews = train_reviews.to_numpy()
    # Testing set (what we will use to test the trained model)
    test_tags = test_tags.to_numpy()
    test_reviews = test_reviews.to_numpy()

    for test_review in test_reviews:
        reg_query = test_review
        reg_k_nearest_neighbors, reg_prediction = knn(
            train_reviews, reg_query, k=3
        )

        print(reg_k_nearest_neighbors)
        print(reg_prediction)


if __name__ == '__main__':
    main()
