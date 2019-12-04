import csv
from sklearn.model_selection import train_test_split


def csv2dataset(path):
    with open(path, 'r') as target:
        reader = csv.reader(target)
        with open('../data/all_data.text', 'w') as txt, open('../data/all_data.emoji', 'w') as emoji:
            for row in reader:
                txt.write(row[0] + '\n')
                emoji.write(row[1] + ' ' + row[2] + '\n')


def split_dataset(txt_path, emoji_path, validation=False):
    X = []
    y = []
    with open(txt_path, 'r') as target:
        for line in target:
            X.append(line)
    with open(emoji_path, 'r') as target:
        for line in target:
            y.append(line)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)

    with open('../data/t2e_train.text', 'w') as train:
        for line in X_train:
            train.write(line)
    with open('../data/t2e_train.emoji', 'w') as train:
        for line in y_train:
            train.write(line)

    if validation:
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5,
                                                                      random_state=10)
        with open('../data/t2e_validation.text', 'w') as validation:
            for line in X_validation:
                validation.write(line)
        with open('../data/t2e_validation.emoji', 'w') as validation:
            for line in y_validation:
                validation.write(line)

    with open('../data/t2e_test.text', 'w') as test:
        for line in X_test:
            test.write(line)
    with open('../data/t2e_test.emoji', 'w') as test:
        for line in y_test:
            test.write(line)


if __name__ == "__main__":
    path = '../data/TEST_OUTPUT_200000.csv'
    csv2dataset(path)
    split_dataset('../data/all_data.text', '../data/all_data.emoji', validation=False)
