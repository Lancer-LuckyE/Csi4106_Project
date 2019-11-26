import csv
from sklearn.model_selection import train_test_split


def csv2dataset(path):
    with open(path, 'r') as target:
        reader = csv.reader(target)
        with open('../data/train.text', 'w') as txt, open('../data/train.emoji', 'w') as emoji:
            for row in reader:
                txt.write(row[0] + '\n')
                emoji.write(row[1] + ' ' + row[2] + '\n')


def split_dataset(txt_path, emj_path):
    X = []
    y = []
    with open(txt_path, 'r') as target:
        for line in target:
            X.append(line)
    with open(emj_path, 'r') as target:
        for line in target:
            y.append(line)

    X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(X, y, test_size=0.1, random_state=10)

    X_test, X_validation, y_test, y_validation = train_test_split(X_test_tmp, y_test_tmp, test_size=0.5,
                                                                  random_state=10)

    with open('data/t2e_train.text', 'w') as train:
        for line in X_train:
            train.write(line)
    with open('data/t2e_train.emoji', 'w') as train:
        for line in y_train:
            train.write(line)

    with open('data/t2e_test.text', 'w') as test:
        for line in X_test:
            test.write(line)
    with open('data/t2e_test.emoji', 'w') as test:
        for line in y_test:
            test.write(line)

    with open('data/t2e_validation.text', 'w') as vald:
        for line in X_validation:
            vald.write(line)
    with open('data/t2e_validation.emoji', 'w') as vald:
        for line in y_validation:
            vald.write(line)


if __name__ == "__main__":
    path = 'data/TEST_OUTPUT.csv'
    csv2dataset(path)

    split_dataset('data/train.text', 'data/train.emoji')
