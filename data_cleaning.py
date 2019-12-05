import csv
import pandas as pd
import numpy as np

def load_files():
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
        train_tag_1 = []
        train_tag_2 = []
        for i, line in enumerate(t2e_train_emoji):
            el = line.split(' ')
            train_tag_1.append(el[0])
            train_tag_2.append(el[1].rstrip())

    # construct testing emoji
    with open('data/splitted/t2e_test.emoji') as t2e_test_emoji:
        test_tag_1 = []
        test_tag_2 = []
        for i, line in enumerate(t2e_test_emoji):
            el = line.split(' ')
            test_tag_1.append(el[0])
            test_tag_2.append(el[1].rstrip())
    return train_texts, test_texts, train_tag_1, train_tag_2, test_tag_1, test_tag_2


def write_to_csv(text, tag1, tag2, type):
    with open('data/NB/%s_data.csv'%type, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "emoji_1", "emoji_2"])
        for t, e1, e2 in zip(text, tag1, tag2):
            writer.writerow([t, e1, e2])
        print("Done.")


def clean_low_freq_vocab():

    return


def clean_low_freq_class(train_df, test_df, emoji,thershole=0.0005):
    normal_counts = train_df[emoji].value_counts(normalize=True)
    to_remove = normal_counts[normal_counts < thershole].index
    train_res = train_df[~train_df[emoji].isin(to_remove)]
    test_res = test_df[~test_df[emoji].isin(to_remove)]
    return train_res, test_res


def clean_improper_vocab():
    # informal spelling
    # wrong spelling
    return


def phrase_input():
    return


if __name__ == "__main__":
    train_texts, test_texts, train_tag_1, train_tag_2, test_tag_1, test_tag_2 = load_files()
    write_to_csv(train_texts, train_tag_1, train_tag_2, "training")
    write_to_csv(test_texts, test_tag_1, test_tag_2, "testing")
    train_df = pd.read_csv('data/NB/training_data.csv')
    test_df = pd.read_csv('data/NB/testing_data.csv')

    clean_class_1 = clean_low_freq_class(train_df, test_df, "emoji_1")
    clean_class_2 = clean_low_freq_class(train_df, test_df, "emoji_2")

