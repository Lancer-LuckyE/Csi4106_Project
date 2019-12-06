import csv
import pandas as pd
import numpy as np

def load_files():
    df = pd.read_csv('../data/TEST_OUTPUT_300000.csv', names=["text", "emoji_1", "emoji_2"])
    return df

def clean_low_freq_vocab():

    return


def clean_low_freq_class(df, emoji, thershole=0.0005):
    normal_counts = df[emoji].value_counts(normalize=True)
    to_remove = normal_counts[normal_counts < thershole].index
    res = df[~df[emoji].isin(to_remove)]
    return res


def phrase_input():
    return


if __name__ == "__main__":
    df = load_files()
    clean_class_1 = clean_low_freq_class(df, "emoji_1")
    clean_class_2 = clean_low_freq_class(df, "emoji_2")


