import csv
import nltk
from emoji_extractor import extract_emoji
from text_preprocessor import preprocess_text


def contain_emoji(emoji_extraction_result):
    # If there is no emoji in the text, we receive none
    return emoji_extraction_result is not None


def contain_meaningful_text(text_preprocessing_result):
    # If the text are all removed by our preprocessing pipeline (since they are not meaningful), we receive '' as result
    return len(text_preprocessing_result) != 0

def preprocess(start_point, amount_to_be_preprocessed):
    # The file to be processed
    target = 'data/emojitweets-01-04-2018.txt'

    # The line to start from
    # start from (start_point+1)th line

    wnl = nltk.WordNetLemmatizer()

    with open(target, 'r', encoding='utf-8') as f:

        pointer = 0
        with open('data/TEST_OUTPUT.csv', 'w', encoding='utf-8') as o:
            writer = csv.writer(o)
            writer.writerow(["text", "emoji_1", "emoji_2"])
            for i, line in enumerate(f):

                if pointer < start_point:
                    pointer += 1
                    continue

                if i == amount_to_be_preprocessed:
                    break

                emoji_extraction_result = extract_emoji(line)

                # Examine if there is any emoji in text
                if contain_emoji(emoji_extraction_result):
                    try:
                        raw_text = emoji_extraction_result[0]

                        # Preprocess text
                        preprocessed_text = preprocess_text(wnl, raw_text)

                        if contain_meaningful_text(preprocessed_text):
                            writer.writerow(
                                [preprocessed_text, emoji_extraction_result[1][0], emoji_extraction_result[1][1]])
                            print("line " + str(i + 1) + "is done.")
                    except Exception as e:
                        print(i + 1)  # This will print which line in txt having errors.
                        print(e)

                pointer += 1
    print('\nPreprocess is Done!')


if __name__ == "__main__":
    preprocess(0, 200000)
