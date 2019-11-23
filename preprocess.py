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


if __name__ == "__main__":

    target = 'data/emojitweets-01-04-2018.txt'

    amount_to_be_preprocessed = 10

    wnl = nltk.WordNetLemmatizer()
    with open(target, 'r', encoding='utf-8') as f:

        counter = 0
        with open('data/TEST_OUTPUT.csv', 'w', encoding='utf-8') as o:
            writer = csv.writer(o)
            for line in f:
                if counter == amount_to_be_preprocessed:
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
                            counter = counter + 1
                    except Exception as e:
                        print(counter)  # This will print which line in txt having errors.
                        print(e)
        print('Done!')
