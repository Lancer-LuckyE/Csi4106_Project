import csv
import nltk
from time import process_time
from emoji_extractor import extract_emoji
from text_preprocessor import preprocess_text


def contain_emoji(emoji_extraction_result):
    # If there is no emoji in the text, we receive none
    return emoji_extraction_result is not None


def contain_meaningful_text(text_preprocessing_result):
    # If the text are all removed by our preprocessing pipeline (since they are not meaningful), we receive '' as result
    return len(text_preprocessing_result) != 0


# The line to start from
# start from (start_point+1)th line
def preprocess(amount_to_be_preprocessed, start_point=0):
    # The file to be processed
    target = 'data/emojitweets-01-04-2018.txt'

    wnl = nltk.WordNetLemmatizer()

    start_time = process_time()
    with open(target, 'r', encoding='utf-8') as f:

        pointer = 0
        counter = 0 # count of successfully generated data
        with open('data/TEST_OUTPUT_%s.csv' % amount_to_be_preprocessed, 'w', encoding='utf-8', newline='') as o:
            writer = csv.writer(o)
            for line in f:

                if pointer < start_point:
                    pointer += 1
                    continue

                if counter == amount_to_be_preprocessed:
                    break

                emoji_extraction_result = extract_emoji(line)

                # Examine if there is any emoji in text
                if contain_emoji(emoji_extraction_result):

                    raw_text = emoji_extraction_result[0]

                    # Preprocess text
                    preprocessed_text = preprocess_text(wnl, raw_text)

                    # Processed text may be '', which means that there is no meaningful information in this text.
                    # We therefore do not put this entry in our result.
                    if contain_meaningful_text(preprocessed_text):
                        writer.writerow(
                            [preprocessed_text, emoji_extraction_result[1][0], emoji_extraction_result[1][1]])
                        counter += 1

                pointer += 1
    finish_time = process_time()
    print('Done! Processed %s records in %s seconds.' % (amount_to_be_preprocessed, round(finish_time - start_time, 2)))


if __name__ == '__main__':
    preprocess(round(3e5), 0)
    
#     Create 6 processes to run parallely 	   
#     p1 = mp.Process(target=preprocess, args=(50000, 0))         # 0 ~ 49999	
#     p2 = mp.Process(target=preprocess, args=(50000, 50000))     # 50000 ~ 99999	
#     p3 = mp.Process(target=preprocess, args=(50000, 100000))    # 100000 ~ 149999	
#     p4 = mp.Process(target=preprocess, args=(50000, 150000))    # 150000 ~ 199999	
#     p5 = mp.Process(target=preprocess, args=(50000, 200000))    # 200000 ~ 249999	
#     p6 = mp.Process(target=preprocess, args=(50000, 250000))    # 250000 ~ 299999	

#      # processes start	
#     p1.start()	
#     p2.start()	
#     p3.start()	
#     p4.start()	
#     p5.start()	
#     p6.start()	

#      # join the result	
#     p1.join()	
#     p2.join()	
#     p3.join()	
#     p4.join()	
#     p5.join()	
#     p6.join()	
#     print("done")
