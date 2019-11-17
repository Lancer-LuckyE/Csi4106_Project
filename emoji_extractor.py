import re
import emoji
import csv

def clean_text(compiled_re_list, text):
    new_text = text
    for compiled_re in compiled_re_list:
        new_text = compiled_re.sub('', new_text)
    return new_text


def extract_emoji(text):

    # Remove '\n'
    tmp = clean_text([nxtln_re], text)

    # '😉' -> ':winking_face:'
    tmp = emoji.demojize(tmp)

    # Remove non-ascii '…'
    # Substitute ’ with ascii '
    tmp = clean_text([special_re_2], tmp)
    tmp = special_re.sub("'", tmp)

    emoji_lst = emoji_re.findall(tmp)

    # no emoji in sentence WTF
    if not (len(emoji_lst) > 0):
        return None

    emoji_count = len(emoji_lst)
    if emoji_count > 2:
        # Strategies required
        emoji_lst = emoji_lst[0:2]
    elif emoji_count == 2:
        emoji_lst = emoji_lst[0:2]
    elif emoji_count == 1:
        emoji_lst.append(emoji_lst[0])

    cleaned = clean_text([emoji_re], tmp)
    return cleaned, emoji_lst


if __name__ == "__main__":
    target = '/Users/mk6/CSI_4106_project/emojifydata-en/emojitweets-01-04-2018.txt'

    emoji_re = ':[a-zA-Z0-9-_]+:'
    emoji_re = re.compile(emoji_re)

    nxtln_re = '\\n'
    nxtln_re = re.compile(nxtln_re)

    special_re = '’'
    special_re = re.compile(special_re)

    special_re_2 = '…'
    special_re_2 = re.compile(special_re_2)

    with open(target, "r") as f:

        # for i in range(0, 1000):
        #     phrase = f.readline()
        #     processed, _ = extract_emoji(phrase)
        #     print('Processed: %s' % processed)

        with open('TEST_OUTPUT.csv', 'w') as o:
            writer = csv.writer(o)

            count = 0
            for line in f:
                if count == 1000:
                    break
                result = extract_emoji(line)
                if result is not None:
                    phrase = result[0]
                    lst = result[1]
                    writer.writerow([phrase, lst[0], lst[1]])
                    count = count + 1

            print(count)