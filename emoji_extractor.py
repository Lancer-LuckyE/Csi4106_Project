import emoji
import re

'''
Used for extracting emoji from raw text.
'''

emoji_re = ':[a-zA-Z0-9-_]+:'
emoji_re = re.compile(emoji_re)


# Remove content matched by the given compiled re list
def clean_text(compiled_re_list, text):
    new_text = text
    for compiled_re in compiled_re_list:
        new_text = compiled_re.sub('', new_text)
    return new_text


# Extract two emoji from original text
# Also perform elementary pre-processing
def extract_emoji(text):
    # '😉' -> ':winking_face:'
    tmp = emoji.demojize(text)

    emoji_lst = emoji_re.findall(tmp)

    # no emoji in sentence WTF
    if len(emoji_lst) == 0:
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
