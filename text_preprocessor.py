import nltk
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from spellchecker import SpellChecker

'''
Used for removing meaningless information from text.
'''

# Reduce repeating letters to at most 2 times
def reduce_reqeated_letters(token):
    reduced_token = token[0]
    i = 1
    repeat_count = 1
    while i < len(token):
        print(i)
        if token[i] != reduced_token[-1]:
            reduced_token += token[i]
            repeat_count = 1
            i += 1
        elif token[i] == reduced_token[-1] and repeat_count < 2:
            reduced_token += token[i]
            repeat_count += 1
            i += 1
        else:
            repeat_count += 1
            i += 1
    return reduced_token

# Reduce repeating letters to at most 2 times
def reduce_repeat(token):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", token)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.ADV  # just use as default, for ADV the lemmatizer doesn't change anything

# process a single text entry
def preprocess_text(word_net_lemmatizer, single_text_entry):
    sentences = nltk.sent_tokenize(single_text_entry)
#     spell = SpellChecker()

    processed_sentences = ''

    for sentence in sentences:
        # Tokenization
        tokens = nltk.word_tokenize(sentence.lower())  # Convert to lowercase

        # Remove non-alphanumeric characters
        tokens = [t for t in tokens if re.match('^[a-zA-Z]+$', t)]

        # Remove stopwords
        tokens = [t for t in tokens if t not in stopwords.words('english')]

#         # Correction spelling (only tested on Naive Bayes)
#         for i in range(len(tokens)):
#             tokens[i] = spell.correction(reduce_repeat(tokens[i]))

        # POS Tagging
        pos_tokens = nltk.pos_tag(tokens)
        # Wordnet tags
        wordnet_tags = [get_wordnet_pos(p[1]) for p in pos_tokens]
        # POS-based lemmatization
        lemmas_pos = [word_net_lemmatizer.lemmatize(t, w) for t, w in zip(tokens, wordnet_tags)]

        processed_sentences += ' '.join(lemmas_pos)

    return processed_sentences
