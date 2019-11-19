# All requirements are already downloaded
import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from emoji import UNICODE_EMOJI


def tokens(text):
    return word_tokenize(text)


def pos_tokens(tokens):
    return nltk.pos_tag(tokens)


def get_wordnet_pos(tree_bank_tag):
    if tree_bank_tag.startswith('J'):
        return wordnet.ADJ
    elif tree_bank_tag.startswith('V'):
        return wordnet.VERB
    elif tree_bank_tag.startswith('N'):
        return wordnet.NOUN
    elif tree_bank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.ADV


def wordnet_tags(pos_tokens):
    result = []
    for p in pos_tokens:
        result.append(get_wordnet_pos(p[1]))
    return result


def pos_lemmas(tokens, wordnet_tags):
    result = []
    for t, w in zip(tokens, wordnet_tags):
        result.append(WordNetLemmatizer().lemmatize(t, w))
    return result


def split_sentence(text):
    return nltk.sent_tokenize(text)

