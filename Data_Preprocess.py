# All requirements are already downloaded
import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from emoji import UNICODE_EMOJI


class preprocessing:
    def __init__(self, sentence):
        self.__sentence = sentence
        self.__tokens = word_tokenize(sentence)
        self.__posTokens = nltk.pos_tag(self.__tokens)
        self.__posLemmas = [WordNetLemmatizer().lemmatize(t, w) for t, w in zip(self.__tokens, self.__wordnet_tags)]
        self.__wordnet_tags = [self.get_wordnet_pos(p[1]) for p in self.__posTokens]
        self.__split_sentences = nltk.sent_tokenize(sentence)
        self.__emojis = []

    def get_tokens(self):
        return self.__tokens

    def get_sentence(self):
        return self.__sentence

    def get_posTokens(self):
        return self.__posTokens

    def get_posLemmas(self):
        return self.__posLemmas

    def get_wordnet_tags(self):
        return self.__wordnet_tags

    def get_split_sentences(self):
        return self.__split_sentences

    def get_emojis(self):
        return self.__emojis

    def emoji_to_str(self):
        for i in range(len(self.__tokens)):
            if self.__tokens[i] in UNICODE_EMOJI:
                self.__emojis.append(self.__tokens[i])
                self.__tokens.remove(self.__tokens[i])

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.ADV

