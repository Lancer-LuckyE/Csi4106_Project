# All requirements are already downloaded
import nltk
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords


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


def preprocess_text(word_net_lemmatizer, single_text_entry):
    sentences = nltk.sent_tokenize(single_text_entry)

    processed_sentences = ''

    for sentence in sentences:
        # Tokenization
        tokens = nltk.word_tokenize(sentence.lower())  # Convert to lowercase

        # Remove non-alphanumeric characters
        tokens = [t for t in tokens if re.match('^[a-zA-Z]+$', t)]

        # Remove stopwords
        tokens = [t for t in tokens if t not in stopwords.words('english')]

        # POS Tagging
        pos_tokens = nltk.pos_tag(tokens)

        # Wordnet tags
        wordnet_tags = [get_wordnet_pos(p[1]) for p in pos_tokens]

        # POS-based lemmatization
        lemmas_pos = [word_net_lemmatizer.lemmatize(t, w) for t, w in zip(tokens, wordnet_tags)]

        processed_sentences += ' '.join(lemmas_pos)

    return processed_sentences
