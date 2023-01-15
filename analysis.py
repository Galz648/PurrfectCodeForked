import abc
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sentence_transformers import SentenceTransformer


import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer


# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

wl = WordNetLemmatizer()




def pairwise_similarity(corpus: list) -> np.ndarray:
    """
    Compares the descriptions of posts and returns a list of tuples.
    """

    vect = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf = vect.fit_transform(corpus)
    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity.toarray()


# def compareWithBert():
#     model_name = 'bert-base-nli-mean-tokens'
#     model = SentenceTransformer(model_name)

#     description_vecs = model.encode(post_titles)
#     print('description_vecs.shape: ',description_vecs.shape)
#     print('description_vecs: ',description_vecs)


class SentencesComparison(abc.ABC):

    @abc.abstractmethod
    def compare(self, canidate: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        pass



class BertComparitor(SentencesComparison): # TODO: 
    def __init__(self):
        self.model_name = 'bert-base-nli-mean-tokens' # move to config file
        self.model = SentenceTransformer(self.model_name)

    def compare(self, canidate: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
        """
        Compare the canidate to the target.
        """

    def process(self, text):
        """
        """

        text = self.preprocess(text)
        return text
    def preprocess(self, text):
        """
        prepare text column for sentence comparison
        """
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = html.unescape(text)
        text = text.strip()
        return text

    def encodeText(self, text):
        """
        use the model to encode the text
        """
        return self.model.encode(text)



def simple_cleanup(text):
  # Convert to lowercase.
    text = text.lower()

    # Remove everything but letters and spaces.
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove single letters.
    text = re.sub(r'(^\w\s)|(\s\w\s)|(\s\w$)', ' ', text)

    # Converge multiple spaces into one.
    text = re.sub(r'\s+', ' ', text)

  # Remove trailing and leading spaces.
    text = text.strip()

    return text

  # Since we found only 6 rows with emojis we decided to remove them.


def remove_urls(text):
    return re.sub('http(s?)://[^\s]+', ' ', text)


def decode_html_entities(text):
    return html.unescape(text)


def remove_stopwords(text):
    eng_stop_words = stopwords.words('english')
    non_stop_words = [
        word for word in text.split() if word not in eng_stop_words]
    return ' '.join(non_stop_words)



# This is a helper function to map NTLK position tags


def get_wordnet_pos(text):
    if text.startswith('J'):
        return wordnet.ADJ
    elif text.startswith('V'):
        return wordnet.VERB
    elif text.startswith('N'):
        return wordnet.NOUN
    elif text.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(text):
    # Split the text to words and get the part of speach (pos) of each
    # of the words (i.e. noun, verb, etc.)
    words = word_tokenize(text)
    words_with_pos = nltk.pos_tag(words)

    # Lemmatize each word.
    res = []
    for x in words_with_pos:
        word = x[0]
        pos = x[1]
        res.append(wl.lemmatize(word, get_wordnet_pos(pos)))

    return " ".join(res)


def preprocess(text):
    text = remove_urls(text)
    text = decode_html_entities(text)
    text = simple_cleanup(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text

