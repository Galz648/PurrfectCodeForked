import abc
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


import re
import html
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

wl = WordNetLemmatizer()


def get_post_descriptions(posts: list):
    """
    Returns a list of post descriptions.
    """
    descriptions = []
    for post in posts:
        descriptions.append(post['body'])
    return descriptions


def get_post_titles(posts: list) -> list:
    """
    Returns a list of post titles.
    """
    titles = []
    for post in posts:
        titles.append(post['title'])
    return titles


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



class BertComparitor(SentencesComparison):
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
        text = emoji.unemojify(text)
        text = text.strip()
        return text

    def encodeText(self, text):
        """
        use the model to encode the text
        """
        return self.model.encode(text)

    def top_n_results(df: pd.DataFrame, df_so: pd.DataFrame,n=10) -> pd.DataFrame:
        """
        return top n results as dataframe
        """
        # get top 100 results
        top_n_results = df_so.head(n)
    
        # reset index inplace
        top_n_results.reset_index(inplace=True)
        result = df.loc[:,'text']        # for each row in dataframe add a new column with the value of 'level_0' as the index for df['text']
        # apply to both column
        top_n_results['text_level_0'] = top_n_results['level_0'].apply(lambda x: result.loc[x])
        top_n_results['text_level_1'] = top_n_results['level_1'].apply(lambda x: result.loc[x])

        # top_n_results['text_level_0'] = df['text'].iloc(0)
        # top_n_results
        # df['text'][100]
        id_results = df.loc[:,'id']
        top_n_results['id_level_0'] = top_n_results['level_0'].apply(lambda x: id_results.loc[x])
        top_n_results['id_level_1'] = top_n_results['level_1'].apply(lambda x: id_results.loc[x])

        # set author names
        # df['author']
        top_n_results['author_level_0'] = top_n_results['level_0'].apply(lambda x: df['author'].loc[x])
        top_n_results['author_level_1'] = top_n_results['level_1'].apply(lambda x: df['author'].loc[x])
        result = top_n_results[['text_level_0', 'text_level_1', 'id_level_0', 'id_level_1', 'similarity', 'author_level_0', 'author_level_1']]

        return result


class PostComparison:
    def __init__(self, strategy: SentencesComparison):
        self.strategy = strategy

    def compare(self, canidate: str, data: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.compare(canidate, data)


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


def remove_emojis(text):
    return emoji.replace_emoji(text)


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
    text = remove_emojis(text)
    text = decode_html_entities(text)
    text = simple_cleanup(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text


def process(df):
    df['text'] = df['title'] + ' ' + df['body']
    df['text'] = df['text'].apply(preprocess)
    return df



def encode_text(model, text):
    return model.encode(text)

def compare(canidate, target):
    print('Comparing...')
    df = process(canidate)
    df_target = process(target)
    model_name = 'bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    text_embeddings = model.encode(df['text'])
    target_embeddings = model.encode(df_target['text'])

    pairwise_similarities=cosine_similarity(text_embeddings, target_embeddings)

    # pairwise_similarities
    df_similarity_matrix = pd.DataFrame(pairwise_similarities, columns=range(pairwise_similarities.shape[0]), index=range(pairwise_similarities.shape[1]))

    # In[20]:
    s = df_similarity_matrix.unstack()
    so = s.sort_values(kind="quicksort", ascending=False)

    # so
    df_so = pd.DataFrame(so, columns=['similarity'])
    # df_so = pd.DataFrame(so, columns=['similarity'])
    # dataframe drop values with similarity around 1.0
    mask = df_so['similarity'].apply(lambda x: not math.isclose(x, 1.0, rel_tol=0.01))
    df_so = df_so[mask]
    # drop duplicates based on similarity value
    df_so = df_so.drop_duplicates(subset=['similarity'], keep='first')
    # sort by index
    df_so = df_so.sort_values(by='similarity', ascending=False)
    # df_so.head(50)
    return df, df_so


def top_n_results(df, df_so,n=10):
  #split index into two columns
  # df2 = pd.DataFrame(df_so['b'].tolist(), index=df_so.index)
  # result = df_so.head().index.tolist()

  # get top 100 results
  top_n_results = df_so.head(n)
  print("df_so.head(n): ", top_n_results)
  # reset index inplace
  top_n_results.reset_index(inplace=True)
  result = df.loc[:,'text']
  # for each row in dataframe add a new column with the value of 'level_0' as the index for df['text']
  # apply to both column
  top_n_results['text_level_0'] = top_n_results['level_0'].apply(lambda x: result.loc[x])
  top_n_results['text_level_1'] = top_n_results['level_1'].apply(lambda x: result.loc[x])

  # top_n_results['text_level_0'] = df['text'].iloc(0)
  # top_n_results
  # df['text'][100]
  id_results = df.loc[:,'id']
  top_n_results['id_level_0'] = top_n_results['level_0'].apply(lambda x: id_results.loc[x])
  top_n_results['id_level_1'] = top_n_results['level_1'].apply(lambda x: id_results.loc[x])

# set author names
# df['author']
  top_n_results['author_level_0'] = top_n_results['level_0'].apply(lambda x: df['author'].loc[x])
  top_n_results['author_level_1'] = top_n_results['level_1'].apply(lambda x: df['author'].loc[x])

  # add urls column from df
  top_n_results['url_level_0'] = top_n_results['level_0'].apply(lambda x: df['url'].loc[x])
  top_n_results['url_level_1'] = top_n_results['level_1'].apply(lambda x: df['url'].loc[x])

  result = top_n_results[['text_level_0', 'text_level_1', 'id_level_0', 'id_level_1', 'similarity', 'author_level_0', 'author_level_1', 'url_level_0', 'url_level_1']]

  return result
# top_n_results['id'] = df['id'].iloc[:]
# top_n_results
