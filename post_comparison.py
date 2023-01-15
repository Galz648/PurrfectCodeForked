from database import loadPostsAsDataFrame

import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import math

def encode_text(model_name, posts_df):
    model = SentenceTransformer(model_name)
    text_embeddings = model.encode(posts_df['processed_text'])
    return text_embeddings


def load_posts_as_dataframe():
    posts_df = loadPostsAsDataFrame()
    return posts_df


def calculate_similarities(text_embeddings):
    pairwise_similarities = cosine_similarity(text_embeddings)
    return pairwise_similarities


def create_similarity_matrix(pairwise_similarities):
    df_similarity_matrix = pd.DataFrame(pairwise_similarities, columns=range(pairwise_similarities.shape[0]), index=range(pairwise_similarities.shape[1]))
    s = df_similarity_matrix.unstack()
    so = s.sort_values(kind="quicksort", ascending=False)
    df_so = pd.DataFrame(so, columns=['similarity'])
    return df_so

def drop_duplicates_and_similarities(df_so):
    # drop values with similarity around 1.0 - this is the same post
    mask = df_so['similarity'].apply(lambda x: not math.isclose(x, 1.0, rel_tol=0.01))
    df_so = df_so[mask]

    # drop duplicates based on similarity value - this is an associative comparison (post_id_1,post_id_2) == (post_id_2,post_id_1)
    df_so = df_so.drop_duplicates(subset=['similarity'], keep='first')

    # sort by index
    df_so = df_so.sort_values(by='similarity', ascending=False)

    # take the posts with similarity > 0.85
    df_so = df_so[df_so['similarity'] > 0.85]
    return df_so

def get_similar_posts(model_name):
    posts_df = load_posts_as_dataframe()
    print(posts_df)

    print(posts_df)
    text_embeddings = encode_text(model_name, posts_df)
    pairwise_similarities = calculate_similarities(text_embeddings)
    df_so = create_similarity_matrix(pairwise_similarities)
    df_so = drop_duplicates_and_similarities(df_so)
    return df_so, posts_df








def generate_result_df(similarity_score_df, posts_df):
    results_summary = similarity_score_df.copy()
    # reset index inplace
    results_summary.reset_index(inplace=True)
    result = posts_df.loc[:,'processed_text']

    # set post texts
    results_summary['text_level_0'] = results_summary['level_0'].apply(lambda x: result.loc[x])
    results_summary['text_level_1'] = results_summary['level_1'].apply(lambda x: result.loc[x])

    # set post ids
    id_results = posts_df.loc[:,'post_id']
    results_summary['id_level_0'] = results_summary['level_0'].apply(lambda x: id_results.loc[x])
    results_summary['id_level_1'] = results_summary['level_1'].apply(lambda x: id_results.loc[x])

    # set author names
    results_summary['author_level_0'] = results_summary['level_0'].apply(lambda x: posts_df['author'].loc[x])
    results_summary['author_level_1'] = results_summary['level_1'].apply(lambda x: posts_df['author'].loc[x])
    result_df = results_summary[['text_level_0', 'text_level_1', 'id_level_0', 'id_level_1', 'similarity', 'author_level_0', 'author_level_1']]
    return result_df

def filter_identical_authors(result_df):
    result_df = result_df[result_df['author_level_0'] != result_df['author_level_1']]
    return result_df

# results_summary['id'] = df['id'].iloc[:]
# results_summary

def generate_filtered_similar_posts_df(model_name):
    similarity_score_df, posts_df = get_similar_posts(model_name)
    result_df = generate_result_df(similarity_score_df, posts_df)
    result_df = filter_identical_authors(result_df)
    return result_df
if __name__ == '__main__':
    model_name = 'bert-base-nli-mean-tokens'
    result_df = generate_filtered_similar_posts_df(model_name)
    print(result_df.columns)