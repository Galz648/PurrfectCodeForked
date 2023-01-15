import os
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client, Client
import pandas as pd
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)

from analysis import preprocess

def is_looking_for_buddies_flare(submission) -> bool: # TODO: add type hinting
    if submission.link_flair_text is None:
        return False
    else:
        return submission.link_flair_text.lower() == "LOOKING FOR BUDDIES".lower()


def post_exists(post_id: str) -> bool:
    data = supabase.table("Posts").select("post_id").eq("post_id", post_id).execute()
    return len(data.data) > 0


def insert_post(post_info: dict) -> bool:
    data = supabase.table("Posts").insert(post_info).execute()

def create_post_info(submission) -> dict: # TODO: add type hinting
    body: str = submission.selftext
    text: str = submission.title + " " + body

    post_info = { # TODO: extract this into a function
        "title": submission.title,
        "url": submission.url,
        "body": body,
        "processed_text": preprocess(text),
        "created_time_utc": submission.created_utc,
        "post_id": submission.id,
        "flair": submission.link_flair_text,
        "author": submission.author.name,
    }
    return post_info



def loadPostsAsDataFrame() -> pd.DataFrame:
    data = supabase.table("Posts").select("*").execute()
    posts = data.data
    return pd.DataFrame(posts)



def upsertPostComparisons(comparisons: pd.DataFrame):
    for _, row in comparisons.iterrows():
        post_id_0 = row["id_level_0"] # TODO: rename to post_id_0
        post_id_1 = row["id_level_1"] # TODO: rename to post_id_1
        similarity = float(row["similarity"])

        # check if a comparison already exists
        data = supabase.table("Comparisons").select("*").eq("post_id_0", post_id_0).eq("post_id_1", post_id_1).execute()
        # check if a reverse comparison already exists
        data_reverse = supabase.table("Comparisons").select("*").eq("post_id_0", post_id_1).eq("post_id_1", post_id_0).execute()
        if len(data.data) == 0 and len(data_reverse.data) == 0:
            # insert comparison
            supabase.table("Comparisons").insert({
                "post_id_0": post_id_0,
                "post_id_1": post_id_1,
                "similarity": similarity
            }).execute()
        print(f"comparison between {post_id_0} and {post_id_1} already exists")


