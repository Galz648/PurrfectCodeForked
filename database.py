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
    return data.status_code == 201

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
