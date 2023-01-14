import os
from supabase import create_client, Client
from dotenv import load_dotenv
import praw

from analysis import preprocess

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)


# data = supabase.table("test").insert({"name":"Germany"}).execute()
redditClient = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    )

print(redditClient.user.me())
subreddit = redditClient.subreddit("ProgrammingBuddies")

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
    
def run_posts_stream_processing():
    print('running stream processing')
    for submission in subreddit.stream.submissions():
        if is_looking_for_buddies_flare(submission):
            print(f'post: {submission} \n')

            if not post_exists(submission.id):
                post_info = create_post_info(submission)
                is_created  = insert_post(post_info)
                print('post created: ', is_created)

            else:
                print('post already exists')




def main():
    run_posts_stream_processing()


if __name__ == "__main__":
    main()

