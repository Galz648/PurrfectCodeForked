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


def run_posts_stream_processing():
    print('running stream processing')
    for submission in subreddit.stream.submissions():
        if is_looking_for_buddies_flare(submission):
            print(f'post: {submission} \n')

            body: str = submission.selftext
            text: str = submission.title + " " + body

            post_info = {
                "title": submission.title,
                "url": submission.url,
                "body": body,
                "processed_text": preprocess(text),
                "created": submission.created_utc,
                "post_id": submission.id,
                "flair": submission.link_flair_text,
                "author": submission.author.name,
            }
            

            data = supabase.table("Posts").select("post_id").eq("post_id", submission.id).execute()

            if len(data.data) == 0:        
                print('inserting post')
                print(f"post info: {post_info}")
                supabase.table("Posts").insert(
                    post_info
                ).execute()

            else:
                print('post already exists')
                print(f"data: {data}")




def main():
    run_posts_stream_processing()


if __name__ == "__main__":
    main()

