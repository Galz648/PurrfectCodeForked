import os
from supabase import create_client, Client
from dotenv import load_dotenv
import praw
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
    return submission.link_flair_text.lower() == "LOOKING FOR BUDDIES".lower()

number_of_posts: int = 0
for submission in subreddit.stream.submissions():
    if submission.link_flair_text is None:
        continue
    elif is_looking_for_buddies_flare(submission):
        number_of_posts += 1
        print(f'number of posts: {number_of_posts}')
        print(f'New post: {submission} \n')
        print(f"submission: {redditClient.submission(id=submission.id)}")
    else:
        continue
    # show post properties



