import os
from supabase import create_client, Client
from dotenv import load_dotenv
import praw
load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

print(f"url: {url}, key: {key}")
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

for submission in subreddit.new(limit=10):
    print(submission)
    # data = supabase.table("posts").insert({"title":submission.title}).execute()


