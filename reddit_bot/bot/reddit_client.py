from dotenv import load_dotenv
load_dotenv()
import os
import praw





def get_client(subreddit: str = "ProgrammingBuddies"):
    return praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    ).subreddit(subreddit)