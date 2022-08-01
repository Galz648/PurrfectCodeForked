import abc
import praw
import os
from dotenv import load_dotenv
from analysis import PostComparison

from db import DbService

load_dotenv()


    

    
class ApiClient:
    def __init__(self):
        self.api_config = { # TODO: move to a config file - use hydra to generate a config file
            'reddit_username': os.environ.get("reddit_username"),
            'reddit_password': os.environ["reddit_password"],
            "client_id": os.environ["client_id"],
            "client_secret": os.environ["client_secret"],
            "user_agent": "reddit_bot_v1.0" # move to env file, change user_agent
        }
    

    def getReddit(self) -> praw.Reddit:
        return praw.Reddit(**self.api_config)


class Service:
    def __init__(self, ApiClient: ApiClient, db: DbService, comparitor: PostComparison) -> None:
        self.apiClient = ApiClient
        self.db = db
        self.comparitor = comparitor


    # # db setup
    # db_strategy = SaveJsonToFileStrategy
    # db = DbProvider(db_strategy())
